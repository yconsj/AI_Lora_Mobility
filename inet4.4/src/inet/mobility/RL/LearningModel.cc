//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 


#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omnetpp.h>




#include "LearningModel.h"
#include "tensorflow/lite/micro/system_setup.h"


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "inet/common/geometry/common/Coord.h"
#include "inet/mobility/contract/IMobility.h" // for accessing mobility
#include "SimpleRLMobility.h"
#include "InputState.h"
#include "StateLogger.h"  // Include the StateLogger header
#include <random>  // For random sampling
#include "modelfiles/policy_net_model.h"



namespace inet {

Define_Module(LearningModel);

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;


TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Define buffers for input and output tensors
float* model_input_buffer = nullptr;
float* model_output_buffer = nullptr;

// Define dimensions
const int kInputHeight = 1;
const int kInputWidth = 6; // Change based on your features
const int kOutputHeight = 1;
const int kOutputWidth = 2; // Change based on your model output

constexpr int kTensorArenaSize = 2000;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];


LearningModel::LearningModel()
{
    // stub
}

void LearningModel::initialize()
{

    EV_TRACE << "initializing LearningModel " << omnetpp::endl;


    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
      EV << "Model provided is schema version" << model->version() << " not equal to supported version " << TFLITE_SCHEMA_VERSION << omnetpp::endl;
      return;
    }

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // TODO: Might need to Validate input tensor dimensions for multi-dimensional input
    // TODO: And also might need to validate output tensor dimensions.

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      EV << "AllocateTensors() failed" << omnetpp::endl;
      return;
    }

    // Retrieve input tensor
   model_input = interpreter->input(0);
   if ((model_input->dims->size != 2) ||
       (model_input->dims->data[0] != kInputHeight) ||
       (model_input->dims->data[1] != kInputWidth) ||
       (model_input->type != kTfLiteFloat32)) {
       // Handle incorrect input tensor parameters
       EV << "wrong input dimensions" << omnetpp::endl;
       //return;
   }
   model_input_buffer = model_input->data.f; // Pointer to input data

   // Retrieve output tensor
   model_output = interpreter->output(0);
   if ((model_output->dims->size != 2) ||
       (model_output->dims->data[0] != kOutputHeight) ||
       (model_output->dims->data[1] != kOutputWidth) ||
       (model_output->type != kTfLiteFloat32)) {
       // Handle incorrect output tensor parameters
       EV << "wrong output dimensions" << omnetpp::endl;
       //return;
   }

}

void LearningModel::fetchStateLoggerModule() {
    // Fetch the StateLogger module as a submodule of LearningModel
    stateLogger = check_and_cast<StateLogger*>(getSubmodule("stateLogger"));
    if (stateLogger == nullptr) {
        throw cRuntimeError("StateLogger module not found in LearningModel.");
    }
}

SimpleRLMobility* LearningModel::getMobilityModule() {
    // Fetch the mobility module, which is the parent of LearningModel
    cModule* mobilityModule = getParentModule(); // mobility is the parent of LearningModel
    if (!mobilityModule || strcmp(mobilityModule->getName(), "mobility") != 0) {
        throw cRuntimeError("The parent module is not the expected 'mobility' module.");
    }

    // Verify that the mobility module is an instance of SimpleRLMobility
    SimpleRLMobility* mobility = check_and_cast<SimpleRLMobility*>(mobilityModule);
    if (!mobility) {
        throw cRuntimeError("Failed to cast the mobility module to SimpleRLMobility.");
    }
    return mobility;
}


// Method to get the position (coordinate) from the SimpleRLMobility module
Coord LearningModel::getCoord() {
    // Fetch the mobility module, which is the parent of LearningModel
    SimpleRLMobility* mobility = getMobilityModule();
    // Return the current position of the mobility module
    return mobility->getCurrentPosition();

}


void LearningModel::setPacketInfo(double rssi, double snir, double nReceivedPackets, simtime_t timestamp) {
    currentState.latestPacketRSSI = rssi;
    currentState.latestPacketSNIR = snir;
    currentState.latestPacketTimestamp = timestamp;
    currentState.numReceivedPackets = nReceivedPackets;
}

int LearningModel::getReward() {
    static double lastStateNumberOfPackets = 0.0;
    double reward = (currentState.numReceivedPackets - lastStateNumberOfPackets) * 10;
    lastStateNumberOfPackets = currentState.numReceivedPackets;

    if (getMobilityModule()->isNewGridPosition()) {
        EV << "New grid!" << omnetpp::endl;
        reward += 1;
    }

    return reward;
}

InputState LearningModel::normalizeInputState(InputState state) {
    InputState normalizedState;
    normalizedState.latestPacketRSSI = state.latestPacketRSSI / 255.0;
    normalizedState.latestPacketSNIR = state.latestPacketSNIR / 100.0;
    normalizedState.latestPacketTimestamp = state.latestPacketTimestamp.dbl() / (60 * 60 * 24.0); // one day in seconds
    normalizedState.numReceivedPackets = (state.numReceivedPackets) / (86400.0 / 500.0) * 2.0; //
    normalizedState.currentTimestamp = state.currentTimestamp.dbl() / (60 * 60 * 24.0);;
    normalizedState.coord.x = state.coord.x / 3000.0;
    return normalizedState;
}


int LearningModel::pollModel()
{

    // get remaining state info:
    currentState.currentTimestamp = simTime();
    currentState.coord = getCoord();

    // Ensure that StateLogger is fetched and initialized
    int reward = getReward();
    if (!stateLogger) {
        fetchStateLoggerModule();
    }

    InputState normalizedState = normalizeInputState(currentState);

    int output = invokeModel(normalizedState);
    stateLogger->logStep(normalizedState, output, reward);

    return output;
    /*
    return 2;
    */
}

void LearningModel::PrintOutput(float x_value, float y_value) {
    EV_INFO << "x: " << x_value << ", y:" << y_value << omnetpp::endl;
}

int LearningModel::invokeModel(InputState state) {

    if (interpreter == nullptr) {
        EV << "Interpreter is not initialized." << omnetpp::endl;
        return -1;
    }
    if (model_input == nullptr) {
            EV << "Model output is not initialized." << omnetpp::endl;
            return -1;
    }
    if (model_output == nullptr) {
        EV << "Model output is not initialized." << omnetpp::endl;
        return -1;
    }
    if (model_output->data.f == nullptr || model_output->bytes == 0) {
        EV << "Model output buffer is not valid." << omnetpp::endl;
        return -1;
    }
    // Check the size of the output
    EV << "Model output bytes: " << model_output->bytes << omnetpp::endl;
    if (model_output->bytes <= 0) {
        EV << "Model output size is zero or negative." << omnetpp::endl;
        return -1;
    }

    // Insert input data for the model from state values
    model_input->data.f[0] = state.latestPacketRSSI;
    model_input->data.f[1] = state.latestPacketSNIR;
    model_input->data.f[2] = state.latestPacketTimestamp.dbl();
    model_input->data.f[3] = state.numReceivedPackets;
    model_input->data.f[4] = state.currentTimestamp.dbl();
    model_input->data.f[5] = state.coord.x;
    // model_input->data.f[6] = state.coord.y;

    size_t num_inputs = kInputWidth;
    // Place the quantized input in the model's input tensor
    for (size_t i = 0; i < num_inputs; ++i) { // Adjust based on your actual number of inputs
        model_input->data.f[i] = (model_input_buffer[i]);
        EV << "model_input->data.f"  << model_input->data.f[i] << omnetpp::endl;
    }

    // Run the inference with the model
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        EV << "Invoke failed" << omnetpp::endl;
        return -1;
    }

    // Obtain the quantized output from model's output tensor
    size_t num_outputs = model_output->bytes / sizeof(float); // Adjust based on the number of output elements
    // EV << "output bytes" <<num_outputs << omnetpp::endl;

    // Sample one output based on the weights
    // model uses softmax function on output, so it is already weighted and sums to 1.
    float random_value = static_cast<float>(rand()) / RAND_MAX; // Random value in [0, 1)
    float cumulative_weight = 0.0f;
    int selected_index = 0;
    for (size_t i = 0; i < num_outputs; ++i) {
        cumulative_weight += model_output->data.f[i];
        if (random_value < cumulative_weight) {
            selected_index = i;
            break;
        }
    }
    EV << "Selected output index: " << selected_index << " with weight: " << model_output->data.f[selected_index] << "\n";

    return selected_index; // Return the selected index

}





LearningModel::~LearningModel() {
    // TODO Auto-generated destructor stub
}




} /* namespace inet */
