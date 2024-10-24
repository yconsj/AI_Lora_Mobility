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
#include "modelfiles/model.h"
#include "inet/common/geometry/common/Coord.h"
#include "inet/mobility/contract/IMobility.h" // for accessing mobility
#include "InputState.h"
#include "StateLogger.h"  // Include the StateLogger header
#include <random>  // For random sampling



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
const int kOutputWidth = 3; // Change based on your model output

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
    // TODO: And also might need to validate output tensor dimensiosn.

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
   model_output_buffer = model_output->data.f; // Pointer to output data
}

void LearningModel::fetchStateLoggerModule() {
    // Fetch the StateLogger module as a submodule of LearningModel
    stateLogger = check_and_cast<StateLogger*>(getSubmodule("stateLogger"));

    if (stateLogger == nullptr) {
        throw cRuntimeError("StateLogger module not found in LearningModel.");
    }
}

// Method to get the position (coordinate) from the SimpleRLMobility module
Coord LearningModel::getCoord() {
    // Fetch the mobility module, which is the parent of LearningModel
    cModule* mobilityModule = getParentModule(); // mobility is the parent of LearningModel
    EV << "Parent module name: " << mobilityModule->getName() << "\n";
    if (!mobilityModule || strcmp(mobilityModule->getName(), "mobility") != 0) {
        throw cRuntimeError("The parent module is not the expected 'mobility' module.");
    }

    // Verify that the mobility module implements the IMobility interface
    inet::IMobility* mobility = check_and_cast<inet::IMobility*>(mobilityModule);
    if (!mobility) {
        throw cRuntimeError("Failed to cast the mobility module to IMobility.");
    }

    // Return the current position of the mobility module
    return mobility->getCurrentPosition();
    /*
    return Coord(0, 0, 0);
    */
}


void LearningModel::logPacketInfo(double rssi, double snir, int nReceivedPackets, simtime_t timestamp) {
    currentState.latestPacketRSSI = rssi;
    currentState.latestPacketSNIR = snir;
    currentState.latestPacketTimestamp = timestamp;
    currentState.numReceivedPackets = nReceivedPackets;
}

int LearningModel::getReward() {
    return 0;
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
    int output = invokeModel();
    stateLogger->logStep(currentState, output, reward);

    return output;
    /*
    return 2;
    */
}

void LearningModel::PrintOutput(float x_value, float y_value) {
    EV_INFO << "x: " << x_value << ", y:" << y_value << omnetpp::endl;
}

int LearningModel::invokeModel() {
    EV << "LearningModel.invokeModel()" << omnetpp::endl;

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

    float scale = model_input->params.scale;
    int32_t zero_point = model_input->params.zero_point;
    // Prepare input data for the model from currentState values
    model_input_buffer[0] = currentState.latestPacketRSSI / scale + zero_point;  // Quantized RSSI
    model_input_buffer[1] = currentState.latestPacketSNIR / scale + zero_point; // Quantized SNIR
    model_input_buffer[2] = currentState.latestPacketTimestamp.dbl() / scale + zero_point; // Quantized timestamp
    model_input_buffer[3] = currentState.currentTimestamp.dbl() / scale + zero_point; // Quantized current timestamp
    model_input_buffer[4] = currentState.coord.x / scale + zero_point;  // Quantized x coordinate
    model_input_buffer[5] = currentState.coord.y / scale + zero_point; // Quantized y coordinate

    size_t num_inputs = 6;
    // Place the quantized input in the model's input tensor
    for (size_t i = 0; i < num_inputs; ++i) { // Adjust based on your actual number of inputs
        model_input->data.int8[i] = static_cast<int8_t>(model_input_buffer[i]);
    }

    // Run the inference with the model
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        EV << "Invoke failed" << omnetpp::endl;
        return -1;
    }
    /*
    size_t float_size = sizeof(float);
    size_t num_floats = model_output->bytes / float_size;
    std::vector<float> output_values(num_floats); // Adjust based on output size

    // Check if the number of floats is valid
    if (model_output->bytes < float_size) {
        EV << "Error: model_output->bytes is less than sizeof(float). Expected at least "
           << float_size << " but got " << model_output->bytes << omnetpp::endl;
        return -1; // Handle the error appropriately
    }

    EV <<  "data.f " << model_output->data.f << omnetpp::endl;
    EV <<  "bytes " << model_output->bytes << omnetpp::endl;
    EV <<  "ratio " << num_floats << omnetpp::endl;
    std::memcpy(output_values.data(), model_output->data.f, model_output->bytes); // Copy output data to vector

    */


    // Obtain the quantized output from model's output tensor
    size_t num_outputs = model_output->bytes / sizeof(int8_t); // Adjust based on the number of output elements
    std::vector<int8_t> y_quantized(num_outputs);
    std::memcpy(y_quantized.data(), model_output->data.int8, model_output->bytes);

    // Dequantize the output
    std::vector<float> output_values(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        output_values[i] = (y_quantized[i] - zero_point) * scale;
    }

    // Implement sampling logic based on output_values
    float total_weight = std::accumulate(output_values.begin(), output_values.end(), 0.0f);
    std::vector<float> weights(output_values.size());
    for (size_t i = 0; i < output_values.size(); ++i) {
        weights[i] = output_values[i] / total_weight;
    }

    // Sample one output based on the weights
    float random_value = static_cast<float>(rand()) / RAND_MAX; // Random value in [0, 1)
    float cumulative_weight = 0.0f;
    int selected_index = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative_weight += weights[i];
        if (random_value < cumulative_weight) {
            selected_index = i;
            break;
        }
    }
    EV << "Selected output index: " << selected_index << " with weight: " << weights[selected_index] << "\n";

    return selected_index; // Return the selected index

    //return -1;
        /*
    */
}


void LearningModel::Test() {
/*
    EV << "LearningModel.Test()" << omnetpp::endl;
    // Calculate an x value to feed into the model. We compare the current
   // inference_count to the number of inferences per cycle to determine
   // our position within the range of possible x values the model was
   // trained on, and use this to calculate a value.

    float position = static_cast<float>(inference_count) /
                    static_cast<float>(kInferencesPerCycle);
    float x = position * kXrange;

    // Quantize the input from floating-point to integer
    int8_t x_quantized = x / input->params.scale + input->params.zero_point;
    // Place the quantized input in the model's input tensor
    input->data.int8[0] = x_quantized;



    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
       EV << "Invoke failed on x: " << static_cast<double>(x) << "\n" << omnetpp::endl;
     return;
    }

    // Obtain the quantized output from model's output tensor
    int8_t y_quantized = output->data.int8[0];
    // Dequantize the output from integer to floating-point
    float y = (y_quantized - output->params.zero_point) * output->params.scale;

    // Output the results. A custom HandleOutput function can be implemented
    // for each supported hardware target.
    PrintOutput(x, y);

    // Increment the inference_counter, and reset it if we have reached
    // the total number per cycle
    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;
*/
}


LearningModel::~LearningModel() {
    // TODO Auto-generated destructor stub
}




} /* namespace inet */
