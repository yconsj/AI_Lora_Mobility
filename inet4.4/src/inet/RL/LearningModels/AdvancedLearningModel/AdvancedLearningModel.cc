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


#include "AdvancedLearningModel.h"

namespace inet {

Define_Module(AdvancedLearningModel);
const tflite::Model* model = nullptr;

// This pulls in all the operation implementations we need.
// NOLINTNEXTLINE(runtime-global-variables)
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Define buffers for input and output tensors
float* model_input_buffer = nullptr;
float* model_output_buffer = nullptr;

// Define dimensions

const int kInputHeight = 1;
const int kInputWidth = 3 * number_of_nodes;
const int kOutputHeight = 1;
const int kOutputWidth = 5;

constexpr int kTensorArenaSize = const_g_model_length;
// Keep aligned to 16 bytes for CMSIS

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

AdvancedLearningModel::AdvancedLearningModel()
{
    std::vector<uint8_t> model_data(const_g_model_length, 0);
    while (recent_packets.size() < recent_packets_length) {
        recent_packets.push_back(-1.0);
    }

}

AdvancedLearningModel::~AdvancedLearningModel() {
    // Clean up dynamically allocated resources
    delete interpreter; // Delete interpreter, if allocated
    interpreter = nullptr;

    // No need to delete model_input and model_output as they are managed by TensorFlow Lite
    model_input = nullptr;
    model_output = nullptr;
}

void AdvancedLearningModel::initialize(int stage)
{

    cSimpleModule::initialize(stage);
    if (stage == 0)
    {

        EV << "initializing AdvancedLearningModel " << endl;

        // Load the model data from a file
        model_data = ReadModelFromFile(model_file_path);


        model = tflite::GetModel(model_data.data());
        interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);

        // After loading the model
        if (model == nullptr) {
            throw cRuntimeError("AdvancedLearningModel.cc:Failed to load model.");
        } else {
            //EV << "Model loaded successfully. Number of subgraphs: " << model->subgraphs()->size() << omnetpp::endl;
        }

        if (model->version() != TFLITE_SCHEMA_VERSION) {
          //EV << "Model provided is schema version" << model->version() << " not equal to supported version " << TFLITE_SCHEMA_VERSION << omnetpp::endl;
          throw cRuntimeError("AdvancedLearningModel.cc: Model provided's schema version is not equal to supported version. ");
        }


        // TODO: Might need to Validate input tensor dimensions for multi-dimensional input
        // TODO: And also might need to validate output tensor dimensions.
        // Allocate memory from the tensor_arena for the model's tensors.
        TfLiteStatus allocate_status = interpreter->AllocateTensors();
        if (allocate_status != kTfLiteOk) {
          throw cRuntimeError("AdvancedLearningModel.cc: AllocateTensors() failed with error code: " );
        }

        // Retrieve input tensor
       model_input = interpreter->input(0);
       if ((model_input->dims->size != 2) ||
           (model_input->dims->data[0] != kInputHeight) ||
           (model_input->dims->data[1] != kInputWidth) ||
           (model_input->type != kTfLiteFloat32)) {
           EV << "size, data0, data1" << omnetpp::endl;
           EV << model_input->dims->size << omnetpp::endl;
           EV << model_input->dims->data[0] << omnetpp::endl;
           EV << model_input->dims->data[1] << omnetpp::endl;

           throw cRuntimeError("AdvancedLearningModel.cc: wrong input dimensions");
       }
       model_input_buffer = model_input->data.f; // Pointer to input data

       // Retrieve output tensor
       model_output = interpreter->output(0);
       if ((model_output->dims->size != 2) ||
           (model_output->dims->data[0] != kOutputHeight) ||
           (model_output->dims->data[1] != kOutputWidth) ||
           (model_output->type != kTfLiteFloat32)) {
           EV << "size, data0, data1" << omnetpp::endl;
           EV << model_output->dims->size << omnetpp::endl;
           EV << model_output->dims->data[0] << omnetpp::endl;
           EV << model_output->dims->data[1] << omnetpp::endl;
           throw cRuntimeError("AdvancedLearningModel.cc: wrong output dimensions");
       }
       EV << " AdvancedLearningModel EVinit stage 0 end " << endl;

    }
   else if (stage == INITSTAGE_LAST) {
       EV << "initstage after mobility. fetch node values" <<omnetpp::endl;
       nodeValueInitialization();

       send_interval_norm_factor = MAX_SEND_INTERVAL;
       max_cross_distance = getMobilityModule()->getMaxCrossDistance();


   }
}


void AdvancedLearningModel::nodeValueInitialization() {
    std::vector<cModule *> loRaNodes;
    cModule *network = getSimulation()->getSystemModule();

    const char* lora_mod_str = "loRaNodes";
    int number_of_lora_nodes_in_scenario = network->getSubmoduleVectorSize(lora_mod_str);
    if (number_of_lora_nodes_in_scenario != number_of_nodes) {
        throw cRuntimeError("AdvancedLearningModel: Number of LoRa nodes does not match the expected number of nodes in the scenario");
    }
    for (int i = 0; i < number_of_lora_nodes_in_scenario; i++) {
        cModule *lora_node_module = network->getSubmodule(lora_mod_str, i);
        loRaNodes.push_back(lora_node_module);
    }


    EV << "loRaNodes.size() = " << loRaNodes.size() << endl;
    nodes.resize(loRaNodes.size(), nullptr);
    node_positions.resize(loRaNodes.size(), Coord(0,0,0));
    expected_send_times.resize(loRaNodes.size(), 0);
    send_intervals.resize(loRaNodes.size(), 0);
    number_of_received_packets_per_node.resize(loRaNodes.size(), 0);

    for (size_t i = 0; i < loRaNodes.size(); ++i) {
        cModule *loRaNode = loRaNodes[i];
        int node_index = loRaNode->getIndex();
        nodes[node_index] = loRaNode;
        EV << "node_index: " << node_index << endl;

        // Retrieve LoRa application
        auto *loRaApp = check_and_cast<cModule *>(loRaNode->getSubmodule("app", 0));
        if (!loRaApp) {
            throw cRuntimeError("Invalid LoRa application for node %d", node_index);
        }

        // access & store node properties

        // 'timeToFirstPacket' and 'timeToNextPacket' time parameters
        simtime_t timeToFirstPacket = loRaApp->par("timeToFirstPacket");
        simtime_t timeToNextPacket = loRaApp->par("timeToNextPacket");
        expected_send_times[node_index] = timeToFirstPacket;
        send_intervals[node_index] = timeToNextPacket;
        auto *mobility = check_and_cast<StationaryMobility *>(loRaNode->getSubmodule("mobility"));

        // get stationary lora node positions
        if (!mobility) {
            throw cRuntimeError("Error, node missing mobility module in LoRa application for node %d", node_index);
        }
        node_positions[node_index] = mobility->getCurrentPosition();
    }
}





StateLogger* AdvancedLearningModel::getStateLoggerModule() {
    // Retrieve the network module directly
    cModule* network = getSimulation()->getSystemModule();
    if (network == nullptr) {
        throw cRuntimeError("Failed to find the network module.");
    }

    // Fetch the StateLogger module as a submodule of the network
    StateLogger* stateLogger = check_and_cast<StateLogger*>(network->getSubmodule("stateLogger"));
    if (stateLogger == nullptr) {
        throw cRuntimeError("StateLogger module not found in the network.");
    }

    return stateLogger;
}

AdvancedRLMobility* AdvancedLearningModel::getMobilityModule() {
    // Fetch the mobility module, which is the parent of AdvancedLearningModel
    cModule* mobilityModule = getParentModule(); // mobility is the parent of AdvancedLearningModel
    if (!mobilityModule || strcmp(mobilityModule->getName(), "mobility") != 0) {
        throw cRuntimeError("The parent module is not the expected 'mobility' module.");
    }

    // Verify that the mobility module is an instance of AdvancedRLMobility
    AdvancedRLMobility* mobility = check_and_cast<AdvancedRLMobility*>(mobilityModule);
    if (!mobility) {
        throw cRuntimeError("Failed to cast the mobility module to AdvancedRLMobility.");
    }
    return mobility;
}

// Method to get the position (coordinate) from the AdvancedRLMobility module
const Coord AdvancedLearningModel::getCoord() {
    // Fetch the mobility module, which is the parent of AdvancedLearningModel
    AdvancedRLMobility* mobility = getMobilityModule();
    // Return the current position of the mobility module
    const Coord pos = mobility->getCurrentPosition();
    return pos;
}

double AdvancedLearningModel::calculateNormalizedAngle(const Coord& coord1, const Coord& coord2) {
    // Calculate the difference in coordinates
   double dx = coord2.x - coord1.x;
   double dy = coord2.y - coord1.y;

   // Calculate the angle in radians
   double angleRadians = atan2(dy, dx);

   // Convert radians to degrees
   double angleDegrees = angleRadians * (180.0 / M_PI);

   // Ensure the angle is in the range [0, 360]
   if (angleDegrees < 0) {
       angleDegrees += 360.0;
   }

   // Normalize the angle to [0, 1]
   double normalizedAngle = angleDegrees / 360.0;

   return normalizedAngle;
}

// Function to read the model from a file
std::vector<uint8_t> AdvancedLearningModel::ReadModelFromFile(const char* filename) {
    std::vector<uint8_t> model_data;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening model file: " << filename << std::endl;
        throw cRuntimeError("AdvancedLearningModel.cc: Error reading the model from file. ");
    }

    // Read the entire file into the model_data vector
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    if (file_size == -1) {
        throw cRuntimeError("AdvancedLearningModel.cc: Error getting file size.");
    }

    model_data.resize(file_size);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(model_data.data()), file_size);
    // Check if the read was successful
     if (!file) {
         std::cerr << "Error reading model data from file: " << filename << std::endl;
         throw cRuntimeError("AdvancedLearningModel.cc: Error reading the model data from file.");
     }

    return model_data;
}


int AdvancedLearningModel::pollModel()
{
    EV << "pollmodel chk" << endl;

    int output = invokeModel();
    return output;

}


int AdvancedLearningModel::selectOutputIndex(float random_choice_probability, const TfLiteTensor* model_output, size_t num_outputs, bool deterministic) {
    if (num_outputs == 0) {
        EV << "Error: num_outputs is zero, cannot determine output index.\n";
        return -1;
    }
    int selected_index = -1;

    for (int i = 0; i < num_outputs; i++) {
        EV << "model output [" << i << "] weight: " << model_output->data.f[i] << endl;

    }

    if (deterministic) {
        float output = *(model_output->data.f);
        // Deterministic mode: choose the index with the highest weight
        selected_index = std::distance(model_output->data.f,
                          std::max_element(model_output->data.f, model_output->data.f + num_outputs));
        EV << "Deterministic mode: selected output index: " << selected_index
           << " with highest weight: " << model_output->data.f[selected_index] << endl;
        return selected_index;
    }

    // Stochastic mode
    float random_value = rand() / RAND_MAX; // Random value in [0, 1)
    if (random_choice_probability > random_value) {
        // Choose a random integer from 0 to num_outputs - 1
        selected_index = rand() % num_outputs;
        EV << "By random choice, selected output index: " << selected_index
           << " with weight: " << model_output->data.f[selected_index] << "\n";
        return selected_index;
    } else {
        random_value = rand() / RAND_MAX; // Take a new random value in [0, 1)
        // Sample one output based on the weights
        // Model uses softmax function on output, so it is already weighted and sums to 1.
        float cumulative_weight = 0.0f;
        for (size_t i = 0; i < num_outputs; ++i) {
            cumulative_weight += model_output->data.f[i];
            if (random_value < cumulative_weight) {
                selected_index = i;
                break;
            }
        }
        EV << "Selected output index: " << selected_index
           << " with weight: " << model_output->data.f[selected_index] << "\n";
        return selected_index;
    }
}



void AdvancedLearningModel::setPacketInfo(int index) {
    EV << "Packet Info! Received Packet" << endl;
    number_of_received_packets_per_node[index] += 1;
    // update recent packets
    while (recent_packets.size() >= recent_packets_length) {
        recent_packets.pop_front();
    }
    recent_packets.push_back(index);

    // update expected send time
    expected_send_times[index] = simTime() + send_intervals[index];
    return;
}

int AdvancedLearningModel::invokeModel() {
    if (interpreter == nullptr) {
        EV << "Interpreter is not initialized." << omnetpp::endl;
        return -1;
    }
    if (model == nullptr) {
        EV << "Model is not initialized." << omnetpp::endl;
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

/*
    // prepare input:
    state = (
        normalized_expected_send_time +
        normalized_node_distances +
        normalized_node_directions +
        onehot_encoded_recent_packets
    )
*/

    Coord gw_pos = getCoord();

    std::vector<float> normalized_expected_send_time;
    std::vector<float> node_distances;
    std::vector<float> normalized_node_distances;
    std::vector<float> normalized_node_directions;
    for (int i = 0; i < nodes.size(); i++) {

        float delta_time = expected_send_times[i].dbl() - simTime().dbl();
        if (delta_time < 0) { // update expected send time
            expected_send_times[i] += (-delta_time) + send_intervals[i];
        }
        float time = (expected_send_times[i].dbl() - simTime().dbl()) / send_interval_norm_factor;
        normalized_expected_send_time.push_back(time);

        Coord node_pos = node_positions[i];
        float node_distance = gw_pos.distance(node_pos);
        node_distances.push_back(node_distance); // used for logging
        float norm_distance = node_distance / max_cross_distance;
        normalized_node_distances.push_back(norm_distance);

        float norm_direction = calculateNormalizedAngle(getCoord(), node_pos);
        normalized_node_directions.push_back(norm_direction);
    }


    // Insert input data for the model from state values
    // Inserting the normalized expected send times
    int index = 0;  // Start with index 0 in model_input->data.f
    for (int i = 0; i < normalized_expected_send_time.size(); ++i) {
        EV << "expected_send_time = " << normalized_expected_send_time[i] << endl;
        model_input->data.f[index] = normalized_expected_send_time[i];
        index++;
    }

    // Inserting the normalized node distances
    for (int i = 0; i < normalized_node_distances.size(); ++i) {
        EV << "distances = " << normalized_node_distances[i] << endl;
        model_input->data.f[index] = normalized_node_distances[i];
        index++;
    }

    // Inserting the normalized node directions
    for (int i = 0; i < normalized_node_directions.size(); ++i) {
        EV << "directions = " << normalized_node_directions[i] << endl;
        model_input->data.f[index] = normalized_node_directions[i];
        index++;
    }


    // Place the quantized input in the model's input tensor
    for (size_t i = 0; i < kInputWidth; ++i) { // Adjust based on your actual number of inputs
        //model_input->data.f[i] = (model_input_buffer[i]);
        EV << "model_input->data.f" << i << ": "  << model_input->data.f[i] << omnetpp::endl;
        if (!std::isfinite(model_input->data.f[i])) {
            EV << "Invalid input detected at index " << i << ": " << model_input->data.f[i] << omnetpp::endl;
            throw cRuntimeError("NaN or inf detected in model input");
        }
    }


    // Run the inference with the model

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        EV << "Invoke failed" << omnetpp::endl;
        return -1;
    }


    // Obtain the quantized output from model's output tensor
    size_t num_outputs = model_output->bytes / sizeof(float); // Adjust based on the number of output elements



    bool deterministic = true;
    int selected_index = selectOutputIndex(random_choice_probability, model_output, num_outputs, deterministic);

    StateLogger* stateLogger = getStateLoggerModule();

    // gw position
    // distances
    // packets
    // action
    // time
    stateLogger->logStep(getCoord(), node_distances, number_of_received_packets_per_node, simTime().dbl(), selected_index);

    return selected_index;
}


} /* namespace inet */
