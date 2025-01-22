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


#include "BasicLearningModel.h"




namespace inet {

Define_Module(BasicLearningModel);
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
const int kInputWidth = 5; // Change based on your features
const int kOutputHeight = 1;
const int kOutputWidth = 3; // Change based on your model output

constexpr int kTensorArenaSize = const_g_model_length;
// Keep aligned to 16 bytes for CMSIS

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

BasicLearningModel::BasicLearningModel()
{
    lastStateNumberOfPackets = 0.0;
    currentState = InputStateBasic();

    model = nullptr;
    std::vector<uint8_t> model_data(const_g_model_length, 0);
    packet_reward = -1.0;
    exploration_reward = -1.0;
}

BasicLearningModel::~BasicLearningModel() {
    // Clean up the model interpreter
    if (interpreter) {
        delete interpreter; // Assuming interpreter is dynamically allocated
        interpreter = nullptr; // Avoid dangling pointer
    }

    // Clean up the input tensor if allocated
    if (model_input) {
        model_input = nullptr; // Avoid dangling pointer
    }

    // Clean up the output tensor if allocated
    if (model_output) {
        model_output = nullptr; // Avoid dangling pointer
    }
}

void BasicLearningModel::initialize()
{
    EV_TRACE << "initializing BasicLearningModel " << omnetpp::endl;

    // get rewards from training_info_file
    readJsonFile(training_info_path);

    // Load the model data from a file
    model_data = ReadModelFromFile(model_file_path);

    model = tflite::GetModel(model_data.data());
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // After loading the model
    if (model == nullptr) {
        throw cRuntimeError("BasicLearningModel.cc:Failed to load model.");
    } else {
        //EV << "Model loaded successfully. Number of subgraphs: " << model->subgraphs()->size() << omnetpp::endl;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
      //EV << "Model provided is schema version" << model->version() << " not equal to supported version " << TFLITE_SCHEMA_VERSION << omnetpp::endl;
      throw cRuntimeError("BasicLearningModel.cc: Model provided's schema version is not equal to supported version. ");
    }


    // TODO: Might need to Validate input tensor dimensions for multi-dimensional input
    // TODO: And also might need to validate output tensor dimensions.
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      throw cRuntimeError("BasicLearningModel.cc: AllocateTensors() failed");
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

       throw cRuntimeError("BasicLearningModel.cc: wrong input dimensions");
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
       throw cRuntimeError("BasicLearningModel.cc: wrong output dimensions");
   }

   const bool doTest = false;
   if (doTest) {
       // Example test cases
       testModelOutput({0.5, -1.0, -1.0, 1.0, 1.0}, 0, {1.0, 2.6736742e-19, 4.5851665e-09});
       testModelOutput({0.5, 0.29333332, 0.68, 0.0049, 0.0273}, 1, {0.00705481, 0.9893167, 0.00362856});
       testModelOutput({0.5, 0.29333332, 0.73333335, 0.0151, 0.0075}, 0, {0.77158856, 0.20506233, 0.02334913});
       testModelOutput({0.33333334, 0.0, 0.73333335, 0.0554, 0.1075}, 0, {0.7043012, 0.25701952, 0.03867925});
       testModelOutput({0.46666667, 0.26, 0.7866667, 0.0049, 0.0272}, 1, {0.00789684, 0.98852056, 0.00358255});
       testModelOutput({0.46666667, 0.28666666, 0.8, 0.0153, 0.007}, 0, {0.74286765, 0.23371261, 0.02341967});
       testModelOutput({0.33333334, 0.32, 0.79333335, 0.0258, 0.0169}, 0, {0.62283444, 0.34535939, 0.03180616});
       testModelOutput({0.6, 0.3, 0.6666667, 0.0055, 0.027}, 1, {0.0486112, 0.9424392, 0.00894961});
       testModelOutput({0.46666667, 0.29333332, 0.68666667, 0.0154, 0.0073}, 0, {0.6372505, 0.33337855, 0.02937095});
       testModelOutput({0.33333334, 0.24666667, 0.7, 0.0247, 0.0165}, 0, {0.6030182, 0.35945818, 0.03752354});
   }
}

double BasicLearningModel::readJsonValue(const json& jsonData, const std::string& key) {
    if (jsonData.contains(key)) {
        return jsonData[key].get<double>(); // Retrieve the value and return it as a double
    } else {
        throw cRuntimeError("Missing '%s' in JSON file.", key.c_str());
    }
}

void BasicLearningModel::readJsonFile(const std::string& filepath) {
    // Open the JSON file
    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        throw cRuntimeError("Error opening file: %s", filepath.c_str());
    }

    // Parse the JSON
    json jsonData;
    try {
        inputFile >> jsonData;
    } catch (json::parse_error& e) {
        throw cRuntimeError("JSON parsing error in file %s: %s", filepath.c_str(), e.what());
    }

    // Check and retrieve required keys
    if (!jsonData.contains("packet_reward") || !jsonData.contains("exploration_reward")) {
        throw cRuntimeError("JSON file %s does not contain required keys 'packet_reward' or 'exploration_reward'.", filepath.c_str());
    }

    // Retrieve values using the generic readJsonValue function
    try {
        packet_reward = readJsonValue(jsonData, "packet_reward");
        exploration_reward = readJsonValue(jsonData, "exploration_reward");
        random_choice_probability = readJsonValue(jsonData, "random_choice_probability");

        const json& normalization = jsonData["normalization"];

        // Read normalization factors
        latest_packet_rssi_norm_factor = readJsonValue(normalization, "latest_packet_rssi");
        latest_packet_snir_norm_factor = readJsonValue(normalization, "latest_packet_snir");
        latest_packet_timestamp_norm_factor = readJsonValue(normalization, "latest_packet_timestamp");
        num_received_packets_norm_factor = readJsonValue(normalization, "num_received_packets");
        current_timestamp_norm_factor = readJsonValue(normalization, "current_timestamp");
        coord_x_norm_factor = readJsonValue(normalization, "coord_x");

    } catch (const cRuntimeError& e) {
        throw cRuntimeError("Error reading JSON values: %s", e.what());
    }


    EV << "Packet Reward: " << packet_reward << std::endl;
    EV << "Exploration Reward: " << exploration_reward << std::endl;
}

StateLogger* BasicLearningModel::getStateLoggerModule() {
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

BasicRLMobility* BasicLearningModel::getMobilityModule() {
    // Fetch the mobility module, which is the parent of BasicLearningModel
    cModule* mobilityModule = getParentModule(); // mobility is the parent of BasicLearningModel
    if (!mobilityModule || strcmp(mobilityModule->getName(), "mobility") != 0) {
        throw cRuntimeError("The parent module is not the expected 'mobility' module.");
    }

    // Verify that the mobility module is an instance of SimpleRLMobility
    BasicRLMobility* mobility = check_and_cast<BasicRLMobility*>(mobilityModule);
    if (!mobility) {
        throw cRuntimeError("Failed to cast the mobility module to SimpleRLMobility.");
    }
    return mobility;
}

// Method to get the position (coordinate) from the SimpleRLMobility module
const Coord BasicLearningModel::getCoord() {
    // Fetch the mobility module, which is the parent of BasicLearningModel
    BasicRLMobility* mobility = getMobilityModule();
    // Return the current position of the mobility module
    const Coord pos = mobility->getCurrentPosition();
    return pos;

}

// Function to read the model from a file
std::vector<uint8_t> BasicLearningModel::ReadModelFromFile(const char* filename) {
    std::vector<uint8_t> model_data;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening model file: " << filename << std::endl;
        throw cRuntimeError("BasicLearningModel.cc: Error reading the model from file. ");
    }

    // Read the entire file into the model_data vector
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    if (file_size == -1) {
        throw cRuntimeError("BasicLearningModel.cc: Error getting file size.");
    }

    model_data.resize(file_size);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(model_data.data()), file_size);
    // Check if the read was successful
     if (!file) {
         std::cerr << "Error reading model data from file: " << filename << std::endl;
         throw cRuntimeError("BasicLearningModel.cc: Error reading the model data from file.");
     }

    return model_data;
}



void BasicLearningModel::setPacketInfo(double rssi, double snir, double nReceivedPackets, simtime_t timestamp, int id) {
    currentState.numReceivedPackets = nReceivedPackets;


    // we will normalize these later, when doing inference
    if (id == 0) {
        currentState.stampPos1 = getCoord() * coord_x_norm_factor;;
        currentState.timestamp1 = simTime().dbl() * current_timestamp_norm_factor;
    }
    else if (id == 1) {
        currentState.stampPos2 = getCoord() * coord_x_norm_factor;;
        currentState.timestamp2 = simTime().dbl() * current_timestamp_norm_factor;
    }
}


double BasicLearningModel::getReward() {

    double reward = (currentState.numReceivedPackets - lastStateNumberOfPackets) * packet_reward * rewardModifier;
    lastStateNumberOfPackets = currentState.numReceivedPackets;

    if (getMobilityModule()->isNewGridPosition()) {
        EV << "New grid!" << omnetpp::endl;
        reward += exploration_reward;
    }
    return reward;
}



int BasicLearningModel::pollModel()
{
    // get remaining state info:
    currentState.gwPosition = getCoord() * coord_x_norm_factor;
    currentState.timeOfSample = simTime().dbl();
    int reward = getReward();

    EV << "printing state: " << omnetpp::endl;
    EV << "currentState.gwPosition: " << currentState.gwPosition << omnetpp::endl;
    EV << "currentState.stampPos1: " << currentState.stampPos1 << omnetpp::endl;
    EV << "currentState.stampPos2: " << currentState.stampPos2 << omnetpp::endl;
    EV << "currentState.timestamp1: " << currentState.timestamp1 << omnetpp::endl;
    EV << "currentState.timestamp2: " << currentState.timestamp2 << omnetpp::endl;

    int output = invokeModel(currentState);
    StateLogger* stateLogger = getStateLoggerModule();
    stateLogger->logStep(currentState, output, reward);

    return output;

}


int BasicLearningModel::selectOutputIndex(float random_choice_probability, const TfLiteTensor* model_output, size_t num_outputs, bool deterministic) {
    int selected_index = -1;

    if (deterministic) {
        // Deterministic mode: choose the index with the highest weight
        selected_index = std::distance(model_output->data.f,
                          std::max_element(model_output->data.f, model_output->data.f + num_outputs));
        EV << "Deterministic mode: selected output index: " << selected_index
           << " with highest weight: " << model_output->data.f[selected_index] << "\n";
        return selected_index;
    }

    // Stochastic mode
    float random_value = static_cast<float>(rand()) / RAND_MAX; // Random value in [0, 1)
    if (random_choice_probability > random_value) {
        // Choose a random integer from 0 to num_outputs - 1
        selected_index = rand() % num_outputs;
        EV << "By random choice, selected output index: " << selected_index
           << " with weight: " << model_output->data.f[selected_index] << "\n";
        return selected_index;
    } else {
        random_value = static_cast<float>(rand()) / RAND_MAX; // Take a new random value in [0, 1)
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

int BasicLearningModel::invokeModel(InputStateBasic state) {
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
    unsigned long model_sum = 0;
    for (int i = 0; i < g_model_len; i++) {
        model_sum += g_model[i] & 0x0F; // get last 4 bytes.
    }
    EV << "debug msg. model_4_bit_per_byte_sum:" << model_sum << omnetpp::endl;
     */

    // Insert input data for the model from state values
    model_input->data.f[0] = state.gwPosition.x;
    model_input->data.f[1] = state.stampPos1.x;
    model_input->data.f[2] = state.stampPos2.x;

    // time since packets were received;
    double timesince1 = simTime().dbl() * current_timestamp_norm_factor - state.timestamp1;
    double timesince2 = simTime().dbl() * current_timestamp_norm_factor - state.timestamp2;

    model_input->data.f[3] = timesince1;
    model_input->data.f[4] = timesince2;
    // model_input->data.f[6] = state.coord.y;

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
    return selected_index;
}

// A helper function to compare two floating-point arrays with a tolerance
bool BasicLearningModel::compareArrays(const std::array<double, 3>& predicted, const std::array<double, 3>& expected, double tolerance) {
    for (size_t i = 0; i < 3; ++i) {
        if (std::abs(predicted[i] - expected[i]) > tolerance) {
            return false; // Return false if any element doesn't match within tolerance
        }
    }
    return true;
}

// The function you can call during initialize()
void BasicLearningModel::testModelOutput(
    const std::array<double, 5>& state,
    int expectedAction,
    const std::array<double, 3>& expectedActionProbs) {

    // Check if interpreter, model, model_input, or model_output are uninitialized
    if (interpreter == nullptr || model == nullptr || model_input == nullptr || model_output == nullptr
            || model_output->data.f == nullptr || model_output->bytes <= 0
    ) {
        EV << "Model or necessary components are not initialized." << omnetpp::endl;
        return;
    }
    if ((model_input->dims->size != 2) ||
        (model_input->dims->data[0] != kInputHeight) ||
        (model_input->dims->data[1] != kInputWidth) ||
        (model_input->type != kTfLiteFloat32)) {
        EV << "size, data0, data1" << omnetpp::endl;
        EV << model_input->dims->size << omnetpp::endl;
        EV << model_input->dims->data[0] << omnetpp::endl;
        EV << model_input->dims->data[1] << omnetpp::endl;

        throw cRuntimeError("BasicLearningModel.cc: wrong input dimensions");
    }
    if ((model_output->dims->size != 2) ||
        (model_output->dims->data[0] != kOutputHeight) ||
        (model_output->dims->data[1] != kOutputWidth) ||
        (model_output->type != kTfLiteFloat32)) {
        EV << "size, data0, data1" << omnetpp::endl;
        EV << model_output->dims->size << omnetpp::endl;
        EV << model_output->dims->data[0] << omnetpp::endl;
        EV << model_output->dims->data[1] << omnetpp::endl;
        throw cRuntimeError("BasicLearningModel.cc: wrong output dimensions");
    }

    model_input->data.f[0] = state[0];  // Assuming state[0] corresponds to x-position of gateway
    model_input->data.f[1] = state[1];  // Corresponds to x-position of first stationary node
    model_input->data.f[2] = state[2];  // Corresponds to x-position of second stationary node
    model_input->data.f[3] = state[3];
    model_input->data.f[4] = state[4];

    for (size_t i = 0; i < kInputWidth; ++i) { // Adjust based on your actual number of inputs
        //model_input->data.f[i] = (model_input_buffer[i]);
        EV << "model_input->data.f" << i << ": "  << model_input->data.f[i] << omnetpp::endl;
        if (!std::isfinite(model_input->data.f[i])) {
            EV << "Invalid input detected at index " << i << ": " << model_input->data.f[i] << omnetpp::endl;
            throw cRuntimeError("NaN or inf detected in model input");
        }
    }

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
        EV << "Invoke failed" << omnetpp::endl;
        throw cRuntimeError("Invoke failed");
    }
    // Obtain the model's output probabilities (Assuming output tensor contains probabilities for actions)
    std::array<double, 3> predictedActionProbs = {model_output->data.f[0], model_output->data.f[1], model_output->data.f[2]};


    // Compare the predicted action with the expected action
    int predictedAction = std::distance(predictedActionProbs.begin(),
                                        std::max_element(predictedActionProbs.begin(), predictedActionProbs.end()));

    if (predictedAction == expectedAction) {
        EV << "Action test passed for state: [";
        for (const auto& val : state) EV << val << " ";
        EV << "]\n";
    } else {
        EV << "Action test failed for state: [";
        for (const auto& val : state) EV << val << " ";
        EV << "].";
    }
    EV << "Expected: " << expectedAction << ", Got: " << predictedAction << "\n";

    // Compare the predicted action probabilities with the expected ones
    if (compareArrays(predictedActionProbs, expectedActionProbs)) {
        EV << "Probabilities test passed for state: [";
        for (const auto& val : state) EV << val << " ";
        EV << "]\n";
    } else {
        EV << "Probabilities test failed for state: [";
        for (const auto& val : state) EV << val << " ";
        EV << "].";
    }
    EV << "Expected: [";
        for (const auto& val : expectedActionProbs) EV << val << " ";
        EV << "], Got: [";
        for (const auto& val : predictedActionProbs) EV << val << " ";
        EV << "]\n";
}

} /* namespace inet */
