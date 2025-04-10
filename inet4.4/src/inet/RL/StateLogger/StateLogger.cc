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

#include "inet/RL/StateLogger/StateLogger.h"
#include "inet/RL/modelfiles/policy_net_model.h"
#include "inet/RL/LearningModels/AdvancedLearningModel/AdvancedLearningModel.h"
#include <omnetpp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>




namespace inet {

Define_Module(StateLogger);

StateLogger::StateLogger() {
    // TODO Auto-generated constructor stub
}

void StateLogger::initialize(int stage) {
    cSimpleModule::initialize(stage);
    if (stage == INITSTAGE_PHYSICAL_LAYER) {
        cModule *network = getSimulation()->getSystemModule();
        const char* lora_mod_str = "loRaNodes";
        size_t n_sim_nodes =  network->getSubmoduleVectorSize(lora_mod_str);
        runnumber = getSimulation()->getActiveEnvir()->getConfigEx()->getActiveRunNumber();
        transmission_times_vec.resize(n_sim_nodes, std::vector<double>());
        transmissions_per_node_current_vec.resize(n_sim_nodes, 0);

        // cModule *network = getSimulation()->getSystemModule();
        int number_of_stationary_gw = network->getSubmoduleVectorSize("StationaryLoraGw") + network->getSubmoduleVectorSize("SmartStationaryLoraGw");
        transmission_id_vec.resize(n_sim_nodes, -1);
        stationary_gw_received_packets_per_node_current_vec.resize(n_sim_nodes, 0);

        static_mobility_gw_received_packets_per_node_current_vec.resize(n_sim_nodes, 0);
    }
}

void StateLogger::addTransmissionTime(int node_index) {
    transmission_times_vec[node_index].push_back(simTime().dbl());
    transmissions_per_node_current_vec[node_index] += 1;

}

void StateLogger::logStationaryGatewayPacketReception(int lora_gw_index, int lora_node_index, int transmitter_sequence_number) {
    // Log the reception time when a packet is received by a stationary gateway
    // But check if the packet has been received by any of the other stationary gateways, already.

    // TODO: Dont split it up by gateway index.
    EV << "lora_gw_index="<< lora_gw_index << endl;
    EV << "lora_node_index="<< lora_node_index << endl;
    EV << "transmitter_sequence_number="<< transmitter_sequence_number << endl;
    if (transmission_id_vec[lora_node_index] >= transmitter_sequence_number) {
        EV << "Duplicate packet received" << endl;
        return;
    }
    transmission_id_vec[lora_node_index] = std::max(transmission_id_vec[lora_node_index], transmitter_sequence_number);
    stationary_gw_received_packets_per_node_current_vec[lora_node_index] += 1;
}

void StateLogger::logStaticMobilityGatewayPacketReception( int lora_node_index, int transmitter_sequence_number) {
    // TODO: Consider adding node distances
    EV << "lora_node_index="<< lora_node_index << endl;
    EV << "transmitter_sequence_number="<< transmitter_sequence_number << endl;
    /*
    if (transmission_id_vec[lora_node_index] >= transmitter_sequence_number) {
        EV << "Duplicate packet received" << endl;
        return;
    }
    transmission_id_vec[lora_node_index] = std::max(transmission_id_vec[lora_node_index], transmitter_sequence_number);
    */

    static_mobility_gw_received_packets_per_node_current_vec[lora_node_index] += 1;
}



void StateLogger::logStep(
        Coord gw_pos,
        std::vector<float> node_distances,
        std::vector<int> number_of_received_packets_per_node,
        double time,
        int choice) {
    gw_positions_x_vec.push_back(gw_pos.x);
    gw_positions_y_vec.push_back(gw_pos.y);
    node_distances_vec.push_back(node_distances);

    mobile_gw_number_of_received_packets_per_node_vec.push_back(number_of_received_packets_per_node);
    times_vec.push_back(time);
    actions_vec.push_back(choice);

    transmissions_per_node_vec.push_back(transmissions_per_node_current_vec);
    stationary_gw_number_of_received_packets_per_node_vec.push_back(stationary_gw_received_packets_per_node_current_vec);
    static_mobility_gw_number_of_received_packets_per_node_vec.push_back(static_mobility_gw_received_packets_per_node_current_vec);
}


void StateLogger::writeUnifiedCSVWithRuns(const std::string& filename) {
    bool writeHeader = (runnumber == 0);

    std::ofstream file;
    if (writeHeader) {
        file.open(filename, std::ios::out | std::ios::trunc);  // overwrite
    } else {
        file.open(filename, std::ios::out | std::ios::app);    // append
    }

    if (!file.is_open()) {
        EV << "Failed to open CSV file: " << filename << "\n";
        return;
    }

    size_t num_timesteps = times_vec.size();
    size_t num_nodes = node_distances_vec.empty() ? 0 : node_distances_vec[0].size();

    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(3);

    if (writeHeader) {
        buffer << "run,t,gw_x,gw_y,action";
        for (size_t i = 0; i < num_nodes; ++i) buffer << ",node" << i << "_dist";
        for (size_t i = 0; i < num_nodes; ++i) buffer << ",node" << i << "_tx";
        for (size_t i = 0; i < num_nodes; ++i) buffer << ",node" << i << "_rx_mobile";
        for (size_t i = 0; i < num_nodes; ++i) buffer << ",node" << i << "_rx_stationary";
        for (size_t i = 0; i < num_nodes; ++i) buffer << ",node" << i << "_rx_staticmob";
        buffer << "\n";
    }

    for (size_t t = 0; t < num_timesteps; ++t) {
        buffer << runnumber << "," << times_vec[t] << "," << gw_positions_x_vec[t] << "," << gw_positions_y_vec[t] << "," << actions_vec[t];
        for (size_t i = 0; i < num_nodes; ++i) buffer << "," << node_distances_vec[t][i];
        for (size_t i = 0; i < num_nodes; ++i) buffer << "," << transmissions_per_node_vec[t][i];
        for (size_t i = 0; i < num_nodes; ++i) buffer << "," << mobile_gw_number_of_received_packets_per_node_vec[t][i];
        for (size_t i = 0; i < num_nodes; ++i) buffer << "," << stationary_gw_number_of_received_packets_per_node_vec[t][i];
        for (size_t i = 0; i < num_nodes; ++i) buffer << "," << static_mobility_gw_number_of_received_packets_per_node_vec[t][i];
        buffer << "\n";
    }

    file << buffer.str();
    file.close();
}


void StateLogger::writeToFile() {
    runnumber = getSimulation()->getActiveEnvir()->getConfigEx()->getActiveRunNumber();
    if (runnumber < 0) {
        throw cRuntimeError("Failed to fetch runnumber");
    }

    std::string csv_file = std::string(log_file_basename) + "_data.csv";
    std::string json_file = std::string(log_file_basename) + ".json";

    writeUnifiedCSVWithRuns(csv_file);

    json allRunsJson;

    // Read existing JSON if not run 0
    if (runnumber != 0) {
        std::ifstream inFile(json_file);
        if (inFile.is_open()) {
            try {
                inFile >> allRunsJson;
            } catch (const std::exception& e) {
                EV << "Warning: Failed to parse existing JSON. Starting fresh.\n";
            }
            inFile.close();
        }
    }

    // Create run-specific JSON
    json runJson;
    Coord areaMin, areaMax;
    areaMin.x = par("constraintAreaMinX").doubleValue();
    areaMin.y = par("constraintAreaMinY").doubleValue();
    areaMin.z = par("constraintAreaMinZ").doubleValue();
    areaMax.x = par("constraintAreaMaxX").doubleValue();
    areaMax.y = par("constraintAreaMaxY").doubleValue();
    areaMax.z = par("constraintAreaMaxZ").doubleValue();
    runJson["static"]["area_min"] = {areaMin.x, areaMin.y, areaMin.z};
    runJson["static"]["area_max"] = {areaMax.x, areaMax.y, areaMax.z};
    runJson["static"]["number_of_nodes"] = number_of_sim_nodes;

    // Add node positions from loRaNodes module instances
    json node_positions_json = json::array();
    const char* lora_mod_str = "loRaNodes";

    cModule* network = getSimulation()->getSystemModule();
    cModule* loraContainer = network->getSubmodule("loRaNodes");


    for (int node_index = 0; node_index < number_of_sim_nodes; ++node_index) {
        cModule *lora_node_module = network->getSubmodule(lora_mod_str, node_index);
        if (!lora_node_module) {
            EV << "Warning: loRaNode[" << node_index << "] not found.\n";
            continue;
        }
        auto *mobility = check_and_cast<StationaryMobility *>(lora_node_module->getSubmodule("mobility"));

        // get stationary lora node positions
        if (!mobility) {
            throw cRuntimeError("Error, node missing mobility module in LoRa application for node %d", node_index);
        }
        Coord pos =  mobility->getCurrentPosition();
        node_positions_json.push_back({pos.x, pos.y, pos.z});
    }

    runJson["static"]["node_positions"] = node_positions_json;

    runJson["file_reference"] = csv_file;

    // --- Add transmission_times ---
    json tx_json;
    for (size_t i = 0; i < transmission_times_vec.size(); ++i) {
        tx_json[std::to_string(i)] = transmission_times_vec[i];
    }
    runJson["transmission_times"] = tx_json;

    // Add this run to the global metadata
    allRunsJson[std::to_string(runnumber)] = runJson;

    std::ofstream outFile(json_file);
    if (outFile.is_open()) {
        outFile << allRunsJson.dump(2);  // Indented for readability
        outFile.close();
    } else {
        EV << "Error opening metadata JSON file: " << json_file << "\n";
    }
}




void StateLogger::finish() {
    writeToFile();
}


StateLogger::~StateLogger() {
    //autogenerated stub for destructor
}

} /* namespace inet */
