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


namespace inet {

Define_Module(StateLogger);

StateLogger::StateLogger() {
    // TODO Auto-generated constructor stub
}

void StateLogger::initialize() {
    runnumber = getSimulation()->getActiveEnvir()->getConfigEx()->getActiveRunNumber();
    transmission_times_vec.resize(number_of_nodes, std::vector<double>());
    transmissions_per_node_current_vec.resize(number_of_nodes, 0);

    cModule *network = getSimulation()->getSystemModule();
    int number_of_stationary_gw = network->getSubmoduleVectorSize("StationaryLoraGw");
    transmission_id_vec.resize(number_of_nodes, -1);
    stationary_gw_received_packets_per_node_current_vec.resize(number_of_nodes, 0);
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
    stationary_reception_times_vec.push_back(simTime().dbl());
    stationary_gw_received_packets_per_node_current_vec[lora_node_index] += 1;

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
}


void StateLogger::writeToFile() {
    runnumber = getSimulation()->getActiveEnvir()->getConfigEx()->getActiveRunNumber();

    if (runnumber < 0) {
        throw cRuntimeError("Failed to fetch runnumber");
    }

    // Construct the filename based on the current runnumber
    std::string filename = std::string(log_file_basename) + "_" + std::to_string(runnumber) + ".json";
    std::ofstream outFile(filename);

    if (outFile.is_open()) {
        // Create a JSON object
        json outputJson;

        outputJson["static"]["number_of_nodes"] = number_of_nodes;
        outputJson["mobile_gw_data"]["node_distances"] = node_distances_vec;
        outputJson["mobile_gw_data"]["gw_positions_x"] = gw_positions_x_vec;
        outputJson["mobile_gw_data"]["gw_positions_y"] = gw_positions_y_vec;
        outputJson["mobile_gw_data"]["number_of_received_packets_per_node"] = mobile_gw_number_of_received_packets_per_node_vec;
        outputJson["mobile_gw_data"]["times"] = times_vec;
        outputJson["mobile_gw_data"]["actions"] = actions_vec;


        // Add transmission times to the JSON object
        outputJson["nodes"]["transmission_times"] = transmission_times_vec;
        outputJson["nodes"]["transmissions_per_node"] = transmissions_per_node_vec;


        // Add stationary gateway reception times to the JSON object
        outputJson["stationary_gw_data"]["stationary_gateway_reception_times"] = stationary_reception_times_vec;
        outputJson["stationary_gw_data"]["stationary_gw_number_of_received_packets_per_node"] = stationary_gw_number_of_received_packets_per_node_vec;


        // Write the JSON object to the file
        outFile << outputJson.dump(4);  // Pretty print with 4-space indentation

        outFile.close();
    } else {
        EV << "Error opening file to write log data.\n";
    }
}


void StateLogger::finish() {
    writeToFile();
}


StateLogger::~StateLogger() {
    //autogenerated stub for destructor
}

} /* namespace inet */
