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

#ifndef INET_RLSTATE_STATELOGGER_H_
#define INET_RLSTATE_STATELOGGER_H_

#include <vector>
#include <fstream>
#include <algorithm>


#include "inet/RL/include/json.hpp"
#include "inet/linklayer/common/MacAddress.h"
#include "inet/common/geometry/common/Coord.h"
#include "inet/common/InitStages.h"

using json = nlohmann::json;


namespace inet {

class StateLogger : public omnetpp::cSimpleModule {
public:
    StateLogger();
    virtual ~StateLogger();
    virtual void addTransmissionTime(int node_index);
    virtual void logStationaryGatewayPacketReception(int lora_gw_index,
            int lora_node_index,
            int transmitter_sequence_number);
    virtual void logStaticMobilityGatewayPacketReception(int lora_node_index,
                int transmitter_sequence_number);
    virtual void logStep(
            Coord gw_pos,
            std::vector<float> node_distances,
            std::vector<int> number_of_received_packets_per_node,
            double time,
            int choice
            );
    void writeToFile();

protected:
    virtual void finish() override;
    virtual void initialize(int stage) override;
    int numInitStages() const override { return NUM_INIT_STAGES; }
private:
    std::map<MacAddress, cModule*> macToModuleMap; // MAC to module mapping
    std::vector<std::vector<double>> transmission_times_vec;
    std::vector<int> transmissions_per_node_current_vec;
    std::vector<std::vector<int>> transmissions_per_node_vec;

    // RL Mobile gw logging (logStep()), received from AdvancedLearningModule
    std::vector<float> gw_positions_x_vec;
    std::vector<float> gw_positions_y_vec;
    std::vector<std::vector<float>> node_distances_vec;
    std::vector<std::vector<int>> mobile_gw_number_of_received_packets_per_node_vec;
    std::vector<float> times_vec;
    std::vector<int> actions_vec;

    std::vector<int> transmission_id_vec;
    // Stationary gw logging
    std::vector<int> stationary_gw_received_packets_per_node_current_vec;
    std::vector<std::vector<int>> stationary_gw_number_of_received_packets_per_node_vec;

    // StaticMobility gw logging
    std::vector<int> static_mobility_gw_received_packets_per_node_current_vec;
    std::vector<std::vector<int>> static_mobility_gw_number_of_received_packets_per_node_vec;


    int runnumber = -1;

};

} /* namespace inet */

#endif /* INET_RLSTATE_STATELOGGER_H_ */
