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

#ifndef INET_MOBILITY_RL_LEARNINGMODEL_H_
#define INET_MOBILITY_RL_LEARNINGMODEL_H_

#include "inet/common/geometry/common/Coord.h"
#include "StateLogger.h"
#include "InputState.h"
#include "SimpleRLMobility.h"
#include <vector>
#include "include/json.hpp"

#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

using json = nlohmann::json;

namespace inet {


class LearningModel : public omnetpp::cSimpleModule {
public:
    virtual void setPacketInfo(double rssi, double snir, double nReceivedPackets, simtime_t timestamp, int id);
    LearningModel();
    virtual ~LearningModel();
    virtual int pollModel();

protected:
    // The following redefined virtual function holds the algorithm.
    virtual void initialize() override;

private:
    StateLogger* getStateLoggerModule();  // Function to fetch the StateLogger module. should not be virtual
    int invokeModel(InputStateBasic state);
    double getReward();
    const Coord getCoord();

    bool compareArrays(const std::array<double, 3>& predicted, const std::array<double, 3>& expected, double tolerance = 1e-6);
    void testModelOutput(
        const std::array<double, 5>& state,
        int expectedAction,
        const std::array<double, 3>& expectedActionProbs);

    SimpleRLMobility* getMobilityModule();
    virtual std::vector<uint8_t> ReadModelFromFile(const char* filename);
    void readJsonFile(const std::string& filepath);
    double readJsonValue(const json& jsonData, const std::string& key);
    int selectOutputIndex(float random_choice_probability, const TfLiteTensor* model_output, size_t num_outputs, bool deterministic);

private:
    double lastStateNumberOfPackets;
    std::vector<uint8_t> model_data;
    InputStateBasic currentState;
    int lastPacketId = -1;
    double rewardModifier = 1.0;
    double packet_reward = -1.0;
    double exploration_reward = -1.0;
    double random_choice_probability = 0.0;

    // Normalization factors initialized to -1.0
    double latest_packet_rssi_norm_factor = -1.0;    // Normalization factor for packet RSSI
    double latest_packet_snir_norm_factor = -1.0;    // Normalization factor for packet SNIR
    double latest_packet_timestamp_norm_factor = -1.0; // Normalization factor for packet timestamp
    double num_received_packets_norm_factor = -1.0;  // Normalization factor for number of received packets
    double current_timestamp_norm_factor = -1.0;     // Normalization factor for current timestamp
    double coord_x_norm_factor = -1.0;               // Normalization factor for x-coordinate

};

} /* namespace inet */

#endif /* INET_MOBILITY_RL_LEARNINGMODEL_H_ */
