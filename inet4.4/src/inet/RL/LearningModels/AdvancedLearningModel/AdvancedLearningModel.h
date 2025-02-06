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

#ifndef INET_RL_LEARNINGMODELS_ADVANCEDLEARNINGMODEL_H_
#define INET_RL_LEARNINGMODELS_ADVANCEDLEARNINGMODEL_H_

#include "inet/common/geometry/common/Coord.h"
#include <vector>
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "../../MobilityModules/AdvancedRLMobility/AdvancedRLMobility.h"
#include "inet/RL/include/json.hpp"
#include "inet/RL/StateLogger/StateLogger.h"
#include "inet/common/InitStages.h"
#include "inet/common/ModuleAccess.h"

#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omnetpp.h>
#include <array>
#include <cmath>
#include <deque>

#include "inet/common/geometry/common/Coord.h"
#include "inet/mobility/contract/IMobility.h" // for accessing mobility
#include <random>  // For random sampling

#include "../../MobilityModules/AdvancedRLMobility/AdvancedRLMobility.h"
#include "inet/RL/modelfiles/policy_net_model.h"
#include "inet/RL/StateLogger/StateLogger.h"  // Include the StateLogger header

using json = nlohmann::json;

namespace inet {
const size_t recent_packets_length = 1;
const size_t number_of_nodes = 4;

class AdvancedLearningModel : public omnetpp::cSimpleModule {
public:
    AdvancedLearningModel();
    virtual ~AdvancedLearningModel();
    virtual int pollModel();
    virtual void setPacketInfo(int index);

protected:
    // The following redefined virtual function holds the algorithm.
    virtual void initialize(int stage) override;
    int numInitStages() const override { return NUM_INIT_STAGES; }

private:
    virtual void nodeValueInitialization();
    StateLogger* getStateLoggerModule();  // Function to fetch the StateLogger module. should not be virtual
    int invokeModel();
    const Coord getCoord();
    AdvancedRLMobility* getMobilityModule();
    virtual std::vector<uint8_t> ReadModelFromFile(const char* filename);
    int selectOutputIndex(float random_choice_probability, const TfLiteTensor* model_output, size_t num_outputs, bool deterministic);
    double calculateNormalizedAngle(const Coord& coord1, const Coord& coord2);
private:

    std::vector<uint8_t> model_data;
    double random_choice_probability = 0.0;

    simtime_t sim_time_limit;
    float max_cross_distance;

    std::vector<int> number_of_received_packets_per_node;

    std::vector<simtime_t> send_intervals;
    std::vector<simtime_t> expected_send_times;
    std::vector<Coord> node_positions;

    std::deque<float> recent_packets;
    std::vector<cModule *> nodes;


};

} /* namespace inet */

#endif /* INET_RL_LEARNINGMODELS_ADVANCEDLEARNINGMODEL_H_ */
