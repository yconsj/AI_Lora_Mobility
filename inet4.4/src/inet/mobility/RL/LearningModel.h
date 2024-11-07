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
    int invokeModel(InputState state);
    double getReward();
    std::vector<uint8_t> model_data;
    Coord getCoord();
    InputState currentState;
    int lastPacketId = -1;
    double rewardModifier = 1.0;
    double packet_reward = -1.0;
    double exploration_reward = -1.0;
    double random_choice_probability = 0.0;

    SimpleRLMobility* getMobilityModule();
    InputState normalizeInputState(InputState state);
    double lastStateNumberOfPackets;
    virtual std::vector<uint8_t> ReadModelFromFile(const char* filename);
    void readJsonFile(const std::string& filepath);
};

} /* namespace inet */

#endif /* INET_MOBILITY_RL_LEARNINGMODEL_H_ */
