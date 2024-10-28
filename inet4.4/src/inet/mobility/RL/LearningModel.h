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

namespace inet {


class LearningModel : public omnetpp::cSimpleModule {
public:
    virtual void setPacketInfo(double rssi, double snir, double nReceivedPackets, simtime_t timestamp);

    LearningModel();
    virtual ~LearningModel();
    virtual int pollModel();
    void PrintOutput(float x, float y);

protected:
    // The following redefined virtual function holds the algorithm.
    virtual void initialize() override;

private:
    void fetchStateLoggerModule();  // Function to fetch the StateLogger module
    int invokeModel(InputState state);
    int getReward();
    int gridSize;
    Coord getCoord();
    StateLogger* stateLogger;
    InputState currentState;
    SimpleRLMobility* getMobilityModule();
    InputState normalizeInputState(InputState state);


};

} /* namespace inet */

#endif /* INET_MOBILITY_RL_LEARNINGMODEL_H_ */
