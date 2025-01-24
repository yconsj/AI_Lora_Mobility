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

#include "inet/RL/InputState.h"
#include "inet/RL/include/json.hpp"
using json = nlohmann::json;


namespace inet {

class StateLogger : public omnetpp::cSimpleModule {
public:
    StateLogger();
    virtual ~StateLogger();
    virtual void addTransmissionTime();
    virtual void logStationaryGatewayPacketReception(int loragwIndex);
    virtual void logStep(int choice);
    void writeToFile();


protected:
    virtual void finish() override;
    virtual void initialize() override;
private:
    std::vector<InputState> inputStateArray;
    std::vector<int> choiceArray;
    std::vector<double> transmissionTimes;
    std::vector<double> stationaryReceptionTimes;  // Store reception times for stationary gateways
    int runnumber = -1;

};

} /* namespace inet */

#endif /* INET_RLSTATE_STATELOGGER_H_ */
