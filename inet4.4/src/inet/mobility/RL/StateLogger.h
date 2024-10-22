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
#include "InputState.h"
namespace inet {

class StateLogger : public omnetpp::cSimpleModule {
public:
    StateLogger();
    virtual ~StateLogger();
    virtual void logStep(const InputState& inputState, int choice, double reward);

protected:
    virtual void finish() override;

private:
    std::vector<InputState> inputStateArray;
    std::vector<int> choiceArray;
    std::vector<double> rewardArray;

    void writeToFile();

};

} /* namespace inet */

#endif /* INET_RLSTATE_STATELOGGER_H_ */
