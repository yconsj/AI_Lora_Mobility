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

#include "StateLogger.h"
#include <omnetpp.h>
#include <iostream>

namespace inet {

Define_Module(StateLogger);

StateLogger::StateLogger() {
    // TODO Auto-generated constructor stub
}


void StateLogger::logStep(const InputState& inputState, int choice, double reward) {
    inputStateArray.push_back(inputState);
    choiceArray.push_back(choice);
    rewardArray.push_back(reward);
}


void StateLogger::writeToFile() {
    std::ofstream outFile("log.txt");
    if (outFile.is_open()) {
        // Write inputState array
        outFile << "[";  // Start of inputState array
        for (size_t i = 0; i < inputStateArray.size(); ++i) {
            const InputState& state = inputStateArray[i];
            outFile << "(" << state.latestPacketRSSI << ", "
                    << state.latestPacketSNIR << ", "
                    << state.latestPacketTimestamp << ", "
                    << state.currentTimestamp << ", " // Added currentTimestamp
                    << state.coord << ")";
            if (i < inputStateArray.size() - 1) outFile << ", ";
        }
        outFile << "]\n";  // End of inputState array

        // Write choice array
        outFile << "[";
        for (size_t i = 0; i < choiceArray.size(); ++i) {
            outFile << choiceArray[i];
            if (i < choiceArray.size() - 1) outFile << ", ";
        }
        outFile << "]\n";

        // Write reward array
        outFile << "[";
        for (size_t i = 0; i < rewardArray.size(); ++i) {
            outFile << rewardArray[i];
            if (i < rewardArray.size() - 1) outFile << ", ";
        }
        outFile << "]\n";

        outFile.close();
    } else {
        EV << "Error opening file to write log data.\n";
    }
}

void StateLogger::finish() {
    writeToFile();
}

StateLogger::~StateLogger() {
    // TODO Auto-generated destructor stub
}

} /* namespace inet */
