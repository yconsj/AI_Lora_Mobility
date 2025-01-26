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
#include "CustomLoRaMedium.h"
#include "LoRa/LoRaMacFrame_m.h"
#include "LoRaPhy/LoRaBandListening.h"
#include "LoRaPhy/LoRaTransmission.h"
#include "inet/common/INETUtils.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/Simsignals.h"
#include "inet/linklayer/common/MacAddressTag_m.h"
#include "inet/networklayer/common/NetworkInterface.h"
#include "inet/common/ProtocolTag_m.h"
#include "inet/networklayer/contract/IInterfaceTable.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IInterference.h"
#include "inet/physicallayer/wireless/common/radio/packetlevel/Radio.h"
#include "inet/physicallayer/wireless/common/medium/RadioMedium.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/SignalTag_m.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IErrorModel.h"
#include <fstream>

namespace flora {

Define_Module(CustomLoRaMedium);

CustomLoRaMedium::CustomLoRaMedium() : LoRaMedium()
{

}

CustomLoRaMedium::~CustomLoRaMedium()
{
}

void CustomLoRaMedium::addTransmission(const IRadio *transmitterRadio, const ITransmission *transmission) {
    // Call the parent class's addTransmission
    LoRaMedium::addTransmission(transmitterRadio, transmission);

    // check if its a loRaNode transmititng:
    cModule *transmitter_module = transmitterRadio->getRadioGate()->getOwnerModule();
    std::string lora_node_str = "loRaNodes";
    // transmitter module (radio) is a submodule of loranic, which is a submodule of loranodes;
    cModule *loraModule = transmitter_module->getParentModule()->getParentModule();

    EV << "loraModule name: "<< loraModule->getName() << endl;
    if (lora_node_str.compare(loraModule->getName()) == 0) {
        // Fetch the StateLogger module from the network
       static cModule *network = getSimulation()->getSystemModule();
       static auto stateLogger = check_and_cast<StateLogger*>(network->getSubmodule("stateLogger"));

       // get the index of the transmitting lora node
       if (stateLogger) {
           // Add the transmission time to the StateLogger
           stateLogger->addTransmissionTime(loraModule->getIndex());
       } else {
           EV << "Error: StateLogger module not found in the network." << std::endl;
       }
    }


}

void CustomLoRaMedium::finish() {
    // Call the parent class's finish method
    LoRaMedium::finish();
}


}
