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

#include "MobileGatewayLoRaApp.h"

#include "inet/RL/LearningModels/AdvancedLearningModel/AdvancedLearningModel.h"
#include "LoRa/LoRaMac.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/common/ModuleAccess.h"
#include "inet/applications/base/ApplicationPacket_m.h"
#include "LoRaPhy/LoRaRadioControlInfo_m.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/SignalTag_m.h"
#include "inet/linklayer/base/MacProtocolBase.h"

namespace flora {

Define_Module(MobileGatewayLoRaApp);


void MobileGatewayLoRaApp::constructMacSubmoduleTable(cModule *module) {
    for (cModule::SubmoduleIterator it(module); !it.end(); ++it) {
        cModule *submodule = *it;

        // Check if the submodule is a LoRaMac

        if (auto loRaMac = dynamic_cast<flora::LoRaMac *>(submodule)) {
            // Go 2 levels up to get the LoRaNode module
            cModule *nicModule = loRaMac->getParentModule(); // Assume it's LoRaNic
            cModule *nodeModule = nicModule ? nicModule->getParentModule() : nullptr;

            if (nodeModule) {
                // Retrieve the LoRaNode index
                int nodeIndex = nodeModule->getIndex();

                // Retrieve the MAC address
                inet::MacAddress macAddress = loRaMac->getAddress();

                // Store the MAC address and associated module
                macToModuleMap[macAddress] = nodeModule; // Map MAC address to the LoRaNode module
                EV << "Added LoRaNode[" << nodeIndex << "] with MAC address: " << macAddress
                        << " (MAC module: " << loRaMac->getFullPath() << ")" << endl;
            } else {
                EV << "Failed to find grandparent LoRaNode for LoRaMac: " << loRaMac->getFullPath() << endl;
            }
        } else {
            // Recursively search in submodules
            EV << "Skipping non-LoRaMac submodule: " << submodule->getFullName() << endl;
            constructMacSubmoduleTable(submodule);
        }
    }
}

void MobileGatewayLoRaApp::initialize(int stage)
{
    if (stage == 0) {
        rssiVector.setName("Mobile Gateway RSSI");
        snirVector.setName("Mobile Gateway SNIR");
        LoRa_GWPacketReceived = registerSignal("LoRa_GWPacketReceived");
        localPort = par("localPort");
        destPort = par("destPort");
    } else if (stage == INITSTAGE_APPLICATION_LAYER) {
        startUDP();
        getSimulation()->getSystemModule()->subscribe("LoRa_AppPacketSent", this);
        cModule *network = getSimulation()->getSystemModule();
        constructMacSubmoduleTable(network);  // Call the recursive function directly
    }
}


void MobileGatewayLoRaApp::startUDP()
{
    EV << "MobileGatewayLoRaApp: Starting UDP" << endl;
    socket.setOutputGate(gate("socketOut"));
    const char *localAddress = par("localAddress");
    socket.bind(*localAddress ? L3AddressResolver().resolve(localAddress) : L3Address(), localPort);
    EV << "MobileGatewayLoRaApp: Reached past first resolve" << endl;
    // TODO: is this required?
    //setSocketOptions();

    const char *destAddrs = par("destAddresses");
    cStringTokenizer tokenizer(destAddrs);
    const char *token;

    // Create UDP sockets to multiple destination addresses (network servers)
    while ((token = tokenizer.nextToken()) != nullptr) {
        EV << "MobileGatewayLoRaApp: entering loop" << endl;
        EV << token << endl;
        L3Address result;
        L3AddressResolver().tryResolve(token, result);
        EV << "MobileGatewayLoRaApp: entering loop" << endl;
        if (result.isUnspecified())
            EV_ERROR << "MobileGatewayLoRaApp: cannot resolve destination address: " << token << endl;
        else
            EV << "MobileGatewayLoRaApp: Got destination address: " << token << endl;
        destAddresses.push_back(result);
    }
    EV << "MobileGatewayLoRaApp: We've reached the end." << endl;
}




void MobileGatewayLoRaApp::handleMessage(cMessage *msg)
{
    EV << msg->getArrivalGate() << endl;
    if (msg->arrivedOn("lowerLayerIn")) {
        EV << "Received LoRaMAC frame" << endl;
        auto pkt = check_and_cast<Packet*>(msg);
        const auto &frame = pkt->peekAtFront<LoRaMacFrame>();
        if(frame->getReceiverAddress() == MacAddress::BROADCAST_ADDRESS)
            processLoraMACPacket(pkt);
        //send(msg, "upperLayerOut");
        //sendPacket();
    } else if (msg->arrivedOn("socketIn")) {
        // FIXME : debug for now to see if LoRaMAC frame received correctly from network server
        EV << "Received UDP packet" << endl;
        auto pkt = check_and_cast<Packet*>(msg);
        const auto &frame = pkt->peekAtFront<LoRaMacFrame>();

        if (frame == nullptr)
            throw cRuntimeError("Packet type error");
        //EV << frame->getLoRaTP() << endl;
        //delete frame;

       /* auto loraTag = pkt->addTagIfAbsent<LoRaTag>();
        pkt->setBandwidth(loRaBW);
        pkt->setCarrierFrequency(loRaCF);
        pkt->setSpreadFactor(loRaSF);
        pkt->setCodeRendundance(loRaCR);
        pkt->setPower(W(loRaTP));*/

        send(pkt, "lowerLayerOut");
        //
    }
}


void MobileGatewayLoRaApp::logPacketInfoToModel(double rssi, double snir, double nReceivedPackets, simtime_t timestamp, int id) {
    // Get the parent module (MobileLoRaGW)
    cModule *parentModule = getParentModule();
    if (!parentModule)
        throw cRuntimeError("MobileLoRaGatewayApp has no parent module");

    // Fetch the mobility module (SimpleRLMobility)
    cModule *mobilityModule = getParentModule()->getSubmodule("mobility");
    if (!mobilityModule)
        throw cRuntimeError("SimpleRLMobility module not found!");

    // Get the AdvancedLearningModel submodule from SimpleRLMobility
    AdvancedLearningModel *advancedLearningModel = check_and_cast<AdvancedLearningModel*>(mobilityModule->getSubmodule("advancedLearningModel"));
    if (!advancedLearningModel)
        throw cRuntimeError("AdvancedLearningModel module not found");
    // Log the packet information (RSSI, SNIR, and timestamp)
    advancedLearningModel->setPacketInfo(id);
}

void MobileGatewayLoRaApp::processLoraMACPacket(Packet *pk)
{
    // FIXME: Change based on new implementation of MAC frame.
    emit(LoRa_GWPacketReceived, 1);
    if (simTime() >= getSimulation()->getWarmupPeriod())
        counterOfReceivedPackets++;
    pk->trimFront();
    auto frame = pk->removeAtFront<LoRaMacFrame>();

    auto snirInd = pk->getTag<SnirInd>();

    auto signalPowerInd = pk->getTag<SignalPowerInd>();

    W w_rssi = signalPowerInd->getPower();
    double rssi = w_rssi.get()*1000;
    frame->setRSSI(math::mW2dBmW(rssi));
    frame->setSNIR(snirInd->getMinimumSnir());
    pk->insertAtFront(frame);
    //bool exist = false;
    EV << "transmit address " << frame->getTransmitterAddress() << endl;
    //for (std::vector<nodeEntry>::iterator it = knownNodes.begin() ; it != knownNodes.end(); ++it)

    // --- Logging message data --- //
    //rssiVector.record(frame->getRSSI());
    //snirVector.record(snirInd->getMinimumSnir());
    inet::MacAddress transmitterAddress = frame->getTransmitterAddress();
    EV << "Transmitter address: " << transmitterAddress << endl;

    auto it = macToModuleMap.find(transmitterAddress);
    if (it != macToModuleMap.end()) {
        cModule *loRaNode = it->second;
        int nodeIndex = loRaNode->getIndex();
        EV << "Packet received from LoRaNode[" << nodeIndex << "] with MAC: " << transmitterAddress << endl;
        logPacketInfoToModel(frame->getRSSI(), snirInd->getMinimumSnir(), (double)counterOfReceivedPackets, simTime(), nodeIndex );
   } else {
       EV << "Unknown transmitter: " << transmitterAddress.str() << "\n";
   }

    // FIXME : Identify network server message is destined for.
    L3Address destAddr = destAddresses[0];
    if (pk->getControlInfo())
       delete pk->removeControlInfo();

    socket.sendTo(pk, destAddr, destPort);
}

void MobileGatewayLoRaApp::sendPacket()
{
//    LoRaAppPacket *mgmtCommand = new LoRaAppPacket("mgmtCommand");
//    mgmtCommand->setMsgType(TXCONFIG);
//    LoRaOptions newOptions;
//    newOptions.setLoRaTP(uniform(0.1, 1));
//    mgmtCommand->setOptions(newOptions);
//
//    LoRaMacFrame *response = new LoRaMacFrame("mgmtCommand");
//    response->encapsulate(mgmtCommand);
//    response->setLoRaTP(pk->getLoRaTP());
//    response->setLoRaCF(pk->getLoRaCF());
//    response->setLoRaSF(pk->getLoRaSF());
//    response->setLoRaBW(pk->getLoRaBW());
//    response->setReceiverAddress(pk->getTransmitterAddress());
//    send(response, "lowerLayerOut");

}

void MobileGatewayLoRaApp::receiveSignal(cComponent *source, simsignal_t signalID, intval_t value, cObject *details)
{
    if (signalID == LoRa_GWPacketReceived ) {
        if (simTime() >= getSimulation()->getWarmupPeriod())
            counterOfSentPacketsFromNodes++;
    }
}

void MobileGatewayLoRaApp::finish()
{
    recordScalar("LoRa_GW_DER", double(counterOfReceivedPackets)/counterOfSentPacketsFromNodes);

}



} //namespace inet
