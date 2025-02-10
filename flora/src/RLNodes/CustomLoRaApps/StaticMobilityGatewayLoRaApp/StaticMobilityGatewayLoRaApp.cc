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

#include "StaticMobilityGatewayLoRaApp.h"
//#include "inet/networklayer/ipv4/IPv4Datagram.h"
//#include "inet/networklayer/contract/ipv4/IPv4ControlInfo.h"




namespace flora {

Define_Module(StaticMobilityGatewayLoRaApp);


void StaticMobilityGatewayLoRaApp::initialize(int stage)
{
    if (stage == 0) {
        LoRa_GWPacketReceived = registerSignal("LoRa_GWPacketReceived");
        localPort = par("localPort");
        destPort = par("destPort");
    } else if (stage == INITSTAGE_APPLICATION_LAYER) {
        startUDP();
        getSimulation()->getSystemModule()->subscribe("LoRa_AppPacketSent", this);
    }
    //cModule *parentModule = getParentModule();
    //EV << "StaticMobility_app parent_module:" << parentModule->getName() << " index: " << parentModule->getIndex() << omnetpp::endl;
}

int getLoRaNodeIndex(MacAddress node_mac_address) {

    cModule *network = getSimulation()->getSystemModule();

    const char* lora_nodes_string = "loRaNodes";
    const char* lora_node_nic_string = "LoRaNic";
    const char* lora_node_mac_string = "mac";

    for (int node_index = 0; node_index < number_of_nodes; node_index++) {
        cModule *loRa_node = network->getSubmodule(lora_nodes_string, node_index);
        flora::LoRaMac *loRa_node_mac = check_and_cast<flora::LoRaMac *>(loRa_node->getSubmodule(lora_node_nic_string)
                ->getSubmodule(lora_node_mac_string)
                );
        if (loRa_node_mac->getAddress().equals(node_mac_address)) {
            return node_index;
        }
    }
    return -1;
}


void StaticMobilityGatewayLoRaApp::startUDP()
{
    EV << "StaticMobilityGatewayLoRaApp: Starting UDP" << endl;
    socket.setOutputGate(gate("socketOut"));
    const char *localAddress = par("localAddress");
    socket.bind(*localAddress ? L3AddressResolver().resolve(localAddress) : L3Address(), localPort);
    EV << "StaticMobilityGatewayLoRaApp: Reached past first resolve" << endl;
    // TODO: is this required?
    //setSocketOptions();

    const char *destAddrs = par("destAddresses");
    cStringTokenizer tokenizer(destAddrs);
    const char *token;

    // Create UDP sockets to multiple destination addresses (network servers)
    while ((token = tokenizer.nextToken()) != nullptr) {
        EV << "StaticMobilityGatewayLoRaApp: entering loop" << endl;
        EV << token << endl;
        L3Address result;
        L3AddressResolver().tryResolve(token, result);
        EV << "StaticMobilityGatewayLoRaApp: entering loop" << endl;
        if (result.isUnspecified())
            EV_ERROR << "cannot resolve destination address: " << token << endl;
        else
            EV << "Got destination address: " << token << endl;
        destAddresses.push_back(result);
    }
    EV << "We've reached the end." << endl;
}


void StaticMobilityGatewayLoRaApp::handleMessage(cMessage *msg)
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

void StaticMobilityGatewayLoRaApp::processLoraMACPacket(Packet *pk)
{



    // FIXME: Change based on new implementation of MAC frame.
    emit(LoRa_GWPacketReceived, 42);
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
    EV << frame->getTransmitterAddress() << endl;
    //for (std::vector<nodeEntry>::iterator it = knownNodes.begin() ; it != knownNodes.end(); ++it)


    // Added feature:
    cModule* network = getSimulation()->getSystemModule();
    inet::StateLogger* stateLogger = omnetpp::check_and_cast<inet::StateLogger*>(network->getSubmodule("stateLogger"));
    cModule *parentModule = getParentModule();


    // TODO: Current progress. Get the sender node idx, and the packets sent by that node so far. Use that as an "id".
    // TODO: Consider having this logic by encapsulating sender idx and message count of sender in the message's content.
    EV << "pk->getSenderModule():" << pk->getSenderModule()->getName() << omnetpp::endl;
    auto transmitter_mac_address = frame->getTransmitterAddress();
    auto transmitter_sequence_number = frame->getSequenceNumber();

    stateLogger->logStaticMobilityGatewayPacketReception(getLoRaNodeIndex(transmitter_mac_address),
            transmitter_sequence_number);
    EV << "StaticMobility_app parent_module:" << parentModule->getName() << omnetpp::endl;
    // packet counter:
    // frame->getSequenceNumber()



    // FIXME : Identify network server message is destined for.
    L3Address destAddr = destAddresses[0];
    if (pk->getControlInfo())
       delete pk->removeControlInfo();

    socket.sendTo(pk, destAddr, destPort);
}

void StaticMobilityGatewayLoRaApp::sendPacket()
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

void StaticMobilityGatewayLoRaApp::receiveSignal(cComponent *source, simsignal_t signalID, intval_t value, cObject *details)
{
    if (simTime() >= getSimulation()->getWarmupPeriod())
        counterOfSentPacketsFromNodes++;
}

void StaticMobilityGatewayLoRaApp::finish()
{
    recordScalar("LoRa_GW_DER", double(counterOfReceivedPackets)/counterOfSentPacketsFromNodes);
}



} //namespace inet
