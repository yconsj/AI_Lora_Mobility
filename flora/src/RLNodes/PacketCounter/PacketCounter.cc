#include "PacketCounter.h"
#include "inet/RL/StateLogger/StateLogger.h"

namespace flora {
Define_Module(PacketCounter);


void PacketCounter::initialize() {
    // Set the runnumber from the simulation environment
    runNumber = getSimulation()->getActiveEnvir()->getConfigEx()->getActiveRunNumber();

    // Subscribe to the LoRa_GWPacketReceived signal
    getSimulation()->getSystemModule()->subscribe("LoRa_GWPacketReceived", this);

}

void PacketCounter::receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID, omnetpp::intval_t value, omnetpp::cObject *details) {
    // Ensure the source is "packetForwarder"
    if (strcmp(source->getFullName(), "packetForwarder") == 0) {
        // Get the parent module of the packetForwarder
        cModule *parentModule = source->getParentModule();
        if (parentModule != nullptr) {

            const char *parentName = parentModule->getName(); // Get the full name of the parent module
            static cModule* network = getSimulation()->getSystemModule();
            static inet::StateLogger* stateLogger = omnetpp::check_and_cast<inet::StateLogger*>(network->getSubmodule("stateLogger"));
            EV << "parentmodname: " << parentName << omnetpp::endl;
            if (strcmp(parentName, "StationaryLoraGw") == 0) {
                //stateLogger->logStationaryGatewayPacketReception(parentModule->getIndex());
            }

        }
    }
}

void PacketCounter::finish() {
}

}

