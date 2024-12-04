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
            const char *parentName = parentModule->getFullName(); // Get the full name of the parent module
            cModule* network = getSimulation()->getSystemModule();
            inet::StateLogger* stateLogger = check_and_cast<inet::StateLogger*>(network->getSubmodule("stateLogger"));

            // Check if the parent module is StationaryLoraGw[0] or StationaryLoraGw[1] (stationary gateways)
            if (strcmp(parentName, "StationaryLoraGw[0]") == 0) {
                // Access the StateLogger submodule and call the logStationaryGatewayPacketReception function for StationaryLoraGw[0]
                stateLogger->logStationaryGatewayPacketReception(0); // Pass index 0 for StationaryLoraGw[0]
            } else if (strcmp(parentName, "StationaryLoraGw[1]") == 0) {
                // Access the StateLogger submodule and call the logStationaryGatewayPacketReception function for StationaryLoraGw[1]
                stateLogger->logStationaryGatewayPacketReception(1); // Pass index 1 for StationaryLoraGw[1]
            }
        }
    }
}

void PacketCounter::finish() {
}

}

