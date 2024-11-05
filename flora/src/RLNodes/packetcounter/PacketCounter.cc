#include "PacketCounter.h"

namespace flora {
Define_Module(PacketCounter);


void PacketCounter::initialize() {
    // Set the runnumber from the simulation environment
    runNumber = getSimulation()->getActiveEnvir()->getConfigEx()->getActiveRunNumber();

    // Subscribe to the LoRa_GWPacketReceived signal
    getSimulation()->getSystemModule()->subscribe("LoRa_GWPacketReceived", this);

    // Get the output file name from parameters
    outputFileName = par("outputFileName").stringValue();
}

void PacketCounter::receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID, omnetpp::intval_t value, omnetpp::cObject *details) {
    // Ensure the source is "packetForwarder"
    if (strcmp(source->getFullName(), "packetForwarder") == 0) {
        // Get the parent module of the packetForwarder
        cModule *parentModule = source->getParentModule();

        if (parentModule != nullptr) {
            const char *parentName = parentModule->getFullName(); // Get the full name of the parent module

            // Check if the parent module is loRaGW[0] or loRaGW[1]
            if (strcmp(parentName, "loRaGW[0]") == 0) {
                counterGW1++;
            } else if (strcmp(parentName, "loRaGW[1]") == 0) {
                counterGW2++;
            }
        }

        // Log the source's name for debugging
        EV << "TEST RECEIVE SIGNAL: " << source->getFullName() << " from parent: " << (parentModule ? parentModule->getFullName() : "No parent") << omnetpp::endl;
    }
}

void PacketCounter::finish() {
    updateJsonFile();
}

void PacketCounter::updateJsonFile() {
    json j;

    // Reading from the file
    std::ifstream inputFile(outputFileName);
    if (inputFile.is_open()) {
        inputFile >> j; // Parse the JSON from the file
        inputFile.close(); // Close the input file
    } else {
        // If the file does not exist, initialize a new JSON object
        j = json::object();
    }

    // Create or update the entry for the current run number
    if (!j.contains(std::to_string(runNumber))) {
        j[std::to_string(runNumber)] = json::object(); // Create a new object for this run number
    }

    // Update the counts for each gateway
    j[std::to_string(runNumber)]["loragw[0]"] = counterGW1;
    j[std::to_string(runNumber)]["loragw[1]"] = counterGW2;

    // Writing to the file
    std::ofstream outputFile(outputFileName);
    if (outputFile.is_open()) {
        outputFile << j.dump(4); // Pretty print with an indent of 4 spaces
        outputFile.close(); // Close the output file
    } else {
        throw omnetpp::cRuntimeError("Failed to open output file for writing: %s", outputFileName.c_str());
    }
}

}

