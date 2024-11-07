

#ifndef RLNODES_PACKETCOUNTER_H_
#define RLNODES_PACKETCOUNTER_H_
#include <omnetpp.h>
#include <fstream>
#include <string>
#include "../include/json.hpp"
using json = nlohmann::json;

namespace flora {

class PacketCounter : public omnetpp::cSimpleModule, public omnetpp::cListener {
private:
    int counterGW1 = 0;
    int counterGW2 = 0;
    int runNumber;
    std::ofstream outputFile;
    std::string outputFileName; // Changed to std::string for file name

    void updateJsonFile();

public:
    virtual void initialize() override;
    virtual void receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID, omnetpp::intval_t value, omnetpp::cObject *details) override;
    virtual void finish() override;
};

}
#endif /* RLNODES_PACKETCOUNTER_H_ */
