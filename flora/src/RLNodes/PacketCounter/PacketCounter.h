

#ifndef RLNODES_PACKETCOUNTER_H_
#define RLNODES_PACKETCOUNTER_H_
#include <omnetpp.h>
#include <fstream>
#include <string>
#include "inet/RL/StateLogger/StateLogger.h"

namespace flora {

class PacketCounter : public omnetpp::cSimpleModule, public omnetpp::cListener {
private:
    int runNumber;
    void updateJsonFile();

public:
    virtual void initialize() override;
    virtual void receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID, omnetpp::intval_t value, omnetpp::cObject *details) override;
    virtual void finish() override;
};

}
#endif /* RLNODES_PACKETCOUNTER_H_ */
