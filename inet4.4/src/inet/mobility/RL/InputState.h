

#ifndef INET_MOBILITY_RL_INPUTSTATE_H_
#define INET_MOBILITY_RL_INPUTSTATE_H_


#include "inet/common/geometry/common/Coord.h"

namespace inet {


typedef struct InputState {
    double latestPacketRSSI;      // RSSI value for the latest packet
    double latestPacketSNIR;      // SNIR value for the latest packet
    omnetpp::simtime_t latestPacketTimestamp;  // Timestamp of the latest packet
    double numReceivedPackets;
    omnetpp::simtime_t currentTimestamp;
    Coord coord;

    InputState() : latestPacketRSSI(0.0), latestPacketSNIR(0.0), latestPacketTimestamp(0.0), numReceivedPackets(0.0), currentTimestamp(0.0), coord(0, 0) {} // Default constructor
} InputState;

}
#endif
