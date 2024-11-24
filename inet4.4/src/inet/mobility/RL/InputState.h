

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

typedef struct InputStateBasic {

    Coord gwPosition; // gateways position
    Coord stampPos1; // position of gw when last packet of node 1 received
    Coord stampPos2; // position of gw when last packet of node 2 received
    double timestamp1; // time since gw received packet from node 1
    double timestamp2; // time since gw received packet from node 2
    double numReceivedPackets;

    InputStateBasic() : gwPosition(-1,-1), stampPos1(-1,-1), stampPos2(-1,-1), timestamp1(-1.0), timestamp2(-1.0), numReceivedPackets(0.0) {} // Default constructor
} InputStateBasic;


}
#endif
