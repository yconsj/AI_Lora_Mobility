

#ifndef INET_RL_INPUTSTATE_H_
#define INET_RL_INPUTSTATE_H_


#include "inet/common/geometry/common/Coord.h"

namespace inet {

typedef struct InputStateBasic {

    Coord gwPosition; // gateways position
    Coord stampPos1; // position of gw when last packet of node 1 received
    Coord stampPos2; // position of gw when last packet of node 2 received
    double timestamp1; // time since gw received packet from node 1
    double timestamp2; // time since gw received packet from node 2
    double numReceivedPackets; // not normalized, not used for inference, but for data analysis
    double timeOfSample; // not normalized, not used for inference, but for data analysis
    InputStateBasic() : gwPosition(-1,-1), stampPos1(-1,-1), stampPos2(-1,-1), timestamp1(-1.0), timestamp2(-1.0), numReceivedPackets(0.0), timeOfSample(0.0) {} // Default constructor
} InputStateBasic;


}
#endif
