//
// Copyright (C) 2005 Emin Ilker Cetinbas
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//
//
// Author: Emin Ilker Cetinbas (niw3_at_yahoo_d0t_com)
//

#include "inet/mobility/RL/SimpleRLMobility.h"

#include "inet/common/INETMath.h"
//#include <stdio.h>
//#include <string.h>

namespace inet {

Define_Module(SimpleRLMobility);

SimpleRLMobility::SimpleRLMobility()
{
    speed = 1;

}

void SimpleRLMobility::initialize(int stage)
{
    MovingMobilityBase::initialize(stage);

    EV_TRACE << "initializing SimpleRLMobility stage " << stage << endl;
    if (stage == INITSTAGE_LOCAL) {
        speed = par("speed");
        stationary = (speed == 0);
        rad heading = deg(fmod(par("initialMovementHeading").doubleValue(), 360));
        rad elevation = deg(fmod(par("initialMovementElevation").doubleValue(), 360));
        Coord direction = Quaternion(EulerAngles(heading, -elevation, rad(0))).rotate(Coord::X_AXIS);

        lastVelocity = direction * speed;
        mySpeed = getSpeedFromXML(par("configScript"));
    }
}
double SimpleRLMobility::getSpeedFromXML(cXMLElement *nodes) {
    // Recursively traverse the whole config file, looking for
            // speed attributes
            cXMLElementList childs = nodes->getChildren();
            for (auto& child : childs) {
                const char *speedAttr = child->getAttribute("speed");
                if (speedAttr) {
                    double speed = std::stod(speedAttr);
                    EV_TRACE << speed << endl;
                    return speed;
                }
            }
}

void SimpleRLMobility::move()
{
    double elapsedTime = (simTime() - lastUpdate).dbl();
    //cModule
    //double newSpeed = subjectModule->RSSI;
    //cast_and_check
    //char* text = sprintf("%d",newSpeed);

    //subjectModule->
    //    EV_TRACE << "test " <<  endl;

    lastPosition += lastVelocity * elapsedTime * mySpeed;
    mySpeed *= 1.5;

    // do something if we reach the wall
    Coord dummyCoord;
    handleIfOutside(REFLECT, dummyCoord, lastVelocity);
}

} // namespace inet

