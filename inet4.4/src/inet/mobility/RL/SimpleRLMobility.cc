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
#include "inet/common/geometry/common/Coord.h"

#include "LearningModel.h"
//#include <stdio.h>
//#include <string.h>

namespace inet {

Define_Module(SimpleRLMobility);

SimpleRLMobility::SimpleRLMobility()
{
    speed = 10;
}

void SimpleRLMobility::initialize(int stage)
{
    MovingMobilityBase::initialize(stage);

    EV << "initializing SimpleRLMobility stage " << stage << endl;
    if (stage == INITSTAGE_LOCAL) {
        speed = par("speed");
        stationary = (speed == 0);
        heading = deg(fmod(par("initialMovementHeading").doubleValue(), 360));
        elevation = deg(fmod(par("initialMovementElevation").doubleValue(), 360));
        direction = Quaternion(EulerAngles(heading, -elevation, rad(0))).rotate(Coord::X_AXIS);
        lastVelocity = direction * speed;

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
    EV << "test LM" <<  endl;
    cModule* submodule = getSubmodule("learningModel");
    LearningModel *learningModel = check_and_cast<LearningModel*>(submodule);

    // Call a function from LearningModel
    if (learningModel) {
        int choice = learningModel->pollModel();
        if (choice == 0) {
            direction.x = 1;
        }
        else {
            direction.x = -1;
        }
    } else {
        EV << "LearningModel submodule not found!" << endl;
    }
    lastVelocity = direction * speed;
    lastPosition += lastVelocity * elapsedTime;
    // mySpeed *= 1.5;

    // do something if we reach the wall
    Coord dummyCoord;
    handleIfOutside(REFLECT, dummyCoord, lastVelocity);
}

const Coord& SimpleRLMobility::getCurrentPosition() {
    // Assuming `currentPosition` is a member variable of type Coord
    return lastPosition; // Return a reference to the member variable
}


double SimpleRLMobility::pollModel()
{
    return uniform(0,1);
}

} // namespace inet

