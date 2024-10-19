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
        rad heading = deg(fmod(par("initialMovementHeading").doubleValue(), 360));
        rad elevation = deg(fmod(par("initialMovementElevation").doubleValue(), 360));
        Coord direction = Quaternion(EulerAngles(heading, -elevation, rad(0))).rotate(Coord::X_AXIS);
        directionX = pollModel(); //getDirectionFromXML(par("configScript"));

        // Fetch directionX from config (0 = left, 1 = right, values between represent probability)
        //directionX = par("configScript").doubleValue();  // Get directionX as a float value

        // Determine actual direction based on the value of directionX
        // For values between 0 and 1, use random probability to move left or right
        double randomValue = uniform(0, 1);  // Generate a random number between 0 and 1
        if (randomValue < directionX) {
            // Move right (if random value is less than directionX)
            direction.x = 1;
        } else {
            // Move left (if random value is greater than or equal to directionX)
            direction.x = -1;
        }

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
        learningModel->Test();
    } else {
        EV << "LearningModel submodule not found!" << endl;
    }



    lastPosition += lastVelocity * elapsedTime;
    // mySpeed *= 1.5;

    // do something if we reach the wall
    Coord dummyCoord;
    handleIfOutside(REFLECT, dummyCoord, lastVelocity);
}

double SimpleRLMobility::pollModel()
{
    return uniform(0,1);
}

} // namespace inet

