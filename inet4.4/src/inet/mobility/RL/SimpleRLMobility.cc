//
// Copyright (C) 2005 Emin Ilker Cetinbas
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//
//
// Author: Emin Ilker Cetinbas (niw3_at_yahoo_d0t_com)
//

#include "inet/mobility/RL/SimpleRLMobility.h"
#include "inet/mobility/static/StationaryMobility.h"

#include "inet/common/INETMath.h"
#include "inet/common/geometry/common/Coord.h"

#include "LearningModel.h"

namespace inet {

Define_Module(SimpleRLMobility);

SimpleRLMobility::SimpleRLMobility()
{
    pollModelTimer = nullptr;
    modelUpdateInterval = 0;
    initialPosition = Coord(NAN, NAN, NAN);
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


        pollModelTimer = new cMessage("pollModel");
        modelUpdateInterval = par("modelUpdateInterval");
        simtime_t firstModelUpdate = 0.0;
        scheduleAt(firstModelUpdate, pollModelTimer); // schedule a model update for time 0.0;
        //schedulePollModelUpdate();
    }
    if (stage == INITSTAGE_SINGLE_MOBILITY) {
        initialPosition = lastPosition;
    }
}


void SimpleRLMobility::schedulePollModelUpdate()
{
   cancelEvent(pollModelTimer);
   simtime_t nextUpdate = simTime() + modelUpdateInterval;
   scheduleAt(nextUpdate, pollModelTimer);
}

void SimpleRLMobility::handleSelfMessage(cMessage *message)
{
    if (message == moveTimer) {
        moveAndUpdate();
        scheduleUpdate();
    }
    else if (message == pollModelTimer) {
        EV << "pollModel timer" << omnetpp::endl;
        pollModel();
        schedulePollModelUpdate();
    }
}


const Coord& SimpleRLMobility::getInitialPosition() {
    return initialPosition;
}

const Coord& SimpleRLMobility::getLoRaNodePosition(int index)
{
    // Access the parent network module
    cModule *network = getParentModule()->getParentModule();
    cModule *targetNode = network->getSubmodule("loRaNodes", index);

    if (targetNode) {
        cModule *mobilityModule = targetNode->getSubmodule("mobility");
        if (mobilityModule) {
            // Directly cast to StationaryMobility as expected
            StationaryMobility *stationaryMobility = check_and_cast<StationaryMobility *>(mobilityModule);
            return stationaryMobility->getCurrentPosition();
        }
        else {
            EV << "Mobility submodule not found for loRaNodes[" << index << "]!" << endl;
        }
    } else {
        EV << "loRaNodes[" << index << "] module not found!" << endl;
    }
    // Return an invalid coordinate if the node or mobility submodule is not found
    return Coord(NAN, NAN, NAN);
}

void SimpleRLMobility::pollModel() {
    //subjectModule->
    EV << "test LM" <<  endl;
    cModule* submodule = getSubmodule("learningModel");
    LearningModel *learningModel = check_and_cast<LearningModel*>(submodule);

    // Call a function from LearningModel
    if (learningModel) {
        int choice = learningModel->pollModel();

        Coord targetPosition = getLoRaNodePosition(choice); // Change 0 to the desired node index if needed

        if (!std::isnan(targetPosition.x)) { // Check if position is valid
            // Move towards the target node in x-direction only
            double deltaX = targetPosition.x - lastPosition.x;
            direction = Coord(deltaX, 0, 0); // Set direction vector only in x-direction
            direction.normalize();
            lastVelocity = direction * speed;

            // Optional: Print debug information
            EV << "Moving towards LoRaNode[" << choice << "] in x-direction at position: " << targetPosition.x << endl;
        } else {
            EV << "Invalid target position; LoRaNode[" << choice << "]! may not be reachable." << endl;
        }

    } else {
        EV << "LearningModel submodule not found!" << endl;
    }
}

void SimpleRLMobility::move()
{
    double elapsedTime = (simTime() - lastUpdate).dbl();

    lastPosition += lastVelocity * elapsedTime;

    // do something if we reach the wall
    Coord dummyCoord;
    handleIfOutside(REFLECT, dummyCoord, lastVelocity);
}

const Coord& SimpleRLMobility::getCurrentPosition() {
    return lastPosition;
}


int getSign(int num) {
    return (num > 0) - (num < 0);
}

bool SimpleRLMobility::isNewGridPosition() {
    float distance = getCurrentPosition().x - getInitialPosition().x;
    int currentGridSlice = (distance / (float)gridSize);  // Incorporate direction with sign
    EV << "currentX : " << getCurrentPosition().x << ", initialX : " << getInitialPosition().x << omnetpp::endl;
    EV << "grid: " << currentGridSlice << omnetpp::endl;
    for (int i = 0; i < visitedGrids.size(); i++) {
        if (currentGridSlice == visitedGrids[i]) {
            return false; // Already visited
        }
    }
    visitedGrids.push_back(currentGridSlice);  // Mark as visited
    return true;
}


} // namespace inet

