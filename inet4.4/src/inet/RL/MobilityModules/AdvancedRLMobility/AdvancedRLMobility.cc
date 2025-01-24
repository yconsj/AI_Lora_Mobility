#include "AdvancedRLMobility.h"
#include "../../LearningModels/AdvancedLearningModel/AdvancedLearningModel.h"


namespace inet {

Define_Module(AdvancedRLMobility);

AdvancedRLMobility::AdvancedRLMobility()
{
    speed = 0;
    pollModelTimer = nullptr;
    modelUpdateInterval = 0;
    initialPosition = lastPosition;

    heading = deg(360);
    elevation = deg(0.0);
    direction = Quaternion(EulerAngles(heading, -elevation, rad(0))).rotate(Coord::X_AXIS);
}

void AdvancedRLMobility::initialize(int stage)
{
    MovingMobilityBase::initialize(stage);
    EV << "initializing AdvancedRLMobility stage " << stage << endl;
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


void AdvancedRLMobility::schedulePollModelUpdate()
{
   cancelEvent(pollModelTimer);
   simtime_t nextUpdate = simTime() + modelUpdateInterval;
   scheduleAt(nextUpdate, pollModelTimer);
}

void AdvancedRLMobility::handleSelfMessage(cMessage *message)
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


const Coord& AdvancedRLMobility::getInitialPosition() {
    return initialPosition;
}

const double AdvancedRLMobility::getMaxCrossDistance() {
    return constraintAreaMin.distance(constraintAreaMax);
}

const Coord& AdvancedRLMobility::getLoRaNodePosition(int index)
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
    throw cRuntimeError("node or mobility submodule is not found");
    // Return an invalid coordinate if the node or mobility submodule is not found
    return Coord(NAN, NAN, NAN);
}

void AdvancedRLMobility::pollModel() {
    //subjectModule->
    EV << "test LM" <<  endl;
    cModule* submodule = getSubmodule("advancedLearningModel");
    AdvancedLearningModel *advancedLearningModel = check_and_cast<AdvancedLearningModel*>(submodule);
    if (!advancedLearningModel) {
        EV << "AdvancedLearningModel submodule not found!" << endl;
        return;
    }
    // Call a function from AdvancedLearningModel
    int choice = advancedLearningModel->pollModel();
    switch(choice) {
    case 0: // stand still
        lastVelocity = direction * 0.0;
        EV << "Stalling at position " << lastPosition.x << omnetpp::endl;
        break;
    case 1: // left
        direction = Coord(-1, 0, 0);
        break;
    case 2: // right
        direction = Coord(1, 0, 0);
        break;
    case 3:  // up
        direction = Coord(0, 1, 0);
        break;
    case 4: // down
        direction = Coord(0, -1, 0);
        break;
    default:
       // raise error
        EV << "Invalid Action: " << choice << endl;
    }
    direction.normalize();           // Normalize the direction
    lastVelocity = direction * speed;
    EV << "direction: " << direction << omnetpp::endl;
}

void AdvancedRLMobility::move()
{
    double elapsedTime = (simTime() - lastUpdate).dbl();

    lastPosition += lastVelocity * elapsedTime;
    // do something if we reach the wall
    Coord dummyCoord;
    handleIfOutside(REFLECT, dummyCoord, lastVelocity);
}


} // namespace inet

