#include "BasicRLMobility.h"



namespace inet {

Define_Module(BasicRLMobility);

BasicRLMobility::BasicRLMobility()
{
    speed = 0;
    pollModelTimer = nullptr;
    modelUpdateInterval = 0;
    initialPosition = lastPosition;

    heading = deg(360);
    elevation = deg(0.0);
    direction = Quaternion(EulerAngles(heading, -elevation, rad(0))).rotate(Coord::X_AXIS);
}

void BasicRLMobility::initialize(int stage)
{
    MovingMobilityBase::initialize(stage);
    EV << "initializing BasicRLMobility stage " << stage << endl;
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


void BasicRLMobility::schedulePollModelUpdate()
{
   cancelEvent(pollModelTimer);
   simtime_t nextUpdate = simTime() + modelUpdateInterval;
   scheduleAt(nextUpdate, pollModelTimer);
}

void BasicRLMobility::handleSelfMessage(cMessage *message)
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


const Coord& BasicRLMobility::getInitialPosition() {
    return initialPosition;
}

const Coord& BasicRLMobility::getLoRaNodePosition(int index)
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

void BasicRLMobility::pollModel() {
    //subjectModule->
    EV << "test LM" <<  endl;
    cModule* submodule = getSubmodule("basicLearningModel");
    BasicLearningModel *basicLearningModel = check_and_cast<BasicLearningModel*>(submodule);
    if (!basicLearningModel) {
        EV << "BasicLearningModel submodule not found!" << endl;
        return;
    }
    // Call a function from LearningModel
    int choice = basicLearningModel->pollModel();
    if (choice == 2) {
        lastVelocity = direction * 0.0;
        EV << "Stalling at position " << lastPosition.x << omnetpp::endl;
        return;
    }
    else {
        Coord targetPosition = getLoRaNodePosition(choice); // Change 0 to the desired node index if needed
        if (!std::isnan(targetPosition.x)) { // Check if position is valid
            // Move towards the target node in x-direction only
            double epsilon = 1e-6; // Small tolerance for floating-point comparison
            double deltaX = targetPosition.x - lastPosition.x;
            if (std::abs(deltaX) > epsilon) { // Check if deltaX is significantly different from zero
                direction = Coord(deltaX, 0, 0); // Set direction vector only in x-direction
                direction.normalize();           // Normalize the direction
                lastVelocity = direction * speed;
                EV << "targetPosition: " << targetPosition << omnetpp::endl;
                EV << "lastPosition: " << lastPosition << omnetpp::endl;
                EV << "direction: " << direction << omnetpp::endl;
                EV << "Moving towards LoRaNode[" << choice << "] in x-direction at position: " << targetPosition.x << endl;
                EV << "lastVelocity: " << lastVelocity << omnetpp::endl;
            } else {
                // If deltaX is too small, set velocity to zero
                lastVelocity = Coord(0, 0, 0);
                EV << "Moving towards LoRaNode[" << choice << "] in x-direction at position: " << targetPosition.x << endl;
                EV << "But already on top. Velocity set to 0: " << lastVelocity << omnetpp::endl;
            }
        } else {
            EV << "Invalid target position; LoRaNode[" << choice << "]! may not be reachable." << endl;
        }
    }
}

void BasicRLMobility::move()
{
    double elapsedTime = (simTime() - lastUpdate).dbl();

    lastPosition += lastVelocity * elapsedTime;
    // do something if we reach the wall
    Coord dummyCoord;
    handleIfOutside(REFLECT, dummyCoord, lastVelocity);
}


int getSign(int num) {
    return (num > 0) - (num < 0);
}

bool BasicRLMobility::isNewGridPosition() {
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

