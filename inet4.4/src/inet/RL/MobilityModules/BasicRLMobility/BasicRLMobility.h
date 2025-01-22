/*
 * BasicRLMobility.h
 *
 *  Created on: 31. jul. 2024
 *      Author: august
 */

#ifndef INET_MOBILITY_SINGLE_BASICRLMOBILITY_H_
#define INET_MOBILITY_SINGLE_BASICRLMOBILITY_H_

#include "inet/mobility/base/MovingMobilityBase.h"
#include "inet/common/geometry/common/Coord.h"
#include "../../LearningModels/BasicLearningModel/BasicLearningModel.h"
#include "inet/mobility/static/StationaryMobility.h"
#include "inet/common/INETMath.h"
#include "inet/common/geometry/common/Coord.h"

#include <vector>

namespace inet {

/**
 * @brief Linear movement model. See NED file for more info.
 *
 * @ingroup mobility
 * @author Emin Ilker Cetinbas
 */
class INET_API BasicRLMobility : public MovingMobilityBase
{
  public:
    virtual double getMaxSpeed() const override { return speed; }
    virtual const Coord& getInitialPosition();
    virtual bool isNewGridPosition();
    BasicRLMobility();
  protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    /** @brief Initializes mobility model parameters.*/
    virtual void initialize(int stage) override;
    /** @brief Move the host*/
    virtual void move() override;

    double speed;
    cMessage *pollModelTimer;
    simtime_t modelUpdateInterval;
    virtual const Coord& getLoRaNodePosition(int index);
    virtual void pollModel();
    virtual void schedulePollModelUpdate();
    virtual void handleSelfMessage(cMessage *message) override;

    rad heading ;
    rad elevation;
    Coord direction;
  private:
    Coord initialPosition;
    const int gridSize = 200;
    std::vector<int> visitedGrids;

};

} // namespace inet



#endif /* INET_MOBILITY_SINGLE_CUSTOMMOBILITY_H_ */
