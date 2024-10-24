/*
 * SimpleRLMobility.h
 *
 *  Created on: 31. jul. 2024
 *      Author: august
 */

#ifndef INET_MOBILITY_SINGLE_SIMPLERLMOBILITY_H_
#define INET_MOBILITY_SINGLE_SIMPLERLMOBILITY_H_

#include "inet/mobility/base/MovingMobilityBase.h"

namespace inet {

/**
 * @brief Linear movement model. See NED file for more info.
 *
 * @ingroup mobility
 * @author Emin Ilker Cetinbas
 */
class INET_API SimpleRLMobility : public MovingMobilityBase
{
  public:
    double mySpeed;
  protected:
    double speed;

  protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }

    /** @brief Initializes mobility model parameters.*/
    virtual void initialize(int stage) override;

    /** @brief Move the host*/
    virtual void move() override;

  public:
    virtual double getMaxSpeed() const override { return speed; }
    SimpleRLMobility();
};

} // namespace inet



#endif /* INET_MOBILITY_SINGLE_CUSTOMMOBILITY_H_ */
