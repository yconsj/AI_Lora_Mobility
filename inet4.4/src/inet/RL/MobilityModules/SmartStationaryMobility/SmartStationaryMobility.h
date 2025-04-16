//
// Copyright (C) 2006 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


#ifndef __INET_SMARTSTATIONARYMOBILITY_H
#define __INET_SMARTSTATIONARYMOBILITY_H

#include "inet/mobility/base/StationaryMobilityBase.h"
#include "inet/mobility/base/MovingMobilityBase.h"
#include <vector>
#include <stack>
#include <algorithm>   // for std::sort, std::min
#include <cmath>       // for std::min, std::max (if not already included)
#include "inet/common/geometry/common/Coord.h"
#include "inet/mobility/contract/IMobility.h" // for accessing mobility
#include "inet/common/InitStages.h"
#include "inet/common/ModuleAccess.h"
#include "inet/mobility/static/StationaryMobility.h"


namespace inet {

/**
 * This mobility module does not move at all; it can be used for standalone stationary nodes.
 *
 * @ingroup mobility
 */
class INET_API SmartStationaryMobility : public StationaryMobilityBase
{
  protected:
    bool updateFromDisplayString;
    rad heading ;
    rad elevation;
    Coord direction;

  private:
    static bool alreadyOptimized;
    static std::vector<Coord> optimizedGatewayPositions;
    Coord initialPosition;

  public:
    SmartStationaryMobility();
  protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    /** @brief Initializes mobility model parameters.*/
    virtual void initialize(int stage) override;

  private:
    std::vector<Coord> computeOptimizedPositions(const std::vector<Coord>& nodePositions, int numGateways, double maxRange);


};

} // namespace inet

#endif

