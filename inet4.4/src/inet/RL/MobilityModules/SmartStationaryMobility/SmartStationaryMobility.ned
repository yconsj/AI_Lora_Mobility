//
// Copyright (C) 2006 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


package inet.RL.MobilityModules.SmartStationaryMobility;

import inet.mobility.base.MobilityBase;

//
// This mobility module does nothing; it can be used for stationary nodes.
//
simple SmartStationaryMobility extends MobilityBase
{
    parameters:
		double initialX @unit(m) = default(uniform(this.constraintAreaMinX, this.constraintAreaMaxX));
        double initialY @unit(m) = default(uniform(this.constraintAreaMinY, this.constraintAreaMaxY));
        double initialZ @unit(m) = default(nanToZero(uniform(this.constraintAreaMinZ, this.constraintAreaMaxZ)));
        bool initFromDisplayString = default(true); // enables one time initialization from the subject module's display string
        double initialMovementHeading @unit(deg) = default(0deg); //uniform(0deg, 360deg)
        double initialMovementElevation @unit(deg) = default(0deg);
        bool updateFromDisplayString = default(true); // enables continuous update from the subject module's display string for dragging and rotating it
        @class(SmartStationaryMobility);
}

