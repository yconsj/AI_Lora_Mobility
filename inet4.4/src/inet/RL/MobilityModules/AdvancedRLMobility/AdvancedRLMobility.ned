//
// Copyright (C) 2005 Emin Ilker Cetinbas
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//
//
// Author: Emin Ilker Cetinbas (niw3_at_yahoo_d0t_com)
//

package inet.RL.MobilityModules.AdvancedRLMobility;

import inet.mobility.base.MovingMobilityBase;
import inet.RL.LearningModels.AdvancedLearningModel.AdvancedLearningModel;

module AdvancedRLMobility extends MovingMobilityBase
{
    parameters:
        double initialX @unit(m) = default(uniform(this.constraintAreaMinX, this.constraintAreaMaxX));
        double initialY @unit(m) = default(uniform(this.constraintAreaMinY, this.constraintAreaMaxY));
        double initialZ @unit(m) = default(nanToZero(uniform(this.constraintAreaMinZ, this.constraintAreaMaxZ)));
        bool initFromDisplayString = default(true); // enables one time initialization from the subject module's display string
        double speed @unit(mps) = default(0mps); // speed of the host
        double initialMovementHeading @unit(deg) = default(0deg); //uniform(0deg, 360deg)
        double initialMovementElevation @unit(deg) = default(0deg);
		//updateInterval = 0.1s;  // parameter how often movement is scheduled. normal default is 0.1 seconds
	    double modelUpdateInterval @unit(s) = default(10s); // how often we poll the model from the mobility module
        @class(AdvancedRLMobility);
        
    submodules:
        advancedLearningModel: AdvancedLearningModel;
}

