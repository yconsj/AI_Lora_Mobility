//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef LORAPHY_LORAANALOGMODEL_H_
#define LORAPHY_LORAANALOGMODEL_H_

#include "inet/physicallayer/wireless/common/base/packetlevel/ScalarAnalogModelBase.h"
#include "inet/physicallayer/wireless/common/radio/packetlevel/BandListening.h"
#include "inet/physicallayer/wireless/common/analogmodel/packetlevel/ScalarNoise.h"

#include "LoRaBandListening.h"

namespace flora {

class LoRaAnalogModel : public ScalarAnalogModelBase
{
  public:
    const W getBackgroundNoisePower(const LoRaBandListening *listening) const;
    virtual std::ostream& printToStream(std::ostream& stream, int level, int evFlags = 0) const override;
    virtual W computeReceptionPower(const IRadio *radio, const ITransmission *transmission, const IArrival *arrival) const override;
    virtual const IReception *computeReception(const IRadio *radio, const ITransmission *transmission, const IArrival *arrival) const override;
    const INoise *computeNoise(const IListening *listening, const IInterference *interference) const override;
    virtual const ISnir *computeSNIR(const IReception *reception, const INoise *noise) const override;
};

} // namespace inet

#endif // ifndef __INET_SCALARANALOGMODEL_H

