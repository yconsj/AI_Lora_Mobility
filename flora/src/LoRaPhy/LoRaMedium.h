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
#ifndef LORAPHY_LORAMEDIUM_H_
#define LORAPHY_LORAMEDIUM_H_
#include "inet/physicallayer/wireless/common/medium/RadioMedium.h"
#include "LoRa/LoRaRadio.h"
#include "../LoRa/LoRaMacFrame_m.h"

#include "inet/common/IntervalTree.h"
#include "inet/environment/contract/IMaterialRegistry.h"
#include "inet/environment/contract/IPhysicalEnvironment.h"
#include "inet/linklayer/common/MacAddress.h"
#include "inet/physicallayer/wireless/common/medium/CommunicationLog.h"
#include "inet/physicallayer/wireless/common/radio/packetlevel/Radio.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/ICommunicationCache.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IMediumLimitCache.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/INeighborCache.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IRadioMedium.h"
#include <algorithm>

namespace flora {
class LoRaMedium : public RadioMedium
{
    friend class LoRaGWRadio;
    friend class LoRaRadio;

protected:
    virtual bool matchesMacAddressFilter(const IRadio *radio, const Packet *packet) const override;
        //@}
    public:
      LoRaMedium();
      virtual ~LoRaMedium();
      //virtual const IReceptionDecision *getReceptionDecision(const IRadio *receiver, const IListening *listening, const ITransmission *transmission, IRadioSignal::SignalPart part) const override;
      virtual const IReceptionResult *getReceptionResult(const IRadio *receiver, const IListening *listening, const ITransmission *transmission) const override;
      virtual void addTransmission(const IRadio *transmitter, const ITransmission *transmission);
};
}
#endif /* LORAPHY_LORAMEDIUM_H_ */
