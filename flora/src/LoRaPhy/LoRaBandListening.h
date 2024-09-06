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

#ifndef LORAPHY_LORABANDLISTENING_H_
#define LORAPHY_LORABANDLISTENING_H_

#include "inet/physicallayer/wireless/common/radio/packetlevel/BandListening.h"
#include "inet/physicallayer/wireless/common/base/packetlevel/ListeningBase.h"

using namespace inet;
using namespace inet::physicallayer;
namespace flora {

class LoRaBandListening : public BandListening
{
  protected:
    const int LoRaSF;


  public:
    LoRaBandListening(const IRadio *radio, simtime_t startTime, simtime_t endTime, Coord startPosition, Coord endPosition, Hz carrierFrequency, Hz bandwidth, int LoRaSF);

    virtual std::ostream& printToStream(std::ostream& stream, int level, int evFlags = 0) const override;

    //virtual Hz getCarrierFrequency() const { return LoRaCF; }
    //virtual Hz getBandwidth() const { return LoRaBW; }

    virtual Hz getLoRaCF() const { return centerFrequency; }
    virtual int getLoRaSF() const { return LoRaSF; }
    virtual Hz getLoRaBW() const { return bandwidth; }
};

} // namespace inet

#endif /* LORAPHY_LORABANDLISTENING_H_ */
