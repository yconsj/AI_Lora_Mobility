/*
 * LoRaTransmission.h
 *
 *  Created on: Feb 17, 2017
 *      Author: slabicm1
 */

#ifndef LORATRANSMISSION_H_
#define LORATRANSMISSION_H_

#include "inet/physicallayer/wireless/common/base/packetlevel/TransmissionBase.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IRadioSignal.h"

using namespace inet;
using namespace inet::physicallayer;
namespace flora {
class LoRaTransmission : public TransmissionBase, public virtual INarrowbandSignal, public virtual IScalarSignal
{
protected:
    const W LoRaTP;
    const Hz LoRaCF;
    const int LoRaSF;
    const Hz LoRaBW;
    const int LoRaCR;
public:
    LoRaTransmission(const IRadio *transmitter, const Packet *macFrame, const simtime_t startTime, const simtime_t endTime, const simtime_t preambleDuration, const simtime_t headerDuration, const simtime_t dataDuration, const Coord startPosition, const Coord endPosition, const Quaternion startOrientation, const Quaternion endOrientation, W LoRaTP, Hz LoRaCF, int LoRaSF, Hz LoRaBW, int LoRaCR);

    virtual Hz getCenterFrequency() const override { return LoRaCF; }
    virtual Hz getBandwidth() const override { return LoRaBW; }
    virtual W getPower() const override { return LoRaTP; }
    virtual W computeMinPower(const simtime_t startTime, const simtime_t endTime) const override { return LoRaTP; }

    W getLoRaTP() const { return LoRaTP; }
    Hz getLoRaCF() const { return LoRaCF; }
    int getLoRaSF() const { return LoRaSF; }
    Hz getLoRaBW() const { return LoRaBW; }
    int getLoRaCR() const { return LoRaCR; }

    virtual std::ostream& printToStream(std::ostream& stream, int level, int evFlags = 0) const override;
};

} /* namespace inet */

#endif /* LORATRANSMISSION_H_ */
