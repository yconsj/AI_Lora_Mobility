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

#include "LoRaReceiver.h"
#include "LoRaReception.h"
#include "inet/physicallayer/wireless/common/analogmodel/packetlevel/ScalarNoise.h"
#include "../LoRaApp/SimpleLoRaApp.h"
#include "LoRaPhyPreamble_m.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/SignalTag_m.h"

namespace flora {

Define_Module(LoRaReceiver);

LoRaReceiver::LoRaReceiver() :
    snirThreshold(NaN)
{
}

void LoRaReceiver::initialize(int stage)
{
    FlatReceiverBase::initialize(stage);
    if (stage == INITSTAGE_LOCAL)
    {
        snirThreshold = math::dB2fraction(par("snirThreshold"));
        energyDetection = mW(math::dBmW2mW(par("energyDetection")));
        if(strcmp(getParentModule()->getClassName(), "flora::LoRaGWRadio") == 0)
        {
            iAmGateway = true;
        } else iAmGateway = false;
        alohaChannelModel = par("alohaChannelModel");
        LoRaReceptionCollision = registerSignal("LoRaReceptionCollision");
        numCollisions = 0;
        rcvBelowSensitivity = 0;
    }
}

void LoRaReceiver::finish()
{
        recordScalar("numCollisions", numCollisions);
        recordScalar("rcvBelowSensitivity", rcvBelowSensitivity);

}

bool LoRaReceiver::computeIsReceptionPossible(const IListening *listening, const ITransmission *transmission) const
{
    //here we can check compatibility of LoRaTx parameters (or beeing a gateway)
    const LoRaTransmission *loRaTransmission = check_and_cast<const LoRaTransmission *>(transmission);
//    auto *loRaApp = check_and_cast<SimpleLoRaApp *>(getParentModule()->getParentModule()->getSubmodule("SimpleLoRaApp"));
    auto *loRaRadio = check_and_cast<LoRaRadio *>(getParentModule());
//    auto node = getContainingNode(this);
//    auto loRaRadio = check_and_cast<LoRaRadio *>(node->getSubmodule("LoRaNic")->getSubmodule("LoRaRadio"));
    if(iAmGateway || (loRaTransmission->getLoRaCF() == loRaRadio->loRaCF && loRaTransmission->getLoRaBW() == loRaRadio->loRaBW && loRaTransmission->getLoRaSF() == loRaRadio->loRaSF))
        return true;
    else
        return false;
}

bool LoRaReceiver::computeIsReceptionPossible(const IListening *listening, const IReception *reception, IRadioSignal::SignalPart part) const
{
    //here we can check compatibility of LoRaTx parameters (or beeing a gateway) and reception above sensitivity level
    const LoRaBandListening *loRaListening = check_and_cast<const LoRaBandListening *>(listening);
    const LoRaReception *loRaReception = check_and_cast<const LoRaReception *>(reception);
    if (iAmGateway == false && (loRaListening->getLoRaCF() != loRaReception->getLoRaCF() || loRaListening->getLoRaBW() != loRaReception->getLoRaBW() || loRaListening->getLoRaSF() != loRaReception->getLoRaSF())) {
        return false;
    } else {
        W minReceptionPower = loRaReception->computeMinPower(reception->getStartTime(part), reception->getEndTime(part));
        W sensitivity = getSensitivity(loRaReception);
        bool isReceptionPossible = minReceptionPower >= sensitivity;
        EV_DEBUG << "Computing whether reception is possible: minimum reception power = " << minReceptionPower << ", sensitivity = " << sensitivity << " -> reception is " << (isReceptionPossible ? "possible" : "impossible") << endl;
        if(isReceptionPossible == false) {
           const_cast<LoRaReceiver* >(this)->rcvBelowSensitivity++;
        }
        return isReceptionPossible;
    }
}

bool LoRaReceiver::computeIsReceptionAttempted(const IListening *listening, const IReception *reception, IRadioSignal::SignalPart part, const IInterference *interference) const
{
    if(isPacketCollided(reception, part, interference))
    {
        auto packet = reception->getTransmission()->getPacket();
        const auto &chunk = packet->peekAtFront<FieldsChunk>();
        auto loraMac = dynamicPtrCast<const LoRaMacFrame>(chunk);
        auto loraPreamble = dynamicPtrCast<const LoRaPhyPreamble>(chunk);
        MacAddress rec;
        if (loraPreamble)
            rec = loraPreamble->getReceiverAddress();
        else if (loraMac)
            rec = loraMac->getReceiverAddress();

        if (iAmGateway == false) {
            auto *macLayer = check_and_cast<LoRaMac *>(getParentModule()->getParentModule()->getSubmodule("mac"));
            if (rec == macLayer->getAddress()) {
                const_cast<LoRaReceiver* >(this)->numCollisions++;
            }
            //EV << "Node: Extracted macFrame = " << loraMacFrame->getReceiverAddress() << ", node address = " << macLayer->getAddress() << std::endl;
        } else {
            auto *gwMacLayer = check_and_cast<LoRaGWMac *>(getParentModule()->getParentModule()->getSubmodule("mac"));
            EV << "GW: Extracted macFrame = " << rec << ", node address = " << gwMacLayer->getAddress() << std::endl;
            if (rec == MacAddress::BROADCAST_ADDRESS) {
                const_cast<LoRaReceiver* >(this)->numCollisions++;
            }
        }
        return false;
    } else {
        return true;
    }
}

bool LoRaReceiver::isPacketCollided(const IReception *reception, IRadioSignal::SignalPart part, const IInterference *interference) const
{
    //auto radio = reception->getReceiver();
    //auto radioMedium = radio->getMedium();
    auto interferingReceptions = interference->getInterferingReceptions();
    const LoRaReception *loRaReception = check_and_cast<const LoRaReception *>(reception);
    simtime_t m_x = (loRaReception->getStartTime() + loRaReception->getEndTime())/2;
    simtime_t d_x = (loRaReception->getEndTime() - loRaReception->getStartTime())/2;
    EV << "Czas transmisji to " << loRaReception->getEndTime() - loRaReception->getStartTime() << endl;
    double P_threshold = 6;
    W signalRSSI_w = loRaReception->getPower();
    double signalRSSI_mw = signalRSSI_w.get()*1000;
    double signalRSSI_dBm = math::mW2dBmW(signalRSSI_mw);
    EV << signalRSSI_mw << endl;
    EV << signalRSSI_dBm << endl;
    int receptionSF = loRaReception->getLoRaSF();
    for (auto interferingReception : *interferingReceptions) {
        bool overlap = false;
        bool frequencyCollision = false;
        bool captureEffect = false;
        bool timingCollision = false; //Collision is acceptable in first part of preamble
        const LoRaReception *loRaInterference = check_and_cast<const LoRaReception *>(interferingReception);

        simtime_t m_y = (loRaInterference->getStartTime() + loRaInterference->getEndTime())/2;
        simtime_t d_y = (loRaInterference->getEndTime() - loRaInterference->getStartTime())/2;
        if(omnetpp::fabs(m_x - m_y) < d_x + d_y)
        {
            overlap = true;
        }

        if(loRaReception->getLoRaCF() == loRaInterference->getLoRaCF())
        {
            frequencyCollision = true;
        }

        W interferenceRSSI_w = loRaInterference->getPower();
        double interferenceRSSI_mw = interferenceRSSI_w.get()*1000;
        double interferenceRSSI_dBm = math::mW2dBmW(interferenceRSSI_mw);
        int interferenceSF = loRaInterference->getLoRaSF();

        /* If difference in power between two signals is greater than threshold, no collision*/
        if(signalRSSI_dBm - interferenceRSSI_dBm >= nonOrthDelta[receptionSF-7][interferenceSF-7])
        {
            captureEffect = true;
        }

        EV << "[MSDEBUG] Received packet at SF: " << receptionSF << " with power " << signalRSSI_dBm << endl;
        EV << "[MSDEBUG] Received interference at SF: " << interferenceSF << " with power " << interferenceRSSI_dBm << endl;
        EV << "[MSDEBUG] Acceptable diff is equal " << nonOrthDelta[receptionSF-7][interferenceSF-7] << endl;
        EV << "[MSDEBUG] Diff is equal " << signalRSSI_dBm - interferenceRSSI_dBm << endl;
        if (captureEffect == false)
        {
            EV << "[MSDEBUG] Packet is discarded" << endl;
        } else
            EV << "[MSDEBUG] Packet is not discarded" << endl;

        /* If last 6 symbols of preamble are received, no collision*/
        double nPreamble = 8; //from the paper "Do Lora networks..."
        simtime_t Tsym = (pow(2, loRaReception->getLoRaSF()))/(loRaReception->getLoRaBW().get()/1000)/1000;
        simtime_t csBegin = loRaReception->getPreambleStartTime() + Tsym * (nPreamble - 6);
        if(csBegin < loRaInterference->getEndTime())
        {
            timingCollision = true;
        }

        if (overlap && frequencyCollision)
        {
            if(alohaChannelModel == true)
            {
                if(iAmGateway && (part == IRadioSignal::SIGNAL_PART_DATA || part == IRadioSignal::SIGNAL_PART_WHOLE)) const_cast<LoRaReceiver* >(this)->emit(LoRaReceptionCollision, true);
                return true;
            }
            if(alohaChannelModel == false)
            {
                if(captureEffect == false && timingCollision)
                {
                    if(iAmGateway && (part == IRadioSignal::SIGNAL_PART_DATA || part == IRadioSignal::SIGNAL_PART_WHOLE)) const_cast<LoRaReceiver* >(this)->emit(LoRaReceptionCollision, true);
                    return true;
                }
            }

        }
    }
    return false;
}

const IReceptionDecision *LoRaReceiver::computeReceptionDecision(const IListening *listening, const IReception *reception, IRadioSignal::SignalPart part, const IInterference *interference, const ISnir *snir) const
{
    auto isReceptionPossible = computeIsReceptionPossible(listening, reception, part);
    auto isReceptionAttempted = isReceptionPossible && computeIsReceptionAttempted(listening, reception, part, interference);
    auto isReceptionSuccessful = isReceptionAttempted && computeIsReceptionSuccessful(listening, reception, part, interference, snir);
    return new ReceptionDecision(reception, part, isReceptionPossible, isReceptionAttempted, isReceptionSuccessful);
}

Packet *LoRaReceiver::computeReceivedPacket(const ISnir *snir, bool isReceptionSuccessful) const
{
    auto transmittedPacket = snir->getReception()->getTransmission()->getPacket();
    auto receivedPacket = transmittedPacket->dup();
    receivedPacket->clearTags();
//    receivedPacket->addTag<PacketProtocolTag>()->setProtocol(transmittedPacket->getTag<PacketProtocolTag>()->getProtocol());
    if (!isReceptionSuccessful)
        receivedPacket->setBitError(true);
    return receivedPacket;
}

const IReceptionResult *LoRaReceiver::computeReceptionResult(const IListening *listening, const IReception *reception, const IInterference *interference, const ISnir *snir, const std::vector<const IReceptionDecision *> *decisions) const
{
    bool isReceptionSuccessful = true;
    for (auto decision : *decisions)
        isReceptionSuccessful &= decision->isReceptionSuccessful();
    auto packet = computeReceivedPacket(snir, isReceptionSuccessful);

    auto signalPowerInd = packet->addTagIfAbsent<SignalPowerInd>();
    const LoRaReception *loRaReception = check_and_cast<const LoRaReception *>(reception);
    W signalRSSI_w = loRaReception->getPower();
    signalPowerInd->setPower(signalRSSI_w);

    auto snirInd = packet->addTagIfAbsent<SnirInd>();
    snirInd->setMinimumSnir(snir->getMin());
    snirInd->setMaximumSnir(snir->getMax());
    auto signalTimeInd = packet->addTagIfAbsent<SignalTimeInd>();
    signalTimeInd->setStartTime(reception->getStartTime());
    signalTimeInd->setEndTime(reception->getEndTime());
    auto errorRateInd = packet->addTagIfAbsent<ErrorRateInd>();
    errorRateInd->setPacketErrorRate(errorModel ? errorModel->computePacketErrorRate(snir, IRadioSignal::SIGNAL_PART_WHOLE) : 0.0);
    errorRateInd->setBitErrorRate(errorModel ? errorModel->computeBitErrorRate(snir, IRadioSignal::SIGNAL_PART_WHOLE) : 0.0);
    errorRateInd->setSymbolErrorRate(errorModel ? errorModel->computeSymbolErrorRate(snir, IRadioSignal::SIGNAL_PART_WHOLE) : 0.0);

    return new ReceptionResult(reception, decisions, packet);
}

bool LoRaReceiver::computeIsReceptionSuccessful(const IListening *listening, const IReception *reception, IRadioSignal::SignalPart part, const IInterference *interference, const ISnir *snir) const
{
    return true;
    //we don't check the SINR level, it is done in collision checking by P_threshold level evaluation
}

const IListening *LoRaReceiver::createListening(const IRadio *radio, const simtime_t startTime, const simtime_t endTime, const Coord &startPosition, const Coord &endPosition) const
{
    if(iAmGateway == false)
    {
        auto node = getContainingNode(this);
//        auto loRaApp = check_and_cast<SimpleLoRaApp *>(node->getSubmodule("SimpleLoRaApp"));
        auto loRaRadio = check_and_cast<LoRaRadio *>(node->getSubmodule("LoRaNic")->getSubmodule("radio"));
        return new LoRaBandListening(radio, startTime, endTime, startPosition, endPosition, loRaRadio->loRaCF, loRaRadio->loRaBW, loRaRadio->loRaSF);
    }
    else {
        return new LoRaBandListening(radio, startTime, endTime, startPosition, endPosition, LoRaCF, LoRaBW, LoRaSF);
    }
}

const IListeningDecision *LoRaReceiver::computeListeningDecision(const IListening *listening, const IInterference *interference) const
{
    const IRadio *receiver = listening->getReceiver();
    const IRadioMedium *radioMedium = receiver->getMedium();
    const IAnalogModel *analogModel = radioMedium->getAnalogModel();
    const INoise *noise = analogModel->computeNoise(listening, interference);
    const ScalarNoise *loRaNoise = check_and_cast<const ScalarNoise *>(noise);
    W maxPower = loRaNoise->computeMaxPower(listening->getStartTime(), listening->getEndTime());
    bool isListeningPossible = maxPower >= energyDetection;
    delete noise;
    EV_DEBUG << "Computing whether listening is possible: maximum power = " << maxPower << ", energy detection = " << energyDetection << " -> listening is " << (isListeningPossible ? "possible" : "impossible") << endl;
    return new ListeningDecision(listening, isListeningPossible);
}

W LoRaReceiver::getSensitivity(const LoRaReception *reception) const
{
    //function returns sensitivity -- according to LoRa documentation, it changes with LoRa parameters
    //Sensitivity values from Semtech SX1272/73 datasheet, table 10, Rev 3.1, March 2017
    W sensitivity = W(math::dBmW2mW(-126.5) / 1000);
    if(reception->getLoRaSF() == 6)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-121) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-118) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-111) / 1000);
    }

    if (reception->getLoRaSF() == 7)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-124) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-122) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-116) / 1000);
    }

    if(reception->getLoRaSF() == 8)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-127) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-125) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-119) / 1000);
    }
    if(reception->getLoRaSF() == 9)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-130) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-128) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-122) / 1000);
    }
    if(reception->getLoRaSF() == 10)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-133) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-130) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-125) / 1000);
    }
    if(reception->getLoRaSF() == 11)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-135) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-132) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-128) / 1000);
    }
    if(reception->getLoRaSF() == 12)
    {
        if(reception->getLoRaBW() == Hz(125000)) sensitivity = W(math::dBmW2mW(-137) / 1000);
        if(reception->getLoRaBW() == Hz(250000)) sensitivity = W(math::dBmW2mW(-135) / 1000);
        if(reception->getLoRaBW() == Hz(500000)) sensitivity = W(math::dBmW2mW(-129) / 1000);
    }
    return sensitivity;
}

}
