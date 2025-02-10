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

#ifndef __LORANETWORK_StaticMobilityGatewayLoRaApp_H_
#define __LORANETWORK_StaticMobilityGatewayLoRaApp_H_

#include <omnetpp.h>
#include <vector>

#include "LoRa/LoRaMacControlInfo_m.h"
#include "LoRa/LoRaMacFrame_m.h"
#include "LoRa/LoRaMac.h"
#include "LoRaPhy/LoRaRadioControlInfo_m.h"
#include "LoRaApp/SimpleLoRaApp.h"


#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/INETDefs.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/common/ModuleAccess.h"
#include "inet/applications/base/ApplicationPacket_m.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/SignalTag_m.h"
//#include "inet/physicallayer/wireless/common/contract/packetlevel/RadioControlInfo_m.h"

#include "inet/RL/StateLogger/StateLogger.h"
#include "inet/RL/LearningModels/AdvancedLearningModel/AdvancedLearningModel.h"


namespace flora {

class StaticMobilityGatewayLoRaApp : public cSimpleModule, public cListener
{
  protected:
    std::vector<L3Address> destAddresses;
    int localPort = -1, destPort = -1;
    // state
    UdpSocket socket;
    cMessage *selfMsg = nullptr;

  protected:
    virtual void initialize(int stage) override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
    void processLoraMACPacket(Packet *pk);
    void startUDP();
    void sendPacket();
    void setSocketOptions();
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    void receiveSignal(cComponent *source, simsignal_t signalID, intval_t value, cObject *details) override;
  public:
      simsignal_t LoRa_GWPacketReceived;
      int counterOfSentPacketsFromNodes = 0;
      int counterOfReceivedPackets = 0;
};
} //namespace inet
#endif // __LORANETWORK_StaticMobilityGatewayLoRaApp_H_
