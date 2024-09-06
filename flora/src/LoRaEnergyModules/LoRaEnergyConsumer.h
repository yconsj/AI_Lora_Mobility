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

#ifndef LORAENERGYMODULES_LORAENERGYCONSUMER_H_
#define LORAENERGYMODULES_LORAENERGYCONSUMER_H_

#include "inet/physicallayer/wireless/common/energyconsumer/StateBasedEpEnergyConsumer.h"
#include "inet/power/storage/IdealEpEnergyStorage.h"
#include <map>
#include "inet/common/ModuleAccess.h"

using namespace inet;

namespace flora {

class LoRaEnergyConsumer: public inet::physicallayer::StateBasedEpEnergyConsumer {
public:
    void initialize(int stage) override;
    void finish() override;
    virtual W getPowerConsumption() const override;
    bool readConfigurationFile();
    virtual void receiveSignal(cComponent *source, simsignal_t signal, intval_t value, cObject *details) override;

protected:
    int energyConsumerId;
    double totalEnergyConsumed;
    J energyBalance = J(NaN);
    simtime_t lastEnergyBalanceUpdate = -1;
    W lastPowerConsumption = W(0);
    // All supply currents to be define in mA
    double receiverReceivingSupplyCurrent;
    double receiverBusySupplyCurrent;
    double standbySupplyCurrent;
    double idleSupplyCurrent;
    double sleepSupplyCurrent;
    double supplyVoltage;
    // map between txPower (dBm) and supply current (mA)
    std::map<double, double> transmitterTransmittingSupplyCurrent;


};

}
#endif /* LORAENERGYMODULES_LORAENERGYCONSUMER_H_ */
