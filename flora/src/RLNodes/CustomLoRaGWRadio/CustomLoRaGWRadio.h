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

#ifndef LORA_CustomLoRaGWRadio_H_
#define LORA_CustomLoRaGWRadio_H_

#include "inet/physicallayer/wireless/common/base/packetlevel/FlatRadioBase.h"
#include "LoRaPhy/LoRaTransmitter.h"
#include "LoRaPhy/LoRaReceiver.h"
#include "LoRaPhy/LoRaTransmission.h"
#include "LoRaPhy/LoRaReception.h"
#include "LoRa/LoRaMacFrame_m.h"
#include "inet/physicallayer/wireless/common//medium/RadioMedium.h"
#include "LoRaPhy/LoRaMedium.h"
#include "inet/common/LayeredProtocolBase.h"
#include "LoRa/LoRaGWRadio.h"

namespace flora {

class CustomLoRaGWRadio : public LoRaGWRadio {

protected:
    virtual void initialize(int stage) override;
    virtual void finish() override;
};

}

#endif /* LORA_CustomLoRaGWRadio_H_ */
