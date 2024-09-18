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

namespace inet {

class ModelModule : public cSimpleModule
{
  protected:
    // The following redefined virtual function holds the algorithm.
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

Define_Module(ModelModule);

void ModelModule::initialize()
{
    EV_TRACE << "initializing ModelModule stage " << stage << endl;
    if (stage == INITSTAGE_LOCAL) {
       // none
    }
    getSpeedFromXML(par("modelScript"));
}

void ModelModule::getSpeedFromXML(cXMLElement *nodes) {
    // Recursively traverse the whole config file, looking for
            // speed attributes
            cXMLElementList childs = nodes->getChildren();
            for (auto& child : childs) {
                const char *speedAttr = child->getAttribute("speed");
                if (speedAttr) {
                    EV_TRACE << speed << endl;
                    double speed = atof(speedAttr);
                    if (speed > maxSpeed)
                        maxSpeed = speed;
                }
            }
}

void ModelModule::handleMessage(cMessage *msg)
{
    getSpeedFromXML(par("modelScript"));

}
}


