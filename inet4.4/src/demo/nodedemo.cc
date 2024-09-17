/*
 * nodedemo.cc
 *
 *  Created on: 18. jul. 2024
 *      Author: augus
 */
#include <string.h>
#include <omnetpp.h>
#include <stdio.h>
#include "inet/mobility/RL/SimpleRLMobility.h"


namespace inet {

class NodeDemo : public omnetpp::cSimpleModule
{
  protected:
    // The following redefined virtual function holds the algorithm.
    virtual void initialize() override;
    virtual void handleMessage(omnetpp::cMessage *msg) override;
};

// The module class needs to be registered with OMNeT++
Define_Module(NodeDemo);

void NodeDemo::initialize()
{
    // Initialize is called at the beginning of the simulation.
    // To bootstrap the tic-toc-tic-toc process, one of the modules needs
    // to send the first message. Let this be `tic'.

    // Am I Tic or Toc?
    //if (getIndex() == 0) {
        // create and send first message on gate "out". "tictocMsg" is an
        // arbitrary string which will be the name of the message object.
    omnetpp::cMessage *msg = new omnetpp::cMessage("tictocMsg");
        send(msg, "out");
    //}
}

void NodeDemo::handleMessage(omnetpp::cMessage *msg)
{
    // The handleMessage() method is called whenever a message arrives
    // at the module. Here, we just send it to the other module, through
    // gate `out'. Because both `tic' and `toc' does the same, the message
    // will bounce between the two.

    cModule *pMod = getParentModule();
    EV << pMod->getSubmoduleNames()[0] << omnetpp::endl;
    cModule *mobilityMod = pMod->getSubmodule("mobility", -1);
    SimpleRLMobility *custMob = check_and_cast<SimpleRLMobility*>(mobilityMod);
    custMob->mySpeed += 0.1;

    //printf();
    //    bubble();
    //pMod.getSubmodule(name, index);

    send(msg, "out"); // send out the message
}


}

