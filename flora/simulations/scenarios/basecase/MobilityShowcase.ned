package flora.simulations.scenarios.basecase;

import flora.RLNodes.MobileLoRaGW;
import inet.node.inet.StandardHost;
import inet.visualizer.common.IntegratedVisualizer;
import inet.applications.udpapp.UdpBasicApp;
import inet.node.inet.Router;
import inet.node.internetcloud.InternetCloud;
import flora.LoRaPhy.LoRaMedium;
import flora.LoraNode.LoRaNode;
import inet.node.inet.StandardHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth1G;

@namespace(inet);
simple NodeDemo
{
    gates:
        input in;
        output out;
}

module HostNode extends StandardHost {
    gates:
        input in;
        output out; 
        
    submodules:
        nodeDemo: NodeDemo;
     connections:
     	in --> {} --> nodeDemo.in;
     	out <-- {} <-- nodeDemo.out;
     	
}
network MobilityShowcase
{
    parameters:
        int numberOfNodes = default(0);
        int numberOfGateways = default(1);
        int networkSizeX = default(500);
        int networkSizeY = default(500);
        @display("bgb=562,417");
        @display("bgb=400,400");
    submodules:
        loRaNodes[numberOfNodes]: LoRaNode {
            @display("p=400,304");
        }
        MobileLoraGw[numberOfGateways]: MobileLoRaGW {
            
            @display("p=157,238;is=s");
        }
        LoRaMedium: LoRaMedium {
            @display("p=309,102");
        }
        networkServer: StandardHost {
            parameters:
                @display("p=49,44");
        }
        configurator: Ipv4NetworkConfigurator {
            parameters:
                assignDisjunctSubnetAddresses = false;
                @display("p=318,27");
        }
        internetCloud: InternetCloud {
            @display("p=231,44");
        }
        gwRouter[numberOfGateways]: Router {
            @display("p=231,134");
        }
        nsRouter: Router {
            @display("p=137,44");
        }
        connections:
        networkServer.ethg++ <--> Eth1G <--> nsRouter.ethg++;
        nsRouter.pppg++ <--> Eth1G <--> internetCloud.pppg++;
        for i=0..numberOfGateways-1 {
            internetCloud.pppg++ <--> Eth1G <--> gwRouter[i].pppg++;
            gwRouter[i].ethg++ <--> Eth1G <--> MobileLoraGw[i].ethg++;
        }
}