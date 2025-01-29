
package flora.simulations.scenarios.advancedcase;

import flora.RLNodes.Nodes.MobileLoRaGW;
import flora.RLNodes.Nodes.StationaryLoRaGW;
import inet.node.inet.StandardHost;
import inet.visualizer.common.IntegratedVisualizer;
import inet.applications.udpapp.UdpBasicApp;
import inet.node.inet.Router;
import inet.node.internetcloud.InternetCloud;
import flora.LoRaPhy.LoRaMedium;
import flora.RLNodes.CustomLoRaMedium.CustomLoRaMedium;
import flora.LoraNode.LoRaNode;
import inet.node.inet.StandardHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth1G;
import inet.RL.StateLogger.StateLogger;

@namespace(inet);

network AdvancedCaseNetwork
{
    parameters:
        int numberOfNodes = default(1);
        int numberOfMobileGateways = default(1);
        int numberOfStationaryGateways = default(1);
        int networkSizeX = default(500);
        int networkSizeY = default(500);
        @display("bgb=562,417");
        @display("bgb=400,400");
    submodules:
        loRaNodes[numberOfNodes]: LoRaNode {
            @display("p=400,304");
        }
        MobileLoraGw[numberOfMobileGateways]: MobileLoRaGW {
            @display("p=157,238;is=s");
        };
        StationaryLoraGw[numberOfStationaryGateways] : StationaryLoRaGW;
        stateLogger : StateLogger;
        LoRaMedium: CustomLoRaMedium { // LoRaMedium
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
        gwRouter[numberOfMobileGateways + numberOfStationaryGateways]: Router {
            @display("p=231,134");
        }
        nsRouter: Router {
            @display("p=137,44");
        }
        connections:
        networkServer.ethg++ <--> Eth1G <--> nsRouter.ethg++;
        nsRouter.pppg++ <--> Eth1G <--> internetCloud.pppg++;
        for i=0..numberOfMobileGateways-1 {
            internetCloud.pppg++ <--> Eth1G <--> gwRouter[i].pppg++;
            gwRouter[i].ethg++ <--> Eth1G <--> MobileLoraGw[i].ethg++;
        }
        for i=0..numberOfStationaryGateways-1 {
            internetCloud.pppg++ <--> Eth1G <--> gwRouter[numberOfMobileGateways + i].pppg++;
            gwRouter[numberOfMobileGateways + i].ethg++ <--> Eth1G <--> StationaryLoraGw[i].ethg++;
        }
}