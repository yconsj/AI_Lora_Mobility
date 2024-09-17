package flora.simulations.scenarios.basecase;

import inet.node.inet.StandardHost;
import inet.visualizer.common.IntegratedVisualizer;

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
        int numHosts;

        @display("bgb=400,400");
    submodules:
        host[numHosts]: HostNode {
            
            @display("i=misc/smartphone");
        }
        visualizer: IntegratedVisualizer {
            @display("p=50,50");
        }
     connections:
         //host[0].
        host[0].out --> {  delay = 100ms; } --> host[1].in;
        host[0].in <-- {  delay = 100ms; } <-- host[1].out;
        
}