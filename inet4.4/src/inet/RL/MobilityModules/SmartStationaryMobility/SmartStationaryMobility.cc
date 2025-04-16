//
// Copyright (C) 2006 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


#include "SmartStationaryMobility.h"


namespace inet {

bool SmartStationaryMobility::alreadyOptimized = false;
std::vector<Coord> SmartStationaryMobility::optimizedGatewayPositions;

Define_Module(SmartStationaryMobility);

SmartStationaryMobility::SmartStationaryMobility()
{
    initialPosition = lastPosition;
}

void SmartStationaryMobility::initialize(int stage)
{
    MobilityBase::initialize(stage);
    if (stage == INITSTAGE_LOCAL) {
        updateFromDisplayString = par("updateFromDisplayString");
        heading = deg(fmod(par("initialMovementHeading").doubleValue(), 360));
        elevation = deg(fmod(par("initialMovementElevation").doubleValue(), 360));
        direction = Quaternion(EulerAngles(heading, -elevation, rad(0))).rotate(Coord::X_AXIS);
    }
    else if (stage == INITSTAGE_SINGLE_MOBILITY) {
        initialPosition = lastPosition;
    }
    else if (stage == INITSTAGE_LAST) {
        int numGateways = getParentModule()->getVectorSize();
        if (numGateways == 0) {
            return;
        }
        if (!alreadyOptimized) {
            // This should be done statically, i.e. once for ALL "SmartStationaryMobility" modules.
            // one method is to have a check if this has been executed before, using a class attribute?

            std::vector<Coord> node_positions;
            std::vector<cModule *> nodes;
            cModule *network = getSimulation()->getSystemModule();
            const char* lora_mod_str = "loRaNodes";
            int number_of_sim_nodes = network->getSubmoduleVectorSize(lora_mod_str);
            nodes.resize(number_of_sim_nodes, nullptr);
            node_positions.resize(number_of_sim_nodes, Coord(0,0,0));
            for (int node_index = 0; node_index < number_of_sim_nodes; ++node_index) {
                cModule *lora_node_module = network->getSubmodule(lora_mod_str, node_index);
                nodes[node_index] = lora_node_module;
            }

            for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
                cModule* loRaNode = nodes[node_index];
                EV << "node_index: " << node_index << endl;

                // Retrieve LoRa application
                auto *loRaApp = check_and_cast<cModule *>(loRaNode->getSubmodule("app", 0));
                if (!loRaApp) {
                    throw cRuntimeError("Invalid LoRa application for node %d", node_index);
                }

                // access node properties
                auto *mobility = check_and_cast<StationaryMobility *>(loRaNode->getSubmodule("mobility"));

                // get stationary lora node positions
                if (!mobility) {
                    throw cRuntimeError("Error, node missing mobility module in LoRa application for node %d", node_index);
                }
                node_positions[node_index] = mobility->getCurrentPosition();
            }
            // Compute the optimal positions via the dummy heuristic function
            constexpr double maxRange = 600.0; // in meters
            optimizedGatewayPositions = computeOptimizedPositions(node_positions, numGateways, maxRange);
            alreadyOptimized = true;
        }
        // Assign position based on optimized position
        int index = getParentModule()->getIndex();

        if (index < (int)optimizedGatewayPositions.size()) {
            lastPosition = optimizedGatewayPositions[index];
            EV_INFO << "Assigned optimized position for GW[" << index << "]:  " << lastPosition << endl;
        } else {
            // Disable this gateway — mark it as unused
            lastPosition = Coord(-100, -100, 0);  // Move outside visible map
            EV_WARN << "GW[" << index << "] is unused — no associated component. Disabling." << endl;

            // Optional: visually mark disabled GWs
            getDisplayString().setTagArg("i", 1, "gray");  // icon color
            getDisplayString().setTagArg("t", 0, "DISABLED");
        }

    }
}

// Dummy heuristic function: computes evenly spaced positions along the X-axis of the bounding box
std::vector<Coord> SmartStationaryMobility::computeOptimizedPositions(const std::vector<Coord>& nodePositions, int numGateways, double maxRange)
{
    std::vector<Coord> gatewayPositions;
    int N = nodePositions.size();
    if (N == 0 || maxRange <= 0 || numGateways <= 0)
        return gatewayPositions;

    double connectionThreshold = 2.0 * maxRange;
    std::vector<bool> assigned(N, false);
    std::vector<std::vector<int>> clusters;

    // Greedy clique-building algorithm
    for (int i = 0; i < N; ++i) {
        if (assigned[i]) continue;

        std::vector<int> cluster;
        cluster.push_back(i);
        assigned[i] = true;

        for (int j = 0; j < N; ++j) {
            if (assigned[j]) continue;

            bool fitsCluster = true;
            for (int idx : cluster) {
                if (nodePositions[j].distance(nodePositions[idx]) > connectionThreshold) {
                    fitsCluster = false;
                    break;
                }
            }

            if (fitsCluster) {
                cluster.push_back(j);
                assigned[j] = true;
            }
        }

        clusters.push_back(cluster);
    }

    // Sort clusters by size (desc), and place gateways at centroids of largest clusters
    std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size();
    });

    for (int i = 0; i < std::min(numGateways, (int)clusters.size()); ++i) {
        const auto& cluster = clusters[i];
        double sumX = 0, sumY = 0, sumZ = 0;

        for (int idx : cluster) {
            sumX += nodePositions[idx].x;
            sumY += nodePositions[idx].y;
            sumZ += nodePositions[idx].z;
        }

        Coord centroid(sumX / cluster.size(), sumY / cluster.size(), sumZ / cluster.size());
        gatewayPositions.push_back(centroid);

        EV_INFO << "Placed GW at centroid of cluster[" << i << "] (size " << cluster.size() << "): " << centroid << endl;
    }

    EV_INFO << "Final # of GW positions assigned: " << gatewayPositions.size() << endl;
    return gatewayPositions;
}

} // namespace inet

