import json

import matplotlib.pyplot as plt

from baselines3.sb3_to_tflite import sb3_to_tflite_pipeline
from utilities import load_config
from sim_runner import OmnetEnv


def read_log(batch, log_path):
    """
    Reads log data from a JSON file, including extracting the format from the header.
    """
    # Construct the log file path
    log_file = log_path + "_" + str(batch) + ".json"

    with open(log_file, 'r') as file:
        data = json.load(file)  # Load the JSON data
    return data


def create_jagged_line(x_data, y_data):
    """
    This function creates a jagged line by duplicating each x-coordinate with an increasing y-coordinate.
    """
    jagged_x = []
    jagged_y = []
    for i in range(len(x_data)):
        jagged_x.append(x_data[i])
        jagged_y.append(y_data[i])
        if i < len(x_data) - 1:
            jagged_x.append(x_data[i])
            jagged_y.append(y_data[i + 1])
    return jagged_x, jagged_y


import matplotlib.pyplot as plt
import math


def plot_all(data_dict):
    """
    Plots:
    A grid layout where each node has its own subplot, showing:
    - Distance from the gateway to the node over time
    - Vertical transmission lines for the node
    """
    # --- First plot: Subplots of distance to each node over time, in a grid-layout ---
    # Extract data
    distances = data_dict["gw_data"]["node_distances"]
    timestamps = data_dict["gw_data"]["times"]
    transmission_times = data_dict["transmission_times"]


    # Restructure distances into a node-wise format
    number_of_nodes = len(distances[0])
    distances_restructured = [[] for _ in range(number_of_nodes)]
    for node_idx in range(number_of_nodes):
        for distance in distances:
            distances_restructured[node_idx].append(distance[node_idx])

    # Calculate grid dimensions (square-like layout)
    cols = math.ceil(math.sqrt(number_of_nodes))
    rows = math.ceil(number_of_nodes / cols)

    # Create subplots: grid layout
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = axs.flatten()  # Flatten the 2D array for easier indexing

    # Define a color palette
    colors = plt.cm.tab10.colors  # Use matplotlib's tab10 color palette
    color_cycle = len(colors)

    for node_idx in range(number_of_nodes):
        ax = axs[node_idx]

        # Plot the distances for the node
        ax.plot(timestamps, distances_restructured[node_idx], label=f"Distance to Node {node_idx + 1}",
                color=colors[node_idx % color_cycle])

        # Overlay node-wise transmission times as vertical lines
        for time in transmission_times[node_idx]:
            ax.axvline(x=float(time), linestyle="--", alpha=0.7, color=colors[(node_idx + 1) % color_cycle],
                       label="Transmission Time" if time == transmission_times[node_idx][0] else None)

        # Customize each subplot
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Distance (meters)")
        ax.set_title(f"Node {node_idx + 1}")
        ax.legend(loc="best")
        ax.grid()

    # Remove empty subplots if any (in case of uneven grid)
    for ax in axs[number_of_nodes:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

    # --- Second plot: Packets received vs transmission times ---
    timestamps = data_dict["gw_data"]["times"]
    packets_received_mobile = data_dict["gw_data"]["number_of_received_packets_per_node"]
    # Flatten packets_received for the mobile gateway (sum over nodes)
    packet_counts_mobile = [sum(node_packets) for node_packets in packets_received_mobile]

    # Create a new figure for the second plot
    plt.figure(figsize=(10, 6))

    # Plot packets received by the mobile gateway (over all nodes)
    plt.plot(timestamps[:len(packet_counts_mobile)], packet_counts_mobile, label="Mobile GW Packets Received", color="green")

    # Overlay cumulative packets sent by all nodes
    for node_idx, node_times in enumerate(transmission_times):
        # add extra time stamp at t=0 where
        cumulative_packets = range(0, len(node_times) + 1)
        plt.step([0]+node_times, cumulative_packets, linestyle="--", alpha=0.7, where='post',
                 label=f"Node {node_idx + 1} Packets Sent")

    # Customize plot
    plt.xlabel("Time (seconds)")
    plt.ylabel("Packets Received / Transmission Times")
    plt.title("Packets Sent and Received Over Time")
    plt.legend(loc="best")
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()

    # --- Third plot: Packet Delivery Rate ---
    plt.figure(figsize=(10, 6))

    # Calculate the total number of packets sent up to each timestamp
    total_packets_sent = []
    for t in timestamps:
        total_sent = 0
        for node_times in transmission_times:
            # Count how many packets were sent up to and including time t
            total_sent += sum(1 for time in node_times if time <= t)
        total_packets_sent.append(total_sent)

    # Calculate Mobile Gateway Packet Delivery Rate (PDR) as Total Received / Total Sent
    pdr_mobile = [received / sent if sent > 0 else 0 for received, sent in zip(packet_counts_mobile, total_packets_sent)]

    # Calculate total packets received by all stationary gateways at each timestamp
    stationary_gw_reception_times = data_dict["stationary_gateway_reception_times"]
    total_packets_received_stationary = []
    for t in timestamps:
        total_received = 0
        for gateway_reception_times in stationary_gw_reception_times:
            total_received += sum(1 for reception_time in gateway_reception_times if reception_time <= t)
        total_packets_received_stationary.append(total_received)

    # Calculate PDR for stationary gateways as total_received / total_sent
    pdr_stationary = [
        received / sent if sent > 0 else 0
        for received, sent in zip(total_packets_received_stationary, total_packets_sent)
    ]
    # print(f"{total_packets_received_stationary=}\n{total_packets_sent=}")

    # Apply max smoothing: for every pdr[i], set pdr[i] = max(pdr[i], pdr[i+1])
    # This solves the issue of "dips" in PDR due to a gap in time between transmissions and reception.
    for i in range(len(pdr_mobile) - 1):  # Avoid going out of bounds
        pdr_mobile[i] = max(pdr_mobile[i], pdr_mobile[i + 1])
    for i in range(len(pdr_stationary) - 1):  # Avoid going out of bounds
        pdr_stationary[i] = max(pdr_stationary[i], pdr_stationary[i + 1])

    # Plot the Packet Delivery Rate (PDR)

    # Plot the Packet Delivery Rate (PDR) for the subsampled points
    # Plot the Mobile Gateway PDR
    plt.plot(timestamps[:len(pdr_mobile)], pdr_mobile, label="Mobile GW PDR", color="blue", linestyle=":")

    # Plot the Stationary Gateway PDR
    plt.plot(timestamps[:len(pdr_stationary)], pdr_stationary, label="Stationary GW PDR", color="green", linestyle="--")

    # Customize plot
    plt.xlabel("Time (seconds)")
    plt.ylabel("PDR")
    plt.title("Packet Delivery Rate (PDR) Over Time")
    plt.legend(loc="best")

    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the simulation and plot the results.
    """
    # Initialize OmnetEnv from the existing module
    env = OmnetEnv()

    do_export_sb3_to_tflite = False
    if do_export_sb3_to_tflite:
        sb3_to_tflite_pipeline("baselines3/stable-model-2d-best/best_model")

    config = load_config("config.json")
    # Get log path from the configuration
    log_path = config['logfile_path']
    if not log_path:
        print("Log file path is not specified in the configuration.")
        return

    # Run the simulation
    print("Starting simulation...")
    # TODO: Reenable
    env.run_simulation()

    batch = 0  # Specify the batch number if needed
    # Read the log data
    print("Reading log data...")
    data = read_log(batch, log_path)

    # Plot all figures (Gateway Position, Packets, Transmission Times)
    print("Plotting results...")
    plot_all(data)


if __name__ == "__main__":
    main()
