import json
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from sim_runner import OmnetEnv

# Get the absolute path of the parent directory (python_scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "baselines3"))
# Add "baselines3" directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from baselines3.utilities import load_config, jains_fairness_index
from baselines3.sb3_to_tflite import sb3_to_tflite_pipeline


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


def max_smooth(arr, axis=None):
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    if arr.ndim == 2 and axis is not None:
        return np.maximum(arr, np.roll(arr, shift=-1, axis=axis))
    elif arr.ndim == 1:
        return np.maximum(arr[:-1], arr[1:])
    return arr  # Return unchanged if it's not 1D or 2D with axis specified


def extract_episode_stats(data_dict: dict):
    # Extract data
    distances = data_dict["mobile_gw_data"]["node_distances"]
    timestamps = data_dict["mobile_gw_data"]["times"]
    transmission_times = data_dict["nodes"]["transmission_times"]

    number_of_nodes = data_dict["static"]["number_of_nodes"]
    packets_received_mobile_per_node = np.array(data_dict["mobile_gw_data"]["number_of_received_packets_per_node"])
    packets_received_mobile = packets_received_mobile_per_node.sum(axis=1)

    # Calculate the total number of packets sent up to each timestamp
    packets_sent_data = data_dict["nodes"]["transmissions_per_node"]

    # Sum up the total packets sent across all nodes for each timestamp
    packets_sent_per_node = np.array(packets_sent_data)
    total_packets_sent = packets_sent_per_node.sum(axis=1)

    # Calculate Mobile Gateway Packet Delivery Rate (PDR)
    pdr_mobile_per_node = np.divide(packets_received_mobile_per_node, packets_sent_per_node,
                                    where=packets_sent_per_node > 0,
                                    out=np.zeros_like(packets_sent_per_node, dtype=float))
    # Calculate total PDR for mobile gateway
    pdr_mobile = np.divide(packets_received_mobile, total_packets_sent, where=total_packets_sent > 0,
                           out=np.zeros_like(total_packets_sent, dtype=float))

    fairness_mobile = [jains_fairness_index(received, sent) for received, sent in zip(packets_received_mobile_per_node,
                                                                                      packets_sent_per_node)]

    # Process stationary gateway reception
    stationary_gw_reception_times = data_dict["stationary_gw_data"]["stationary_gateway_reception_times"]
    packets_received_per_node_stationary = np.array(
        data_dict["stationary_gw_data"]["stationary_gw_number_of_received_packets_per_node"])

    total_packets_received_stationary = packets_received_per_node_stationary.sum(axis=1)

    # Calculate PDR for stationary gateways
    pdr_stationary_per_node = np.divide(packets_received_per_node_stationary, packets_sent_per_node,
                                        where=packets_sent_per_node > 0,
                                        out=np.zeros_like(packets_sent_per_node, dtype=float))
    # Calculate total PDR for stationary gateways
    pdr_stationary = np.divide(total_packets_received_stationary, total_packets_sent, where=total_packets_sent > 0,
                               out=np.zeros_like(total_packets_sent, dtype=float))
    fairness_stationary = [jains_fairness_index(received, sent)
                           for received, sent in zip(packets_received_per_node_stationary, packets_sent_per_node)]

    # Apply smoothing
    pdr_mobile = max_smooth(pdr_mobile)
    pdr_mobile_per_node = max_smooth(pdr_mobile_per_node, axis=0)
    pdr_stationary = max_smooth(pdr_stationary)
    fairness_stationary = max_smooth(fairness_stationary)
    fairness_mobile = max_smooth(fairness_mobile)

    stats = {
        "number_of_nodes": number_of_nodes,
        "distances": distances,
        "transmission_times": transmission_times,
        "packets_sent": total_packets_sent.tolist(),
        "packets_sent_per_node": packets_sent_per_node.tolist(),
        "packets_received_mobile": packets_received_mobile.tolist(),
        "packets_received_mobile_per_node": packets_received_mobile_per_node.tolist(),
        "pdr_mobile": pdr_mobile.tolist(),
        "pdr_mobile_per_node": pdr_mobile_per_node.tolist(),
        "fairness_mobile": fairness_mobile,
        "packets_received_stationary": total_packets_received_stationary.tolist(),
        "packets_received_per_node_stationary": packets_received_per_node_stationary.tolist(),
        "pdr_stationary": pdr_stationary.tolist(),
        "pdr_stationary_per_node": pdr_stationary_per_node.tolist(),
        "fairness_stationary": fairness_stationary,
        "data_timestamps": timestamps,
    }
    #print(f"{stats = }")
    return stats


def plot_omnet_episode(stats_dict, include_stationary=True):
    """
    Main function to plot all the stats, controlling whether to include stationary gateway data.
    """
    # Plot the distances from gateway to each node
    plot_distances(stats_dict)

    # Plot packets received and transmission times
    plot_packets_received(stats_dict, include_stationary)

    # Plot PDR and fairness
    plot_pdr_fairness(stats_dict, include_stationary)


def plot_distances(stats_dict):
    """
    Plot distance from the gateway to each node over time, with vertical lines indicating transmission times.
    """
    distances = stats_dict["distances"]
    number_of_nodes = stats_dict["number_of_nodes"]
    timestamps = stats_dict["data_timestamps"]
    transmission_times = stats_dict["transmission_times"]

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

        # Extract distance values for the current node (no transposing needed)
        node_distances = [distances[t][node_idx] for t in range(len(timestamps))]

        # Plot the distances for the node
        ax.plot(timestamps, node_distances, label=f"Distance to Node {node_idx + 1}",
                color=colors[node_idx % color_cycle])

        # Overlay node-wise transmission times as vertical lines
        for time in transmission_times[node_idx]:
            ax.axvline(x=float(time), linestyle="--", alpha=0.7, color='black',
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


def plot_packets_received(stats_dict, include_stationary=True):
    """
    Plot packets received by the mobile gateway, total packets sent, and cumulative packets sent by each node.
    Optionally include data for stationary gateways.
    """
    packets_sent_per_node = stats_dict["packets_sent_per_node"]
    total_packets_sent = stats_dict["packets_sent"]
    packets_received_mobile = stats_dict["packets_received_mobile"]
    packets_received_mobile_per_node = stats_dict["packets_received_mobile_per_node"]
    packets_received_stationary_per_node = stats_dict.get("packets_received_stationary_per_node", [])
    timestamps = stats_dict["data_timestamps"]
    transmission_times = stats_dict["transmission_times"]

    # Create a new figure for the second plot
    plt.figure(figsize=(10, 6))

    # Plot packets received by the mobile gateway (over all nodes)
    plt.plot(timestamps[:len(packets_received_mobile)], packets_received_mobile, label="Mobile GW Packets Received",
             color="green")
    plt.plot(timestamps[:len(total_packets_sent)], total_packets_sent, label="Total packets sent", color="black",
             linestyle="--")

    # Plot packets received by each stationary node if `include_stationary` is True
    if include_stationary and packets_received_stationary_per_node:
        for node_idx, node_packets in enumerate(packets_received_stationary_per_node):
            plt.plot(timestamps[:len(node_packets)], node_packets, label=f"Stationary Node {node_idx + 1} Packets Received",
                     linestyle=":", color=f"C{(node_idx + 1) % 10}")

    # Overlay cumulative packets sent by all nodes
    for node_idx, node_times in enumerate(transmission_times):
        # Add extra time stamp at t=0 where number of packets transmitted is 0,
        cumulative_packets = list(range(0, len(node_times) + 1))
        plt.step([0] + node_times, cumulative_packets, linestyle="--", alpha=0.7, where='post',
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


def plot_pdr_fairness(stats_dict, include_stationary=True):
    """
    Plot PDR and fairness for both mobile and stationary gateways.
    """
    timestamps = stats_dict["data_timestamps"]
    pdr_mobile = stats_dict.get("pdr_mobile", [])
    fairness_mobile = stats_dict.get("fairness_mobile", [])
    pdr_stationary = stats_dict.get("pdr_stationary", [])
    fairness_stationary = stats_dict.get("fairness_stationary", [])

    plt.figure(figsize=(10, 6))

    # Plot the Mobile Gateway PDR
    plt.plot(timestamps[:len(pdr_mobile)], pdr_mobile, label="Mobile GW PDR", color="blue", linestyle="--")
    # Plot the Stationary Gateway Fairness if include_stationary is True
    if include_stationary:
        plt.plot(timestamps[:len(fairness_mobile)], fairness_mobile, label="Mobile GW Fairness", color="blue", linestyle=":")
        plt.plot(timestamps[:len(pdr_stationary)], pdr_stationary, label="Stationary GW PDR", color="green", linestyle="--")
        plt.plot(timestamps[:len(fairness_stationary)], fairness_stationary, label="Stationary GW Fairness", color="green", linestyle=":")
    else:
        plt.plot(timestamps[:len(fairness_mobile)], fairness_mobile, label="Mobile GW Fairness", color="blue", linestyle=":")

    # Customize plot
    plt.xlabel("Time (seconds)")
    plt.ylabel("PDR")
    plt.title("Packet Delivery Rate (PDR) Over Time")
    plt.legend(loc="best")

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_batch_performance(final_pdr_mobile_per_node_list, final_pdr_stationary_per_node_list,
                           final_pdr_mobile_list, final_pdr_stationary_list,
                           final_fairness_mobile_list, final_fairness_stationary_list,
                           include_stationary=True):
    assert len(final_pdr_mobile_per_node_list) == len(final_pdr_stationary_per_node_list), \
        f"{len(final_pdr_mobile_per_node_list)} and {len(final_pdr_stationary_per_node_list)} must be equal."
    assert len(final_fairness_mobile_list) == len(final_fairness_stationary_list), \
        f"{len(final_fairness_mobile_list)} and {len(final_fairness_stationary_list)} must be equal."

    # Scale values to percentage
    final_pdr_mobile_list = [x * 100 for x in final_pdr_mobile_list]
    final_fairness_mobile_list = [x * 100 for x in final_fairness_mobile_list]

    if include_stationary:
        final_pdr_stationary_list = [x * 100 for x in final_pdr_stationary_list]
        final_fairness_stationary_list = [x * 100 for x in final_fairness_stationary_list]

    # Box plot setup
    fig2, ax3 = plt.subplots(figsize=(6, 4))

    # Construct data for box plot
    box_data = [final_pdr_mobile_list, final_fairness_mobile_list]
    tick_labels = ["PDR (Mobile)", "Fairness (Mobile)"]

    if include_stationary:
        box_data.insert(1, final_pdr_stationary_list)
        box_data.append(final_fairness_stationary_list)
        tick_labels.insert(1, "PDR (Stationary)")
        tick_labels.append("Fairness (Stationary)")

    ax3.boxplot(box_data,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                flierprops=dict(marker="o", color="red", alpha=0.6))

    ax3.set_xticklabels(tick_labels)
    ax3.set_ylim(0, 100)
    ax3.set_title("Box Plot: PDR and Fairness Distribution")
    ax3.set_ylabel("Values (%)")
    plt.tight_layout()
    plt.show()

    # Bar plot for PDR per node
    fig3, ax4 = plt.subplots(figsize=(8, 5))

    num_nodes = len(final_pdr_mobile_per_node_list[0])
    pdr_mobile_per_node = np.mean(final_pdr_mobile_per_node_list, axis=0) * 100
    pdr_std_mobile = np.std(final_pdr_mobile_per_node_list, axis=0) * 100

    x_indices = np.arange(num_nodes)
    bar_width = 0.35

    bars_mobile = ax4.bar(x_indices - bar_width / 2, pdr_mobile_per_node, yerr=pdr_std_mobile, capsize=5,
                          color='steelblue', alpha=0.7, width=bar_width, label="Mobile")

    if include_stationary:
        pdr_stationary_per_node = np.mean(final_pdr_stationary_per_node_list, axis=0) * 100
        pdr_std_stationary = np.std(final_pdr_stationary_per_node_list, axis=0) * 100

        bars_stationary = ax4.bar(x_indices + bar_width / 2, pdr_stationary_per_node, yerr=pdr_std_stationary,
                                  capsize=5,
                                  color='lightgreen', alpha=0.7, width=bar_width, label="Stationary")

    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([f"Node {i + 1}" for i in range(num_nodes)])

    # Annotate bars
    for bar, pdr in zip(bars_mobile, pdr_mobile_per_node):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{pdr:.1f}%", ha="center", fontsize=10,
                 fontweight="bold")

    if include_stationary:
        for bar, pdr in zip(bars_stationary, pdr_stationary_per_node):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{pdr:.1f}%", ha="center", fontsize=10,
                     fontweight="bold")

    ax4.set_xlabel("Nodes")
    ax4.set_ylabel("PDR (%)")
    fig3.suptitle(f"PDR for Each Node ({'Mobile vs Stationary' if include_stationary else 'Mobile'})", fontsize=14, fontweight="bold", x=0.5, y=0.95)
    ax4.set_ylim(0, 100)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    # Display Overall PDR & Jain's Fairness
    overall_pdr_mobile = np.mean(final_pdr_mobile_list)
    overall_fairness_mobile = np.mean(final_fairness_mobile_list)

    ax4.text(num_nodes - 0.3, 70, f"Jain's Fairness (Mobile) = {overall_fairness_mobile:.2f}",
             color="purple", fontsize=12, fontweight="bold")
    ax4.text(num_nodes - 0.3, 60, f"Overall PDR (Mobile) = {overall_pdr_mobile:.2f}%",
             fontsize=12, fontweight="bold")

    if include_stationary:
        overall_pdr_stationary = np.mean(final_pdr_stationary_list)
        overall_fairness_stationary = np.mean(final_fairness_stationary_list)

        ax4.text(num_nodes - 0.3, 50, f"Jain's Fairness (Stationary) = {overall_fairness_stationary:.2f}",
                 color="green", fontsize=12, fontweight="bold")
        ax4.text(num_nodes - 0.3, 40, f"Overall PDR (Stationary) = {overall_pdr_stationary:.2f}%",
                 fontsize=12, fontweight="bold")

    plt.subplots_adjust(top=0.85)
    ax4.legend()
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

    include_stationary = False
    batch_size = 10
    # Run the simulation
    print("Starting simulation...")
    env.run_simulation(batch_size=batch_size)

    final_pdr_mobile_per_node_list = []
    final_pdr_mobile_list = []
    final_fairness_mobile_list = []

    final_pdr_stationary_per_node_list = []
    final_pdr_stationary_list = []
    final_fairness_stationary_list = []

    print("Reading log data...")
    for batch_idx in range(batch_size):
        # Read the log data, process it
        data = extract_episode_stats(read_log(batch_idx, log_path))

        # Retrieve the final PDR for mobile and stationary gateways
        final_pdr_mobile_per_node = data["pdr_mobile_per_node"][-1]  # Last value of mobile PDR
        final_pdr_mobile = data["pdr_mobile"][-1]
        final_pdr_stationary_per_node = data["pdr_stationary_per_node"][-1]  # Last value of stationary PDR
        final_pdr_stationary = data["pdr_stationary"][-1]

        # Retrieve the final fairness for mobile and stationary gateways
        final_fairness_mobile = data["fairness_mobile"][-1]  # Last value of mobile fairness
        final_fairness_stationary = data["fairness_stationary"][-1]  # Last value of stationary fairness

        # Append the final values to their respective lists
        final_pdr_mobile_per_node_list.append(final_pdr_mobile_per_node)
        final_pdr_mobile_list.append(final_pdr_mobile)
        final_fairness_mobile_list.append(final_fairness_mobile)

        final_pdr_stationary_per_node_list.append(final_pdr_stationary_per_node)
        final_pdr_stationary_list.append(final_pdr_stationary)
        final_fairness_stationary_list.append(final_fairness_stationary)

        if batch_idx + 1 == batch_size:  # Only do these plots for the last episode in the batch
            # Plot all figures (Gateway Position, Packets, Transmission Times)
            print("Plotting results...")
            plot_omnet_episode(data, include_stationary=include_stationary)

    # After reading and processing all batches, plot the batch performance (PDR and fairness)
    plot_batch_performance(final_pdr_mobile_per_node_list, final_pdr_stationary_per_node_list,
                           final_pdr_mobile_list, final_pdr_stationary_list,
                           final_fairness_mobile_list, final_fairness_stationary_list,
                           include_stationary=include_stationary
                           )


if __name__ == "__main__":
    main()
