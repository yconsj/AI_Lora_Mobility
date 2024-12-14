import json
from matplotlib import pyplot as plt, lines

def load_json_log(log_file):
    """
    Loads JSON log file and extracts relevant data for plotting.
    """
    with open(log_file, 'r') as file:
        data = json.load(file)

    # Extract data
    timestamps = [entry['step_time'] for entry in data]
    node_distances = [entry['node_distances'] for entry in data]  # Directly use node_distances from the log
    packets_received = [entry['packets_received'] for entry in data]
    packets_sent = [entry['packets_sent'] for entry in data]

    transmissions_per_node = [[] for _ in range(len(data[0]['transmissions_per_node']))]
    packets_received_per_node = [[] for _ in range(len(data[0]['packets_received_per_node']))]
    packets_sent_per_node = [[] for _ in range(len(data[0]['packets_sent_per_node']))]

    for entry in data:
        for i, transmission in enumerate(entry['transmissions_per_node']):
            if transmission:
                transmissions_per_node[i].append(entry['step_time'])
        for i, received in enumerate(entry['packets_received_per_node']):
            packets_received_per_node[i].append(received)
        for i, sent in enumerate(entry['packets_sent_per_node']):
            packets_sent_per_node[i].append(sent)

    return timestamps, node_distances, packets_received, packets_sent, transmissions_per_node, packets_received_per_node, packets_sent_per_node


def plot_mobile_gateway_with_nodes_advanced(log_file):
    """
    Plots:
    1. Distance from the Gateway to each node over time with transmission times marked.
    2. Number of packets received vs packets sent for each node, and total packets received.
    """
    # Load log data
    timestamps, node_distances, packets_received, packets_sent, transmissions_per_node, packets_received_per_node, packets_sent_per_node = load_json_log(
        log_file)

    # Colors for different nodes
    node_colors = ["red", "blue", "green", "purple"]

    # Initialize figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # --- First subplot: Distance from Gateway to each node over time ---
    # Plot distances for each node
    for idx, color in enumerate(node_colors):
        axs[0].plot(timestamps, [dist[idx] for dist in node_distances], label=f"Distance to Node {idx+1}",
                    color=color, linestyle="-")

        # Plot transmission times for each node (vertical lines)
        for time in transmissions_per_node[idx]:
            axs[0].axvline(x=time, color=color, linestyle="--", alpha=0.7)

    # Custom legend for node-specific distances and transmission times
    custom_transmission_lines = [
        lines.Line2D([], [], color=color, linestyle='--', label=f"Node {i + 1} Transmission")
        for i, color in enumerate(node_colors)
    ]
    custom_distance_lines = [
        lines.Line2D([], [], color=color, linestyle='-', label=f"Node {i + 1} Distance")
        for i, color in enumerate(node_colors)
    ]
    axs[0].legend(
        handles=custom_transmission_lines +  custom_distance_lines,
        loc="upper center",  # Position the legend outside the plot area
        bbox_to_anchor=(0.5, -0.1),  # Move legend below the plot
        borderaxespad=0.0,
        ncol=2  # Two columns for better spacing
    )
    axs[0].set_xlabel("Time (steps)")
    axs[0].set_ylabel("Distance (meters)")
    axs[0].set_title("Distance from Gateway to Nodes Over Time")
    axs[0].grid()

    # --- Second subplot: Packets received vs packets sent per node ---
    # Plot total packets received as the sum of packets_received (already included in the log)
    axs[1].plot(timestamps, packets_received, label="Total Packets Received", color="black", linestyle='-', linewidth=2)
    axs[1].plot(timestamps, packets_sent, label="Total Packets Sent", color="black", linestyle='--', linewidth=2)

    # Plot packets received and sent for each node
    for node_idx, color in enumerate(node_colors):
        axs[1].plot(timestamps, packets_received_per_node[node_idx], label=f"Node {node_idx + 1} Packets Received",
                    color=color, linestyle='-')
        # Plot transmission times for each node (vertical lines)
        for time in transmissions_per_node[node_idx]:
            axs[1].axvline(x=time, color=color, linestyle="--", alpha=0.7)

    # Calculate the total Packet Delivery Rate (PDR) for each time step
    pdr = [r / s if s != 0 else 0 for r, s in zip(packets_received, packets_sent)]

    # Create secondary y-axis for PDR
    ax2 = axs[1].twinx()
    ax2.plot(timestamps, pdr, label="Total PDR", color="black", linestyle=':', linewidth=2)

    # Define custom legend for packets received, packets sent, and PDR
    custom_lines = [
        lines.Line2D([], [], color="black", linestyle='-', label="Total Packets Received"),
        lines.Line2D([], [], color="black", linestyle='--', label="Total Packets Sent"),
        lines.Line2D([], [], color="black", linestyle=':', label="Total PDR")
    ]
    axs[1].legend(
        handles=custom_lines + [
            lines.Line2D([], [], color=color, linestyle='-', label=f"Node {i + 1} Packets Received")
            for i, color in enumerate(node_colors)
        ],
        loc="upper center",  # Place legend below the second subplot
        bbox_to_anchor=(0.5, -0.1),  # Position the legend below the plot
        borderaxespad=0.0,
        ncol=2  # Two columns for better spacing
    )

    # Set labels for the secondary axis
    ax2.set_ylabel("PDR", color="black")
    ax2.set_ylim(0, 1)

    axs[1].set_xlabel("Time (steps)")
    axs[1].set_ylabel("Packets")
    axs[1].set_title("Packets Received vs Packets Sent Per Node & PDR")
    axs[1].grid()

    # Adjust layout and make room for legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Increased space at the bottom for the legends by default
    plt.show()
