import json
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_json_log(log_file):
    """
    Loads JSON log file and extracts relevant data for plotting.
    """
    with open(log_file, 'r') as file:
        data = json.load(file)

    # Extract data
    gw_position_x = [entry['gw_pos_x'] for entry in data]
    gw_position_y = [entry['gw_pos_y'] for entry in data]
    gw_positions = zip(gw_position_x, gw_position_y)
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

    return gw_positions, timestamps, node_distances, packets_received, packets_sent, \
           transmissions_per_node, packets_received_per_node, packets_sent_per_node


def plot_mobile_gateway_with_nodes_advanced(log_file):
    """
    Plots:
    1. Distance from the Gateway to each node over time with transmission times marked.
    2. Number of packets received vs packets sent for each node, and total packets received.
    """
    # Load log data
    gw_positions, timestamps, \
        node_distances, packets_received, packets_sent, \
        transmissions_per_node, packets_received_per_node, packets_sent_per_node = \
        load_json_log(log_file)

    # Colors for different nodes
    node_colors = ["tab:red", "xkcd:bluish", "xkcd:dark grass green", "tab:orange"]

    # Initialize figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # --- First subplot: Distance from Gateway to each node over time ---
    # Plot distances for each node
    for idx, color in enumerate(node_colors):
        axs[0].plot(timestamps, [dist[idx] for dist in node_distances], label=f"Distance to Node {idx}",
                    color=color, linestyle="-", linewidth=1.5, alpha=0.9)

        # Plot transmission times for each node (vertical lines)
        for time in transmissions_per_node[idx]:
            axs[0].axvline(x=time, color=color, linestyle="--", alpha=0.7)

    # Custom legend for node-specific distances and transmission times
    custom_transmission_lines = [
        lines.Line2D([], [], color=color, linestyle='--', label=f"Node {i} Transmission")
        for i, color in enumerate(node_colors)
    ]
    custom_distance_lines = [
        lines.Line2D([], [], color=color, linestyle='-', label=f"Node {i} Distance")
        for i, color in enumerate(node_colors)
    ]
    axs[0].legend(
        handles=custom_transmission_lines + custom_distance_lines,
        loc="upper center",  # Position the legend outside the plot area
        bbox_to_anchor=(0.5, -0.15),  # Move legend below the plot
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

    # Plot packets received for each node
    for node_idx, color in enumerate(node_colors):
        axs[1].plot(timestamps, packets_received_per_node[node_idx], label=f"Node {node_idx} Packets Received",
                    color=color, linestyle='-', alpha=0.9, linewidth=3)
        # Plot transmission times for each node (vertical lines)
        for time in transmissions_per_node[node_idx]:
            axs[1].axvline(x=time, color=color, linestyle="--", alpha=0.7)

    # Calculate the total Packet Delivery Rate (PDR) for each time step
    pdr = [r / s if s != 0 else 0 for r, s in zip(packets_received, packets_sent)]

    # Create secondary y-axis for PDR
    ax2 = axs[1].twinx()
    ax2.plot(timestamps, pdr, label="Total PDR", color="black", linestyle=':', linewidth=2)

    # Define custom legend for packets received, packets sent, and PDR
    # Custom legend for node-specific distances and transmission times
    custom_lines = [
        lines.Line2D([], [], color="black", linestyle='-', label="Total Packets Received"),
        lines.Line2D([], [], color="black", linestyle='--', label="Total Packets Sent"),
        lines.Line2D([], [], color="black", linestyle=':', label="Total PDR")
    ]
    custom_transmission_lines = [
        lines.Line2D([], [], color=color, linestyle='--', label=f"Node {i} Transmission")
        for i, color in enumerate(node_colors)
    ]
    custom_receive_lines = [
        lines.Line2D([], [], color=color, linestyle='-', label=f"Node {i} Packets Received")
        for i, color in enumerate(node_colors)
    ]
    axs[1].legend(
        handles=custom_lines + custom_transmission_lines + custom_receive_lines,
        loc="upper center",  # Place legend below the second subplot
        bbox_to_anchor=(0.5, -0.15),  # Position the legend below the plot
        borderaxespad=0.0,
        ncol=2  # Two columns for better spacing
    )
    # TODO: Add fairness metric?
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


def create_heatmap(positions, step_times, grid_size_x, grid_size_y):
    """
    Creates a normalized heatmap showing the fraction of total time spent in each grid cell.
    """
    # Initialize the grid
    grid = np.zeros((grid_size_y, grid_size_x))

    # Populate the grid with time spent
    for (x, y), time in zip(positions, step_times):
        grid_x = int(x)
        grid_y = int(y)

        # Ensure the indices are within grid bounds
        if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
            grid[grid_y, grid_x] += 1
        else:
            print(f"err, at {grid_x, grid_y =}")
    grid += 0.1  # make sure each cell has a value greater than 0, so we can use log
    grid = np.log(grid)
    grid += abs(np.min(grid))  # make sure all values are at least 0
    # Normalize the grid by max time
    grid /= np.max(grid)  # scale so values are between 0 and 1

    return grid


def _plot_heatmap(grid):
    """
    Plots the heatmap using Matplotlib.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, origin='lower', cmap='hot')
    plt.colorbar(label='log(Time Spent)')
    plt.title('Mobile Gateway Heatmap')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()

def plot_heatmap(log_file, grid_size_x, grid_size_y):
    gw_positions, timestamps, *_ = load_json_log(log_file)
    heatmap_grid = create_heatmap(gw_positions, timestamps, grid_size_x, grid_size_y)
    _plot_heatmap(heatmap_grid)


if __name__ == '__main__':
    # Parameters
    log_file = "env_log.json"
    grid_size_x, grid_size_y = 300, 300  # Define the grid size (e.g., 150x150)
    plot_heatmap(log_file, grid_size_x, grid_size_y)
