import json
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np

def plot_mobile_gateway_with_nodes_advanced(log_file):
    """
    Plots:
    1. Distance from the Gateway to each node over time with transmission times marked.
    2. Number of packets received vs packets sent for each node, and total packets received.
    """
    # Load log data
    with open(log_file, 'r') as file:
        data = json.load(file)
    dynamic_data = data["dynamic"]
    node_distances = [entry['node_distances'] for entry in dynamic_data]
    timestamps = [entry['step_time'] for entry in dynamic_data]
    packets_received = [entry['packets_received'] for entry in dynamic_data]
    packets_sent = [entry['packets_sent'] for entry in dynamic_data]
    transmission_occureds = [entry['transmissions_per_node'] for entry in dynamic_data]
    packets_received_per_node = [[] for _ in range(len(dynamic_data[0]['packets_received_per_node']))]
    packets_sent_per_node = [[] for _ in range(len(dynamic_data[0]['packets_sent_per_node']))]
    for entry in dynamic_data:
        for i, received in enumerate(entry['packets_received_per_node']):
            packets_received_per_node[i].append(received)
        for i, sent in enumerate(entry['packets_sent_per_node']):
            packets_sent_per_node[i].append(sent)

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
    for i in range(len(transmission_occureds)):
        time = timestamps[i]
        for node_idx in range(len(transmission_occureds[0])):
            if transmission_occureds[i][node_idx]:
                color = node_colors[node_idx]
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

    # Define custom legend for packets received, packets sent, and PDR
    # Custom legend for node-specific distances and transmission times
    custom_lines = [
        lines.Line2D([], [], color="black", linestyle='-', label="Total Packets Received"),
        lines.Line2D([], [], color="black", linestyle='--', label="Total Packets Sent")
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
    axs[1].set_xlabel("Time (steps)")
    axs[1].set_ylabel("Packets")
    axs[1].set_title("Packets Received vs Packets Sent Per Node & PDR")
    axs[1].grid()

    # Adjust layout and make room for legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Increased space at the bottom for the legends by default
    plt.show()


def create_heatmap(gw_positions, grid_size_x, grid_size_y):
    """
    Creates a normalized heatmap showing the fraction of total time spent in each grid cell.
    """
    # Initialize the grid
    grid = np.zeros((grid_size_y, grid_size_x))

    # Populate the grid with time spent
    for (x, y) in gw_positions:
        grid_x = round(x)
        grid_y = round(y)

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


def _plot_heatmap(grid, node_positions):
    """
    Plots the heatmap using Matplotlib.
    """
    plt.figure(figsize=(10, 8))
    # invert y-axis to match with render screen, where y ascends in the downwards direction:
    plt.imshow(grid, origin='upper', cmap='hot')
    # Overlay static node positions
    plt.colorbar(label='log(Time Spent)')
    for x, y in node_positions:
        plt.scatter(x, y, color='blue', label='Node', edgecolor='white', s=100, zorder=5)

    plt.title('Mobile Gateway Heatmap')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()


def plot_heatmap(log_file):
    with open(log_file, 'r') as file:
        data = json.load(file)

    # Extract data
    static_data = data["static"]
    node_positions_x = static_data["node_positions_x"]
    node_positions_y = static_data["node_positions_y"]
    node_positions = zip(node_positions_x, node_positions_y)
    max_distance_x = static_data["max_distance_x"] + 1
    max_distance_y = static_data["max_distance_y"] + 1

    dynamic_data = data["dynamic"]
    gw_position_x = [entry['gw_pos_x'] for entry in dynamic_data]
    gw_position_y = [entry['gw_pos_y'] for entry in dynamic_data]
    gw_positions = zip(gw_position_x, gw_position_y)

    heatmap_grid = create_heatmap(gw_positions, max_distance_x, max_distance_y)
    _plot_heatmap(heatmap_grid, node_positions)


def plot_batch_episode_performance(all_pdr, all_fairness):
    # Plot the performance of the current batch of episodes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    n_episodes = min(len(all_pdr), len(all_fairness))

    # Plot PDR and fairness
    ax1.plot(range(1, n_episodes + 1), all_pdr, label="PDR", color="blue", marker="o")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("PDR", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.set_ylim(0, 1)  # Set y-axis for PDR from 0 to 1

    ax2 = ax1.twinx()  # Create another y-axis for fairness
    ax2.plot(range(1, n_episodes + 1), all_fairness, label="Fairness", color="red", marker="x")
    ax2.set_ylabel("Fairness", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.set_ylim(0, 1)  # Set y-axis for Fairness from 0 to 1

    fig.tight_layout()  # Ensure the layout doesn't overlap
    plt.title("Performance: PDR and Fairness over Episodes")
    plt.show()


if __name__ == '__main__':
    # Parameters
    log_file = "env_log.json"
    plot_heatmap(log_file)
