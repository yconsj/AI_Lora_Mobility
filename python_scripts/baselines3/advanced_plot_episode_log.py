import json
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
from utilities import jains_fairness_index


def plot_relative_positions(log_file, number_of_nodes=4):
    for node_idx in range(number_of_nodes):
        plot_relative_position(log_file, node_idx)


def plot_relative_position(log_file, node_idx):
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
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))  # Adjusted figure size for better spacing

    # --- First subplot: Distance from Gateway to each node over time ---
    # Plot distances for each node
    axs.plot(timestamps, [dist[node_idx] for dist in node_distances], label=f"Distance to Node {node_idx}",
             color=node_colors[node_idx], linestyle="-", linewidth=1.5, alpha=0.9)

    # Plot transmission times for each node (vertical lines)
    for i in range(len(transmission_occureds)):
        time = timestamps[i]
        if transmission_occureds[i][node_idx]:
            color = node_colors[node_idx]
            axs.axvline(x=time, color="black", linestyle="--", alpha=0.7)

    # Custom legend for node-specific distances and transmission times
    custom_transmission_lines = [
        lines.Line2D([], [], color="black", linestyle='--', label=f"Node {node_idx} Transmission")
    ]
    custom_distance_lines = [
        lines.Line2D([], [], color=node_colors[node_idx], linestyle='-', label=f"Node {node_idx} Distance")
    ]

    axs.legend(
        handles=custom_transmission_lines + custom_distance_lines,
        loc="upper center",  # Position the legend outside the plot area
        bbox_to_anchor=(0.5, -0.2),  # Move legend below the plot
        borderaxespad=0.0,
        ncol=2,  # Two columns for better spacing
    )
    axs.set_xlabel("Time (steps)", fontsize=24)  # Increase x-axis label size
    axs.set_ylabel("Distance (meters)", fontsize=24)  # Increase y-axis label size
    axs.set_title("Distance from Gateway to Nodes Over Time", fontsize=24)  # Increase title size
    axs.grid()

    plt.tight_layout(pad=3)  # Add padding to prevent overlap
    plt.subplots_adjust(bottom=0.15)  # Ensure space at the bottom for legends
    plt.show()


def plot_mobile_gateway_with_nodes_advanced(log_file):
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

    # Initialize figure with 1 subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot total packets received and sent
    ax.plot(timestamps, packets_received, label="Total Packets Received", color="black", linestyle='-', linewidth=2)
    ax.plot(timestamps, packets_sent, label="Total Packets Sent", color="black", linestyle='--', linewidth=2)

    # Plot packets received for each node
    for node_idx, color in enumerate(node_colors):
        ax.plot(timestamps, packets_received_per_node[node_idx], label=f"Node {node_idx} Packets Received",
                color=color, linestyle='-', alpha=0.9, linewidth=2)

    # Create a custom legend
    custom_lines = [
        lines.Line2D([], [], color="black", linestyle='-', label="Total Packets Received"),
        lines.Line2D([], [], color="black", linestyle='--', label="Total Packets Sent")
    ]
    custom_receive_lines = [
        lines.Line2D([], [], color=color, linestyle='-', label=f"Node {i} Packets Received")
        for i, color in enumerate(node_colors)
    ]
    ax.legend(
        handles=custom_lines + custom_receive_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # Position the legend below the plot
        borderaxespad=0.0,
        ncol=2  # Two columns for better spacing
    )

    # Add labels, title, and grid
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Packets")
    ax.set_title("Packets Received vs Packets Sent Per Node")
    ax.grid()

    # Adjust layout and make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Add extra space at the bottom for the legend
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


def plot_batch_episode_performance(all_final_receives: list[list[int]], all_final_sents: list[list[int]]):
    # Plot the performance of the current batch of episodes
    plt.figure(figsize=(8, 6))
    assert len(all_final_receives) == len(
        all_final_sents), f"{len(all_final_receives) =} and {len(all_final_sents) =} must be equal."

    all_pdr = [sum(final_receives) / sum(final_sents)
               for final_receives, final_sents in zip(all_final_receives, all_final_sents)]
    all_fairness = [jains_fairness_index(final_receives, final_sents)
                    for final_receives, final_sents in zip(all_final_receives, all_final_sents)]
    n_episodes = min(len(all_pdr), len(all_fairness))

    # Plot PDR and fairness on the same y-axis
    plt.plot(range(1, n_episodes + 1), all_pdr, label="PDR", color="blue", marker="o", linestyle="-")
    plt.plot(range(1, n_episodes + 1), all_fairness, label="Fairness", color="red", marker="x", linestyle="--")

    plt.xlabel("Episode")
    plt.ylabel("Values (PDR and Fairness)")
    plt.ylim(0, 1)  # Set y-axis for both PDR and fairness from 0 to 1

    # Add title and legend
    plt.title("Performance: PDR and Fairness over Episodes")
    plt.legend(loc="upper right")  # Place the legend in the upper right corner
    plt.grid(True, linestyle="--", alpha=0.5)  # Optional grid for clarity

    plt.tight_layout()  # Ensure the layout doesn't overlap
    plt.show()

    # Plot the box plot for PDR and fairness
    fig2, ax3 = plt.subplots(figsize=(6, 4))
    ax3.boxplot([all_pdr, all_fairness], labels=["PDR", "Fairness"], patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                flierprops=dict(marker="o", color="red", alpha=0.6))

    ax3.set_title("Box Plot: PDR and Fairness Distribution")
    ax3.set_ylabel("Values")
    # ax3.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()  # Show the box plot

    # Third plot: Bar plot for PDR & Fairness per node
    num_nodes = len(all_final_receives[0])
    pdr_per_node = [sum(received) / sum(sent) if sum(sent) > 0 else 0
                    for received, sent in zip(zip(*all_final_receives), zip(*all_final_sents))]
    pdr_std_per_node = [np.std([r / s if s > 0 else 0 for r, s in zip(received, sent)])
                        for received, sent in zip(zip(*all_final_receives), zip(*all_final_sents))]
    overall_pdr = np.mean(all_pdr)
    overall_fairness = np.mean(all_fairness)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bar_width = 0.4
    x_indices = np.arange(num_nodes)

    bars_pdr = ax3.bar(x_indices - bar_width / 2, np.array(pdr_per_node) * 100,
                       yerr=np.array(pdr_std_per_node) * 100, capsize=5,
                       color='steelblue', alpha=0.7, width=bar_width, label="PDR")

    # Annotate bars
    for bar, pdr in zip(bars_pdr, pdr_per_node):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{pdr * 100:.1f}%",
                 ha="center", fontsize=10, fontweight="bold")

    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f"Node {i + 1}" for i in range(num_nodes)])

    ax3.set_xlabel("Nodes")
    ax3.set_ylabel("Percentage (%)")
    ax3.set_title("PDR for Each Node")
    ax3.set_ylim(0, 100)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    # Display Overall PDR & Jain's Fairness
    ax3.text(num_nodes - 0.5, 80, f"Jain's Fairness Index = {overall_fairness:.2f}",
             color="purple", fontsize=12, fontweight="bold")
    ax3.text(num_nodes - 0.5, 70, f"Overall PDR = {overall_pdr * 100:.2f}%",
             fontsize=12, fontweight="bold")

    ax3.legend()
    plt.tight_layout()
    plt.show()
