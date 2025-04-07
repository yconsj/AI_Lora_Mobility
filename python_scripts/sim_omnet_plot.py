import os

import numpy as np
from matplotlib import pyplot as plt

plot_folder = "omnet_output_plots"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)


def plot_omnet_episode(stats_dict, include_stationary=True, include_static_mobility=True):
    """
    Main function to plot all the stats, controlling whether to include stationary gateway data.
    """
    # Plot the distances from gateway to each node
    plot_distances(stats_dict)

    # Plot packets received and transmission times
    plot_packets_received(stats_dict, include_stationary=include_stationary,
                          include_static_mobility=include_static_mobility)

    # Plot PDR and fairness
    plot_pdr_fairness(stats_dict, include_stationary=include_stationary,
                      include_static_mobility=include_static_mobility)


def plot_distances(stats_dict):
    distances = np.array(stats_dict["distances"])  # [timesteps, nodes]
    number_of_nodes = stats_dict["number_of_nodes"]
    timestamps = np.array(stats_dict["data_timestamps"])
    transmission_times = stats_dict["transmission_times"]  # List of lists

    colors = plt.cm.tab10.colors
    color_cycle = len(colors)

    for node_idx in range(number_of_nodes):
        fig, ax = plt.subplots(figsize=(12, 5))  # Explicit axes

        node_distances = distances[:, node_idx]

        # Plot distances over time
        ax.plot(timestamps, node_distances,
                label=f"Distance to Node {node_idx}",
                color=colors[node_idx % color_cycle])

        # Plot vertical lines at this node’s transmission times only
        tx_times = transmission_times[node_idx]
        for i, time in enumerate(tx_times):
            ax.axvline(x=time, linestyle="--", alpha=0.7, color='black',
                       label="Transmission Time" if i == 0 else None, zorder=2)

        # ax.set_xlim(timestamps.min(), timestamps.max())
        ax.set_ylim(distances[:, node_idx].min(), distances[:, node_idx].max())

        ax.set_xlabel("Time (seconds)", fontsize=14)
        ax.set_ylabel("Distance (meters)", fontsize=14)
        ax.set_title(f"Node {node_idx}", fontsize=18)
        ax.legend(loc="best")
        ax.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"episode_distance_{node_idx}.png"))
        plt.close(fig)


def plot_packets_received(stats_dict, include_stationary=True, include_static_mobility=True):
    """
    Plot packets received by the mobile gateway, total packets sent, and cumulative packets sent by each node.
    Optionally include data for stationary gateways.
    """
    packets_sent_per_node = stats_dict["packets_sent_per_node"]
    total_packets_sent = stats_dict["packets_sent"]
    packets_received_mobile = stats_dict["packets_received_mobile"]
    packets_received_stationary = stats_dict["packets_received_stationary"]
    packets_received_static_mobility = stats_dict["packets_received_static_mobility"]

    timestamps = stats_dict["data_timestamps"]
    transmission_times = stats_dict["transmission_times"]

    # Create a new figure for the second plot
    plt.figure(figsize=(10, 6))

    plt.plot(timestamps[:len(total_packets_sent)], total_packets_sent,
             label="Packets sent",
             color="black", linestyle="-", alpha=0.7)
    # Plot packets received by the mobile gateway (over all nodes)
    plt.plot(timestamps[:len(packets_received_mobile)], packets_received_mobile,
             label="RL Mobile GW",
             color="C1", linestyle=":")
    # Plot packets received by each stationary node if `include_stationary` is True
    if include_stationary:
        plt.plot(timestamps[:len(packets_received_stationary)], packets_received_stationary,
                 label=f"Stationary GW",
                 color=f"C2", linestyle=":")
    if include_static_mobility:
        plt.plot(timestamps[:len(packets_received_static_mobility)], packets_received_static_mobility,
                 label=f"Static Mobility GW Packets Received",
                 color=f"C3", linestyle=":")

    do_plot_packets_sent = False
    if do_plot_packets_sent:
        # Overlay cumulative packets sent by all nodes
        for node_idx, node_times in enumerate(transmission_times):
            # Add extra time stamp at t=0 where number of packets transmitted is 0,
            cumulative_packets = list(range(len(node_times)))  # 0, len(node_times) + 1
            plt.step(node_times, cumulative_packets, linestyle="--", alpha=0.7, where='post',
                     label=f"Node {node_idx} Packets Sent", color=f"C{3 + node_idx}")  # [0] + ..

    # Customize plot
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Packets Sent and Received", fontsize=14)
    plt.title(f"Packets {'Sent and ' if do_plot_packets_sent else ''}Received Over Time", fontsize=16)
    plt.legend(loc="best")
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"episode_packets_received.png"))
    # plt.show()
    plt.close()


def plot_pdr_fairness(stats_dict, include_stationary=True, include_static_mobility=True):
    """
    Plot PDR and fairness for mobile, stationary, and static mobility gateways.
    """
    timestamps = stats_dict["data_timestamps"]
    pdr_mobile = stats_dict["pdr_mobile"]
    fairness_mobile = stats_dict["fairness_mobile"]
    pdr_stationary = stats_dict["pdr_stationary"]
    fairness_stationary = stats_dict["fairness_stationary"]
    pdr_static_mobility = stats_dict["pdr_static_mobility"]
    fairness_static_mobility = stats_dict["fairness_static_mobility"]

    plt.figure(figsize=(10, 6))

    # Plot Mobile Gateway PDR and Fairness
    plt.plot(timestamps[:len(pdr_mobile)], pdr_mobile, label="RL Mobile GW PDR", color="C1", linestyle="--")
    plt.plot(timestamps[:len(fairness_mobile)], fairness_mobile, label="RL Mobile GW Fairness", color="C1",
             linestyle=":")

    # Plot Stationary Gateway PDR and Fairness if enabled
    if include_stationary:
        plt.plot(timestamps[:len(pdr_stationary)], pdr_stationary, label="Stationary GW PDR", color="C2",
                 linestyle="--")
        plt.plot(timestamps[:len(fairness_stationary)], fairness_stationary, label="Stationary GW Fairness",
                 color="C2", linestyle=":")

    # Plot Static Mobility Gateway PDR and Fairness if enabled
    if include_static_mobility:
        plt.plot(timestamps[:len(pdr_static_mobility)], pdr_static_mobility, label="Static Mobility GW PDR",
                 color="C3", linestyle="--")
        plt.plot(timestamps[:len(fairness_static_mobility)], fairness_static_mobility,
                 label="Static Mobility GW Fairness", color="C3", linestyle=":")

    # Customize plot
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("PDR / Fairness", fontsize=14)
    plt.title("Packet Delivery Rate (PDR) and Fairness Over Time", fontsize=16)
    plt.legend(loc="best")
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"episode_pdr_fairness.png"))
    # plt.show()
    plt.close()


def plot_batch_performance(final_pdr_mobile_per_node_list, final_pdr_stationary_per_node_list,
                           final_pdr_static_mobility_per_node_list,
                           final_pdr_mobile_list, final_pdr_stationary_list, final_pdr_static_mobility_list,
                           final_fairness_mobile_list, final_fairness_stationary_list,
                           final_fairness_static_mobility_list,
                           include_stationary=True, include_static_mobility=True):
    assert len(final_pdr_mobile_per_node_list) == len(final_pdr_stationary_per_node_list) == len(
        final_pdr_static_mobility_per_node_list), \
        "Lengths of PDR lists for mobile, stationary, and static mobility must be equal."
    assert len(final_fairness_mobile_list) == len(final_fairness_stationary_list) == len(
        final_fairness_static_mobility_list), \
        "Lengths of fairness lists for mobile, stationary, and static mobility must be equal."

    # Scale values to percentage

    final_pdr_mobile_list = [x * 100 for x in final_pdr_mobile_list]
    final_pdr_stationary_list = [x * 100 for x in final_pdr_stationary_list]
    final_pdr_static_mobility_list = [x * 100 for x in final_pdr_static_mobility_list]

    # final_fairness_mobile_list = [x * 100 for x in final_fairness_mobile_list]
    # final_fairness_stationary_list = [x * 100 for x in final_fairness_stationary_list]
    # final_fairness_static_mobility_list = [x * 100 for x in final_fairness_static_mobility_list]

    # Plot PDR Box Plot
    fig_pdr, ax_pdr = plt.subplots(figsize=(6, 4))
    box_data_pdr = [final_pdr_mobile_list]
    tick_labels_pdr = ["PDR (RL Mobile)"]

    if include_stationary:
        box_data_pdr.append(final_pdr_stationary_list)
        tick_labels_pdr.append("PDR (Stationary)")

    if include_static_mobility:
        box_data_pdr.append(final_pdr_static_mobility_list)
        tick_labels_pdr.append("PDR (Static Mobility)")

    ax_pdr.boxplot(box_data_pdr,
                   patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="blue"),
                   medianprops=dict(color="black"),
                   whiskerprops=dict(color="blue"),
                   capprops=dict(color="blue"),
                   flierprops=dict(marker="o", color="red", alpha=0.6))

    ax_pdr.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f} %'))
    ax_pdr.set_xticklabels(tick_labels_pdr)
    ax_pdr.set_ylim(0, 105)  # Slightly above 100
    ax_pdr.set_yticks(np.arange(0, 101, 20))  # Ensure last y-tick is exactly 100
    ax_pdr.set_title("Box Plot: PDR Statistics")
    ax_pdr.set_ylabel("PDR (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"batch_pdr_box_plot.png"))
    # plt.show()
    plt.close()

    # Plot Fairness Box Plot
    fig_fairness, ax_fairness = plt.subplots(figsize=(6, 4))
    box_data_fairness = [final_fairness_mobile_list]
    tick_labels_fairness = ["Fairness (RL Mobile)"]

    if include_stationary:
        box_data_fairness.append(final_fairness_stationary_list)
        tick_labels_fairness.append("Fairness (Stationary)")

    if include_static_mobility:
        box_data_fairness.append(final_fairness_static_mobility_list)
        tick_labels_fairness.append("Fairness (Static Mobility)")

    ax_fairness.boxplot(box_data_fairness,
                        patch_artist=True,
                        boxprops=dict(facecolor="lightcoral", color="red"),
                        medianprops=dict(color="black"),
                        whiskerprops=dict(color="red"),
                        capprops=dict(color="red"),
                        flierprops=dict(marker="o", color="darkred", alpha=0.6))

    ax_fairness.set_xticklabels(tick_labels_fairness)
    ax_fairness.set_ylim(0, 1.05)  # Fairness remains in [0,1]
    ax_fairness.set_yticks(np.arange(0, 1.01, 0.20))  # Ensure last y-tick is exactly 100
    ax_fairness.set_title("Box Plot: Fairness Statistics")
    ax_fairness.set_ylabel("Fairness", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"batch_fairness_box_plot.png"))
    # plt.show()
    plt.close()

    # Bar plot for PDR per node
    # TODO: Consider changing from standard-deviation 'error bars' to 'min/max values in batch' bars

    fig3, ax_pdr_nodes = plt.subplots(figsize=(16, 7))
    axis_right_x_pos = 1.02  # ax_pdr_nodes.get_xlim()[1] + 0.2  # Get the maximum x value in the plot

    num_nodes = len(final_pdr_mobile_per_node_list[0])
    pdr_mobile_per_node = np.mean(final_pdr_mobile_per_node_list, axis=0) * 100
    pdr_std_mobile = np.std(final_pdr_mobile_per_node_list, axis=0) * 100
    y_lim = 105

    x_indices = np.arange(num_nodes)
    ax_pdr_nodes.set_xticks(x_indices)
    ax_pdr_nodes.set_xticklabels([f"Node\n{i}" for i in range(num_nodes)])
    ax_pdr_nodes.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f} %'))
    ax_pdr_nodes.set_yticks(np.arange(0, 101, 20))

    bars_per_node = 1 + int(include_stationary) + int(include_static_mobility)
    bar_width = 0.975 / bars_per_node
    bars_mobile_x_pos = x_indices - (bar_width * (bars_per_node - 1)) / bars_per_node
    bars_stationary_x_pos = bars_mobile_x_pos + bar_width * int(include_stationary)
    bars_static_mobility_x_pos = bars_stationary_x_pos + bar_width * int(include_static_mobility)

    bars_mobile = ax_pdr_nodes.bar(bars_mobile_x_pos, pdr_mobile_per_node, yerr=pdr_std_mobile, capsize=5,
                                   color='steelblue', alpha=0.7, width=bar_width, label="RL Mobile")
    for bar, pdr in zip(bars_mobile, pdr_mobile_per_node):  # Annotate bars
        ax_pdr_nodes.text(bar.get_x() + bar.get_width() / 2, y_lim * 1.06,
                          f"{pdr:.1f}%", ha="center", va="bottom",
                          fontsize=10, fontweight="bold")

    if include_stationary:
        pdr_stationary_per_node = np.mean(final_pdr_stationary_per_node_list, axis=0) * 100
        pdr_std_stationary = np.std(final_pdr_stationary_per_node_list, axis=0) * 100

        bars_stationary = ax_pdr_nodes.bar(bars_stationary_x_pos, pdr_stationary_per_node, yerr=pdr_std_stationary,
                                           capsize=5,
                                           color='lightgreen', alpha=0.7, width=bar_width, label="Stationary")
        for bar, pdr in zip(bars_stationary, pdr_stationary_per_node):
            ax_pdr_nodes.text(bar.get_x() + bar.get_width() / 2, y_lim * 1.03,
                              f"{pdr:.1f}%", ha="center", va="bottom",
                              fontsize=10, fontweight="bold")

    if include_static_mobility:
        pdr_static_mobility_per_node = np.mean(final_pdr_static_mobility_per_node_list, axis=0) * 100
        pdr_std_static_mobility = np.std(final_pdr_static_mobility_per_node_list, axis=0) * 100

        bars_static_mobility = ax_pdr_nodes.bar(bars_static_mobility_x_pos, pdr_static_mobility_per_node,
                                                yerr=pdr_std_static_mobility,
                                                capsize=5, color='tomato', alpha=0.7, width=bar_width,
                                                label="Static Mobility")
        for bar, pdr in zip(bars_static_mobility, pdr_static_mobility_per_node):
            ax_pdr_nodes.text(bar.get_x() + bar.get_width() / 2, y_lim * 1.0,
                              f"{pdr:.1f}%", ha="center", va="bottom",
                              fontsize=10, fontweight="bold")

    ax_pdr_nodes.set_xlabel("Nodes")
    ax_pdr_nodes.set_ylabel("PDR (%)", fontsize=12)
    fig3.suptitle(f"PDR for Each Node (RL Mobile"
                  f"{' vs Stationary' if include_stationary else ''}"
                  f"{' vs Static Mobility' if include_static_mobility else ''})",
                  fontsize=14, fontweight="bold", x=0.5, y=0.97)
    ax_pdr_nodes.set_ylim(0, y_lim)
    ax_pdr_nodes.grid(axis="y", linestyle="--", alpha=0.6)

    # Display Overall PDR & Jain's Fairness
    overall_pdr_mobile = np.mean(final_pdr_mobile_list)
    overall_fairness_mobile = np.mean(final_fairness_mobile_list)

    ax_pdr_nodes.text(axis_right_x_pos, .70, f"Fairness (RL Mobile) = {overall_fairness_mobile:.2f}",
                      color="blue", fontsize=12, fontweight="bold", transform=ax_pdr_nodes.transAxes, ha="left")
    ax_pdr_nodes.text(axis_right_x_pos, .60, f"Overall PDR (RL Mobile) = {overall_pdr_mobile:.2f}%",
                      color="blue", fontsize=12, fontweight="bold", transform=ax_pdr_nodes.transAxes, ha="left")

    if include_stationary:
        overall_pdr_stationary = np.mean(final_pdr_stationary_list)
        overall_fairness_stationary = np.mean(final_fairness_stationary_list)

        ax_pdr_nodes.text(axis_right_x_pos, .50, f"Fairness (Stationary) = {overall_fairness_stationary:.2f}",
                          color="green", fontsize=12, fontweight="bold", transform=ax_pdr_nodes.transAxes, ha="left")
        ax_pdr_nodes.text(axis_right_x_pos, .40, f"Overall PDR (Stationary) = {overall_pdr_stationary:.2f}%",
                          color="green", fontsize=12, fontweight="bold", transform=ax_pdr_nodes.transAxes, ha="left")

    if include_static_mobility:
        overall_pdr_static_mobility = np.mean(final_pdr_static_mobility_list)
        overall_fairness_static_mobility = np.mean(final_fairness_static_mobility_list)

        ax_pdr_nodes.text(axis_right_x_pos, .30, f"Fairness (Static Mobility) = {overall_fairness_static_mobility:.2f}",
                          color="red", fontsize=12, fontweight="bold", transform=ax_pdr_nodes.transAxes, ha="left")
        ax_pdr_nodes.text(axis_right_x_pos, .20, f"Overall PDR (Static Mobility) = {overall_pdr_static_mobility:.2f}%",
                          color="red", fontsize=12, fontweight="bold", transform=ax_pdr_nodes.transAxes, ha="left")

    ax_pdr_nodes.legend()
    plt.tight_layout(pad=1.01)
    plt.subplots_adjust(top=0.85, right=0.75, left=0.055)
    plt.savefig(os.path.join(plot_folder, f"batch_pdr_bar_plot.png"))
    # plt.show()
    plt.close()

    # Plot PDR over episodes
    fig_pdr_episodes, ax_pdr_episodes = plt.subplots(figsize=(15, 7))
    plt.title("PDR per episode")
    ax_pdr_nodes.set_xlabel("Episode")
    ax_pdr_nodes.set_ylabel("PDR (%)", fontsize=12)

    # Plot data
    ax_pdr_episodes.plot(range(len(final_pdr_mobile_list)), final_pdr_mobile_list,
                         label="RL Mobile GW PDR", color="C1", linestyle="--")

    if include_stationary:
        ax_pdr_episodes.plot(range(len(final_pdr_stationary_list)), final_pdr_stationary_list,
                             label="Stationary GW PDR", color="C2", linestyle="--")

    if include_static_mobility:
        ax_pdr_episodes.plot(range(len(final_pdr_static_mobility_list)), final_pdr_static_mobility_list,
                             label="Static Mobility GW PDR", color="C3", linestyle="--")

    # Add legend after all plots
    ax_pdr_episodes.legend()

    plt.tight_layout(pad=1.01)
    plt.savefig(os.path.join(plot_folder, f"batch_pdr_episodes_plot.png"))
    # plt.show()
    plt.close()
