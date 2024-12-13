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

    # Extract state format and state data
    state_format = data["gw_data"]["stateformat"]
    state_indices = {key: idx for idx, key in enumerate(state_format)}

    states = data["gw_data"]["states"]  # The input state array
    actions = data["gw_data"]["actions"]  # The choice array
    rewards = data["gw_data"]["rewards"]  # The reward array
    transmission_times = data["transmission_times"]  # The transmission times
    stationary_gateway_reception_times = data[
        "stationary_gateway_reception_times"]  # The reception times for stationary gateways

    return states, actions, rewards, transmission_times, stationary_gateway_reception_times, state_indices


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


def plot_all(input_states, state_indices, transmission_times, stationary_gateway_reception_times):
    """
    Plots:
    1. Gateway X position over time with transmission times marked
    2. Number of packets received by mobile and stationary gateways vs transmission times
    """
    # Access indices from state_indices
    gw_x_index = state_indices["gwPosition.x"]
    packet_count_index = state_indices["numReceivedPackets"]
    timestamp_index = state_indices["timeOfSample"]  # Assuming the timestamp is present in the state

    # Extract relevant data from input states
    gw_x = [state[gw_x_index] for state in input_states]  # Using the dynamically found index
    packet_counts = [state[packet_count_index] for state in input_states]  # Using the dynamically found index
    timestamps = [state[timestamp_index] for state in input_states]  # Using the timestamp for the x-axis

    # Initialize figure with 2 subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # --- First subplot: Gateway X position over time with transmission times ---
    axs[0].plot(timestamps, gw_x, label="Gateway X Position", color="blue")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Position (gw_x)")
    axs[0].set_title("Gateway X Position Over Time")

    # Overlay a vertical line for transmission times
    for time in transmission_times:
        axs[0].axvline(x=time, color="red", linestyle="--", alpha=0.5)

    axs[0].legend(["Gateway X Position", "Transmission Time"], loc="best")
    axs[0].grid()

    # --- Second subplot: Number of packets received versus transmission times ---
    axs[1].plot(timestamps[:len(packet_counts)], packet_counts, label="Mobile GW Packets Received", color="green")

    # Create cumulative packets list for transmitted packets (jagged)
    transmitted_packets = []
    total_transmitted_packets = 0
    for time in transmission_times:
        total_transmitted_packets += 1
        transmitted_packets.append(total_transmitted_packets)

    # Make the transmitted packets jagged by applying the create_jagged_line function
    transmitted_x, transmitted_y = create_jagged_line(transmission_times, transmitted_packets)
    axs[1].plot(transmitted_x, transmitted_y, label="Packets Transmitted", color="black", linestyle='-.', linewidth=2)

    # Create cumulative packets list for stationary gateways (jagged)
    cumulative_stationary_packets = []
    total_stationary_packets = 0
    for time in stationary_gateway_reception_times:
        total_stationary_packets += 1
        cumulative_stationary_packets.append(total_stationary_packets)

    # Make the stationary gateway packets jagged
    stationary_x, stationary_y = create_jagged_line(stationary_gateway_reception_times, cumulative_stationary_packets)

    # Add a small vertical offset to the stationary gateway packet curve
    offset = 0.1  # Adjust this value for more or less offset
    axs[1].plot(stationary_x, [y + offset for y in stationary_y], label="Cumulative Stationary GW Packets", color="orange", linestyle='-', linewidth=2)

    # Overlay a vertical line for transmission times (include in legend)
    for time in transmission_times:
        axs[1].axvline(x=time, color="red", linestyle="--", alpha=0.5)

    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Packets Received / Transmission Time")
    axs[1].set_title("Packets Received vs Transmission Times")
    axs[1].legend(["Mobile GW Packets Received", "Packets sent", "Stationary GW Packets Received", "Transmission Time"], loc="best")
    axs[1].grid()

    # Adjust layout and show
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
        sb3_to_tflite_pipeline("baselines3/stable-model-best/best_model")

    config = load_config("config.json")
    # Get log path from the configuration
    log_path = config['logfile_path']
    if not log_path:
        print("Log file path is not specified in the configuration.")
        return

    # Run the simulation
    print("Starting simulation...")
    env.run_simulation()

    batch = 0  # Specify the batch number if needed
    # Read the log data
    print("Reading log data...")
    input_states, actions, rewards, transmission_times, stationary_gateway_reception_times, state_indices = read_log(
        batch, log_path)

    # Plot all figures (Gateway Position, Packets, Transmission Times)
    print("Plotting results...")
    plot_all(input_states, state_indices, transmission_times, stationary_gateway_reception_times)


if __name__ == "__main__":
    main()
