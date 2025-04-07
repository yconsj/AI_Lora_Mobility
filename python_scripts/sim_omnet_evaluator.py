import json
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

from sim_omnet_plot import plot_folder, plot_omnet_episode, plot_batch_performance
from sim_runner import OmnetEnv
import pandas as pd

# Get the absolute path of the parent directory (python_scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "baselines3"))
# Add "baselines3" directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from baselines3.utilities import load_config, jains_fairness_index
from baselines3.sb3_to_tflite import sb3_to_tflite_pipeline


def read_log(batch: int, log_path: str):
    """
    Reads data for a specific run number from a shared CSV log file using metadata from a JSON file.
    Also returns per-node transmission times from JSON.
    """
    json_file = log_path + ".json"
    with open(json_file, 'r') as file:
        metadata = json.load(file)

    run_key = str(batch)
    if run_key not in metadata:
        raise KeyError(f"Run number {run_key} not found in metadata.")

    # Extract necessary info
    number_of_nodes = metadata[run_key]["static"]["number_of_nodes"]
    csv_filename = metadata[run_key]["file_reference"]
    transmission_times = metadata[run_key].get("transmission_times", {})

    # Convert JSON keys to ordered list of per-node tx times
    transmission_times_list = []
    for i in range(number_of_nodes):
        node_key = str(i)
        transmission_times_list.append(transmission_times.get(node_key, []))

    # Construct full path to CSV
    json_dir = os.path.dirname(json_file)
    csv_path = os.path.join(json_dir, csv_filename)

    # Read data and filter for current run
    df = pd.read_csv(csv_path)
    run_df = df[df["run"] == batch].reset_index(drop=True)

    return run_df, number_of_nodes, transmission_times_list


def max_smooth(arr, axis=None):
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    if arr.ndim == 2 and axis is not None:
        return np.maximum(arr, np.roll(arr, shift=-1, axis=axis))
    elif arr.ndim == 1:
        return np.maximum(arr[:-1], arr[1:])
    return arr  # Return unchanged if it's not 1D or 2D with axis specified


def extract_episode_stats(df: pd.DataFrame, number_of_nodes: int, transmission_times: list[list[float]]):
    timestamps = df["t"].to_numpy()
    actions = df["action"].to_numpy()
    gw_x = df["gw_x"].to_numpy()
    gw_y = df["gw_y"].to_numpy()

    distances = df[[f"node{i}_dist" for i in range(number_of_nodes)]].to_numpy()
    tx_per_node = df[[f"node{i}_tx" for i in range(number_of_nodes)]].to_numpy()
    rx_mobile = df[[f"node{i}_rx_mobile" for i in range(number_of_nodes)]].to_numpy()
    rx_stationary = df[[f"node{i}_rx_stationary" for i in range(number_of_nodes)]].to_numpy()
    rx_staticmob = df[[f"node{i}_rx_staticmob" for i in range(number_of_nodes)]].to_numpy()

    total_packets_sent = tx_per_node.sum(axis=1)
    total_rx_mobile = rx_mobile.sum(axis=1)
    total_rx_stationary = rx_stationary.sum(axis=1)
    total_rx_staticmob = rx_staticmob.sum(axis=1)

    # PDR per node
    pdr_mobile_per_node = np.divide(rx_mobile, tx_per_node, where=tx_per_node > 0,
                                    out=np.zeros_like(rx_mobile, dtype=float))
    pdr_stationary_per_node = np.divide(rx_stationary, tx_per_node, where=tx_per_node > 0,
                                        out=np.zeros_like(rx_stationary, dtype=float))
    pdr_staticmob_per_node = np.divide(rx_staticmob, tx_per_node, where=tx_per_node > 0,
                                       out=np.zeros_like(rx_staticmob, dtype=float))

    # Total PDR
    pdr_mobile = np.divide(total_rx_mobile, total_packets_sent, where=total_packets_sent > 0,
                           out=np.zeros_like(total_rx_mobile, dtype=float))
    pdr_stationary = np.divide(total_rx_stationary, total_packets_sent, where=total_packets_sent > 0,
                               out=np.zeros_like(total_rx_stationary, dtype=float))
    pdr_staticmob = np.divide(total_rx_staticmob, total_packets_sent, where=total_packets_sent > 0,
                              out=np.zeros_like(total_rx_staticmob, dtype=float))

    # Fairness
    fairness_mobile = [jains_fairness_index(rx, tx) for rx, tx in zip(rx_mobile, tx_per_node)]
    fairness_stationary = [jains_fairness_index(rx, tx) for rx, tx in zip(rx_stationary, tx_per_node)]
    fairness_staticmob = [jains_fairness_index(rx, tx) for rx, tx in zip(rx_staticmob, tx_per_node)]

    # Smooth results
    pdr_mobile = max_smooth(pdr_mobile)
    pdr_mobile_per_node = max_smooth(pdr_mobile_per_node, axis=0)
    pdr_stationary = max_smooth(pdr_stationary)
    pdr_stationary_per_node = max_smooth(pdr_stationary_per_node, axis=0)
    fairness_stationary = max_smooth(fairness_stationary)
    fairness_mobile = max_smooth(fairness_mobile)
    pdr_staticmob = max_smooth(pdr_staticmob)
    pdr_staticmob_per_node = max_smooth(pdr_staticmob_per_node, axis=0)
    fairness_staticmob = max_smooth(fairness_staticmob)

    return {
        "number_of_nodes": number_of_nodes,
        "distances": distances.tolist(),
        "transmission_times": transmission_times,
        "packets_sent": total_packets_sent.tolist(),
        "packets_sent_per_node": tx_per_node.tolist(),

        "packets_received_mobile": total_rx_mobile.tolist(),
        "packets_received_mobile_per_node": rx_mobile.tolist(),
        "pdr_mobile": pdr_mobile.tolist(),
        "pdr_mobile_per_node": pdr_mobile_per_node.tolist(),
        "fairness_mobile": fairness_mobile,

        "packets_received_stationary": total_rx_stationary.tolist(),
        "packets_received_per_node_stationary": rx_stationary.tolist(),
        "pdr_stationary": pdr_stationary.tolist(),
        "pdr_stationary_per_node": pdr_stationary_per_node.tolist(),
        "fairness_stationary": fairness_stationary,

        "packets_received_static_mobility": total_rx_staticmob.tolist(),
        "packets_received_per_node_static_mobility": rx_staticmob.tolist(),
        "pdr_static_mobility": pdr_staticmob.tolist(),
        "pdr_static_mobility_per_node": pdr_staticmob_per_node.tolist(),
        "fairness_static_mobility": fairness_staticmob,

        "data_timestamps": timestamps.tolist()
    }


def load_all_run_data(log_path: str):
    """
    Loads metadata and entire CSV once. Returns a dictionary mapping run numbers to (DataFrame, node_count, transmission_times).
    """
    json_file = log_path + ".json"
    with open(json_file, 'r') as file:
        metadata = json.load(file)

    json_dir = os.path.dirname(json_file)
    csv_file = os.path.join(json_dir, next(iter(metadata.values()))["file_reference"])
    df = pd.read_csv(csv_file)

    run_data = {}
    for run_key, run_meta in metadata.items():
        run_number = int(run_key)
        node_count = run_meta["static"]["number_of_nodes"]
        tx_times_raw = run_meta.get("transmission_times", {})
        tx_times_list = [tx_times_raw.get(str(i), []) for i in range(node_count)]

        df_run = df[df["run"] == run_number].reset_index(drop=True)
        run_data[run_number] = (df_run, node_count, tx_times_list)

    return run_data


# Save boxplot summary stats
def save_boxplot_stats(filename, data_dict):
    rows = []
    for label, values in data_dict.items():
        q1 = np.percentile(values, 25)
        q2 = np.percentile(values, 50)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        mean = np.mean(values)
        std = np.std(values)
        rows.append({
            "Type": label,
            "Mean": mean,
            "StdDev": std,
            "Q1": q1,
            "Median": q2,
            "Q3": q3,
            "IQR": iqr
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(plot_folder, filename), index=False)


# Save barplot node stats
def save_barplot_node_stats(filename, node_pdr_dict):
    data = {}
    for label, pdr_data in node_pdr_dict.items():
        mean = np.mean(pdr_data, axis=0)
        std = np.std(pdr_data, axis=0)
        for i in range(mean.shape[0]):
            data.setdefault("Node", []).append(i)
            data.setdefault("Type", []).append(label)
            data.setdefault("Mean_PDR", []).append(mean[i])
            data.setdefault("StdDev_PDR", []).append(std[i])
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(plot_folder, filename), index=False)


def main():
    """
    Main function to run the simulation and plot the results.
    """
    # Initialize OmnetEnv from the existing module
    env = OmnetEnv()

    if False:
        sb3_to_tflite_pipeline("baselines3/stable-model-2d-best/best_model")
    config = load_config("config.json")
    # Get log path from the configuration
    log_path = config['logfile_path']
    if not log_path:
        print("Log file path is not specified in the configuration.")
        return

    include_stationary = False
    include_static_mobility = True
    batch_size = 100
    if True:
        print("Starting simulation...")
        env.run_simulation(ini_config="scenario_5_a", batch_size=batch_size)

    # Data storage for batch results
    final_pdr_mobile_per_node_list = []
    final_pdr_mobile_list = []
    final_fairness_mobile_list = []

    final_pdr_stationary_per_node_list = []
    final_pdr_stationary_list = []
    final_fairness_stationary_list = []

    final_pdr_static_mobility_per_node_list = []
    final_pdr_static_mobility_list = []
    final_fairness_static_mobility_list = []

    print("Reading log data...")
    all_runs = load_all_run_data(log_path)
    for batch_idx in range(batch_size):
        df_run, num_nodes, transmission_times = all_runs[batch_idx]
        data = extract_episode_stats(df_run, num_nodes, transmission_times)

        # Extract final batch values
        final_pdr_mobile_per_node_list.append(data["pdr_mobile_per_node"][-1])
        final_pdr_mobile_list.append(data["pdr_mobile"][-1])
        final_fairness_mobile_list.append(data["fairness_mobile"][-1])

        final_pdr_stationary_per_node_list.append(data["pdr_stationary_per_node"][-1])
        final_pdr_stationary_list.append(data["pdr_stationary"][-1])
        final_fairness_stationary_list.append(data["fairness_stationary"][-1])

        final_pdr_static_mobility_per_node_list.append(data["pdr_static_mobility_per_node"][-1])
        final_pdr_static_mobility_list.append(data["pdr_static_mobility"][-1])
        final_fairness_static_mobility_list.append(data["fairness_static_mobility"][-1])

        # Plot only for the last batch
        if batch_idx + 1 == batch_size:
            print(f"Plotting episode {batch_idx + 1} results...")
            # print(data)
            plot_omnet_episode(data,
                               include_stationary=include_stationary,
                               include_static_mobility=include_static_mobility)

    # After reading and processing all batches, plot the batch performance (PDR and fairness)
    print("Plotting batch performance...")

    # Write the summary files for stats

    save_boxplot_stats("boxplot_pdr.csv", {
        "RL Mobile": final_pdr_mobile_list,
        "Stationary": final_pdr_stationary_list,
        "Static Mobility": final_pdr_static_mobility_list
    })

    save_boxplot_stats("boxplot_fairness.csv", {
        "RL Mobile": final_fairness_mobile_list,
        "Stationary": final_fairness_stationary_list,
        "Static Mobility": final_fairness_static_mobility_list
    })

    save_barplot_node_stats("barplot_pdr_nodes.csv", {
        "RL Mobile": final_pdr_mobile_per_node_list,
        "Stationary": final_pdr_stationary_per_node_list,
        "Static Mobility": final_pdr_static_mobility_per_node_list
    })

    plot_batch_performance(
        final_pdr_mobile_per_node_list, final_pdr_stationary_per_node_list, final_pdr_static_mobility_per_node_list,
        final_pdr_mobile_list, final_pdr_stationary_list, final_pdr_static_mobility_list,
        final_fairness_mobile_list, final_fairness_stationary_list, final_fairness_static_mobility_list,
        include_stationary=include_stationary, include_static_mobility=include_static_mobility
    )


if __name__ == "__main__":
    main()
