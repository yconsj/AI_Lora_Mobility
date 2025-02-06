import json
import os
import numpy as np


def load_config(config_filename):
    # Get the absolute path of the current script (utilities.py inside baselines3)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if config.json exists in the current script's parent directory (expected location)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    config_path = os.path.join(project_root, config_filename)

    # Load JSON
    with open(config_path, 'r') as f:
        return json.load(f)


def _jains_fairness_index(delivery_rates: np.ndarray) -> float:
    n = len(delivery_rates)
    temp = np.sum(delivery_rates ** 2)  # Sum of squares
    if temp == 0:
        return 0.0
    return (np.sum(delivery_rates) ** 2) / (n * temp)


def jains_fairness_index(received_per_node: list[int], sent_per_node: list[int]) -> float:
    if len(received_per_node) != len(sent_per_node):
        raise ValueError("Error in 'jains_fairness_index'! "
                         "Lengths of 'received_per_node' and 'sent_per_node' must be equal.")

    # Convert the input lists to numpy arrays
    received_per_node = np.array(received_per_node)
    sent_per_node = np.array(sent_per_node)

    # Using numpy to calculate delivery rates element-wise
    delivery_rates = np.divide(received_per_node, sent_per_node, out=np.zeros_like(received_per_node, dtype=float),
                               where=sent_per_node != 0)
    return _jains_fairness_index(delivery_rates)

