import json
import os


def load_config(config_file):
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the config file
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        return json.load(f)


def _jains_fairness_index(delivery_rates):
    n = len(delivery_rates)
    temp = sum([(x ** 2) for x in delivery_rates])
    if temp == 0:
        return 0.0
    jains_index = sum(delivery_rates) ** 2 / (n * temp)
    return jains_index


def jains_fairness_index(received_per_node: list[int], sent_per_node: list[int]):
    if len(received_per_node) != len(sent_per_node):
        raise ValueError("Error in 'jains_fairness_index'! "
                         "len('received_per_node') must match length of len('misses_per_node')")
    length = max(len(received_per_node), len(sent_per_node))
    delivery_rates = [
        0 if (sent_per_node[i]) == 0 else received_per_node[i] / (sent_per_node[i])
        for i in range(length)
    ]
    return _jains_fairness_index(delivery_rates)
