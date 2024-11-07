import json
import os
from enum import Enum


# class syntax
class InputMembers(Enum):
    """ enum value is the index of the member in the struct/tuple"""
    LATEST_PACKET_RSSI = 0
    LATEST_PACKET_SNIR = 1
    LATEST_PACKET_TIMESTAMP = 2
    NUM_RECEIVED_PACKETS = 3
    CURRENT_TIMESTAMP = 4
    COORD_X = 5


def load_config(config_file):
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the config file
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        return json.load(f)


def denormalize_input_state(noramlized_state: tuple[float, float, float, float, float, float], normalization_factors):
    """ example input could be (-0.530494, 0.0148732, 0.43447892897, 0.0694444, 0.49571759259, 0.432667) """
    denormalized_state = tuple(noramlized_state[member.value] * (1.0 / normalization_factors[member.value]) for member in InputMembers)
    if len(denormalized_state) != len(noramlized_state):
        raise Exception("missing normalization factor(s)")
    return denormalized_state


def export_training_info(export_path, current_episode_num, max_episode_num, packet_reward, exploration_reward,
                         random_choice_probability, normalization_factors: dict[int, float]):
    """ normalization_factors must have a floating point value for each member of the input state"""
    latest_packet_rssi_norm_factor = normalization_factors[InputMembers.LATEST_PACKET_RSSI.value]
    latest_packet_snir_norm_factor = normalization_factors[InputMembers.LATEST_PACKET_SNIR.value]
    latest_packet_timestamp_norm_factor = normalization_factors[InputMembers.LATEST_PACKET_TIMESTAMP.value]
    num_received_packets_norm_factor = normalization_factors[InputMembers.NUM_RECEIVED_PACKETS.value]
    current_timestamp_norm_factor = normalization_factors[InputMembers.CURRENT_TIMESTAMP.value]
    coord_x_norm_factor = normalization_factors[InputMembers.COORD_X.value]

    # TODO: Move the normalization info to the header file in tf_export instead,
    #  since normalization isn't dynamic/training dependent.

    content = {
        "current_episode_num": current_episode_num,
        "max_episode_num": max_episode_num,
        "packet_reward": packet_reward,
        "exploration_reward": exploration_reward,
        "random_choice_probability": random_choice_probability,
        "normalization":
            {
                "latest_packet_rssi": latest_packet_rssi_norm_factor,
                "latest_packet_snir": latest_packet_snir_norm_factor,
                "latest_packet_timestamp": latest_packet_timestamp_norm_factor,
                "num_received_packets": num_received_packets_norm_factor,
                "current_timestamp": current_timestamp_norm_factor,
                "coord_x": coord_x_norm_factor
            }
    }

    json_obj = json.dumps(content, indent=4)

    with open(export_path, 'w', encoding='utf-8') as file:
        file.write(json_obj)

    print(f"Written training info to file at {export_path}.")
