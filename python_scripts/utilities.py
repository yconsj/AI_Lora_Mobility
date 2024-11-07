import json
import os


def load_config(config_file):
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the config file
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        return json.load(f)


def export_training_info(export_path, current_episode_num, max_episode_num, packet_reward, exploration_reward,
                         random_choice_probability):
    content = {
        "current_episode_num": current_episode_num,
        "max_episode_num": max_episode_num,
        "packet_reward": packet_reward,
        "exploration_reward": exploration_reward,
        "random_choice_probability": random_choice_probability
    }
    json_obj = json.dumps(content, indent=4)

    with open(export_path, 'w', encoding='utf-8') as file:
        file.write(json_obj)

    print(f"Written training info to file at {export_path}.")
