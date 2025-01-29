import os

import tensorflow as tf
import subprocess
from utilities import load_config

def tf_export(concrete_func, export_path, episode_num):
    header_file_name = "policy_net_model.h"

    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=[])
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")

    else:
        # Save the converted model to the specified .tflite file path
        try:
            with open(export_path, "wb") as f:
                f.write(tflite_model)
            print(f"Model successfully converted and saved to {export_path}")

            # TODO: This doesn't need to happen everytime the model file is written.

            export_dir = os.path.dirname(export_path)
            header_path = os.path.join(export_dir, header_file_name)
            g_model_length = len(tflite_model)  # Calculate g_model length
            rewrite_policy_net_header(header_path, export_path, g_model_length, episode_num)
        except IOError as e:
            print(f"Error saving the TFLite model to file: {e}")
    # Define the command you want to run to convert the .tflite file to a C array
    # command = ["wsl", "xxd", "-i", "g_model"]
    #
    # # Use subprocess to execute the command
    # try:
    #     with open(export_path, "w") as c_file:
    #         # Write the header and the episode number
    #         c_file.write(f'#include "policy_net_model.h"')
    #
    #     with open(export_path, "ab") as c_file:
    #         subprocess.run(command, stdout=c_file, check=True)
    #     print("Feedforward model successfully converted to C array in " + export_path + "!")
    #
    #
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during conversion to C array: {e}")
    # except FileNotFoundError:
    #     print("xxd command not found. Please ensure WSL is installed and xxd is available.")


def rewrite_policy_net_header(header_file_path, model_file_path, g_model_length):
    # TODO: EPISODE_NUM should be written in some other file, since Header is only parsed at compile-time, meaning EPISODE_NUM won't be updated during training
    config = load_config("config.json")

    header_file_basename = os.path.basename(header_file_path)
    log_file_basename = os.path.basename(config["logfile_path"])
    training_info_path = config["training_info_path"]
    ifdefguard = "INET_MOBILITY_RL_MODELFILES_" + header_file_basename.replace('.', '_').replace(' ', '_').upper() + "_"

    """Rewrite the policy_net_model.h file with updated constants."""
    content = (
        f"#ifndef {ifdefguard}\n"
        f"#define {ifdefguard}\n\n"
        f"constexpr int const_g_model_length = {g_model_length};\n\n"
        f'const char* model_file_path = "{model_file_path}"; // Path to your TFLite model file\n\n'
        f'const char* log_file_basename = "{log_file_basename}"; // name for log file\n\n'
        f'const char* training_info_path = "{training_info_path}"; // name for log file\n\n'
        f"#endif  // {ifdefguard}\n"
    )

    with open(header_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Rewritten {header_file_path} with updated g_model_length.")
