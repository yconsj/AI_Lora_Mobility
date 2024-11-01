import os

import tensorflow as tf
import numpy as np
import subprocess


def tf_export(concrete_func, export_path, episode_num):
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
    else:
        # Save the converted model to a .tflite file
        try:
            with open("g_model", "wb") as f:
                f.write(tflite_model)
            print("Feedforward model successfully converted to .tflite file!")
        except IOError as e:
            print(f"Error saving the TFLite model to file: {e}")
    # Define the command you want to run to convert the .tflite file to a C array
    command = ["wsl", "xxd", "-i", "g_model"]

    # Use subprocess to execute the command
    try:
        with open(export_path, "w") as c_file:
            # Write the header and the episode number
            c_file.write(f'#include "policy_net_model.h"')

        with open(export_path, "ab") as c_file:
            subprocess.run(command, stdout=c_file, check=True)
        print("Feedforward model successfully converted to C array in " + export_path + "!")

        export_dir = os.path.dirname(export_path)
        header_path = os.path.join(export_dir, "policy_net_model.h")
        g_model_length = len(tflite_model)  # Calculate g_model length
        rewrite_policy_net_header(header_path, g_model_length)


    except subprocess.CalledProcessError as e:
        print(f"Error during conversion to C array: {e}")
    except FileNotFoundError:
        print("xxd command not found. Please ensure WSL is installed and xxd is available.")



def rewrite_policy_net_header(file_path, g_model_length):
    """Rewrite the policy_net_model.h file with updated constants."""
    content = (
        f"#ifndef INET_MOBILITY_RL_MODELFILES_POLICY_NET_MODEL_H_\n"
        f"#define INET_MOBILITY_RL_MODELFILES_POLICY_NET_MODEL_H_\n\n"
        f"extern unsigned char g_model[];\n"
        f"extern unsigned int g_model_len;\n"
        f"#define EPISODE_NUM 0\n"
        f"constexpr int const_g_model_length = {g_model_length};\n\n"
        f"#endif  // INET_MOBILITY_RL_MODELFILES_POLICY_NET_MODEL_H_\n"
    )

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Rewritten {file_path} with updated g_model_length.")

