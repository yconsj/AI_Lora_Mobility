import tensorflow as tf
import numpy as np
import subprocess

def tf_export(concrete_func, export_path):
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
        with open(export_path, "wb") as c_file:
            c_file.write(b'#include "policy_net_model.h"\n')
        with open(export_path, "ab") as c_file:
            subprocess.run(command, stdout=c_file, check=True)
        print("Feedforward model successfully converted to C array in " + export_path + "!")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion to C array: {e}")
    except FileNotFoundError:
        print("xxd command not found. Please ensure WSL is installed and xxd is available.")
