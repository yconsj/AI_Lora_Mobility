import tensorflow as tf
import numpy as np
import subprocess

# Define a simple feedforward neural network model
class SimpleFFN(tf.Module):
    def __init__(self):
        super(SimpleFFN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation='relu', input_shape=(5,))
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 5], dtype=tf.float32)])
    def __call__(self, x):
        x = self.dense1(x)
        return self.dense2(x)

# Create an instance of the model
simple_ffn = SimpleFFN()

# Create example input data to initialize the model weights
dummy_input = np.random.rand(1, 5).astype(np.float32)  # Shape (1, 5)

# Run dummy input through the model to initialize weights
simple_ffn(dummy_input)

# Get the concrete function for conversion
concrete_func = simple_ffn.__call__.get_concrete_function()

# Convert the model directly from the concrete function
try:
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
except Exception as e:
    print(f"Error during TFLite conversion: {e}")
else:
    # Save the converted model to a .tflite file
    try:
        with open("simple_ffn_model.tflite", "wb") as f:
            f.write(tflite_model)
        print("Feedforward model successfully converted to .tflite file!")
    except IOError as e:
        print(f"Error saving the TFLite model to file: {e}")

# Define the command you want to run to convert the .tflite file to a C array
command = ["wsl", "xxd", "-i", "simple_ffn_model.tflite"]

# Use subprocess to execute the command
try:
    with open("model_data.c", "wb") as c_file:
        subprocess.run(command, stdout=c_file, check=True)
    print("Feedforward model successfully converted to C array in model_data.c!")
except subprocess.CalledProcessError as e:
    print(f"Error during conversion to C array: {e}")
except FileNotFoundError:
    print("xxd command not found. Please ensure WSL is installed and xxd is available.")
