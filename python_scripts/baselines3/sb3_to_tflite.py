import subprocess

import torch
import tensorflow as tf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env


# Define a PyTorch-to-TensorFlow compatible policy
from baselines3.simple_env import SimpleBaseEnv

# Define a TensorFlow model that matches the PPO structure
class TFPolicy(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(TFPolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.policy_output_layer = tf.keras.layers.Dense(output_dim)

        # Build the model with the provided hidden layers
        self.dense_layers = []
        prev_dim = input_dim
        for layer in hidden_layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Create a Dense layer (from the given hidden layer)
                self.dense_layers.append(layer)
                layer.build(input_shape=(None, prev_dim))  # Explicitly build the layer with correct input shape
                prev_dim = layer.units  # Update the previous dimension to match the output size of this Dense layer
            elif isinstance(layer, tf.keras.layers.Activation):
                # Add an Activation layer
                self.dense_layers.append(layer)
            else:
                raise ValueError(f"Unknown layer type: {type(layer)}")

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.policy_output_layer(x)

    def get_concrete_function(self):
        # Create a concrete function with an input signature
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32)])
        def concrete_function(x):
            return self.call(x)

        return concrete_function.get_concrete_function()


def sb3_to_tensorflow(sb3_model, env):
    input_dim = env.observation_space.shape[0]  # Extract input dimension from SB3
    output_dim = env.action_space.n  # Extract output dimension from SB3

    # Initialize an empty list for the hidden layers
    hidden_layers = []

    # Extract the layers from SB3 model
    sb3_layers = sb3_model.policy.mlp_extractor.policy_net  # Extract policy network
    prev_dim = input_dim

    for sb3_layer in sb3_layers:
        if isinstance(sb3_layer, torch.nn.Linear):
            # Create a Dense layer for TensorFlow and append it to the hidden_layers list
            tf_dense_layer = tf.keras.layers.Dense(sb3_layer.out_features, activation=None)
            hidden_layers.append(tf_dense_layer)

            # Now handle the activation function
            activation_function = torch.nn.Tanh if isinstance(sb3_layer,
                                                              torch.nn.Linear) else None  # Add your conditions for other layers
            if activation_function:
                hidden_layers.append(tf.keras.layers.Activation(activation_function.__name__.lower()))
            prev_dim = sb3_layer.out_features  # Update the previous dimension to match this layer's output size

    # Initialize TFPolicy with the dynamic hidden layers list
    tf_model = TFPolicy(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)

    # Transfer weights from SB3 model to TensorFlow model
    for i, sb3_layer in enumerate(sb3_layers):
        print(f"layer {i}: {sb3_layer}")
        if isinstance(sb3_layer, torch.nn.Linear):
            # Transfer weights and biases
            weights = sb3_layer.weight.detach().numpy().T  # Transpose to match TensorFlow layer format
            bias = sb3_layer.bias.detach().numpy()
            print(f"tf_model layer{i}: {tf_model.dense_layers[i] = }")
            tf_model.dense_layers[i].set_weights([weights, bias])

    return tf_model


def tf_to_tflite(tf_model, output_filename):
    try:
        # Define a concrete function with an input signature
        concrete_func = tf_model.get_concrete_function()

        # Convert the model to TFLite format using the concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=[])
        tflite_model = converter.convert()

        # Save the TFLite model to a binary file
        with open("g_model", "wb") as f:
            f.write(tflite_model)
        print(f"Model successfully converted and saved to {output_filename}")

        command = ["wsl", "xxd", "-i", "g_model"]
        #
        # # Use subprocess to execute the command
        try:
            with open(output_filename, "w") as c_file:
                # Write the header and the episode number
                c_file.write(f'#include "policy_net_model.h"')

            with open(output_filename, "ab") as c_file:
                subprocess.run(command, stdout=c_file, check=True)
            print("Feedforward model successfully converted to C array in " + output_filename + "!")


        except subprocess.CalledProcessError as e:
            print(f"Error during conversion to C array: {e}")
        except FileNotFoundError:
            print("xxd command not found. Please ensure WSL is installed and xxd is available.")

    except Exception as e:
        print(f"Error during TFLite conversion: {e}")

model = PPO.load("stable-model.zip")

env = make_vec_env(SimpleBaseEnv, n_envs=1, env_kwargs=dict())

tf_model = sb3_to_tensorflow(model, env)
tf_to_tflite(tf_model, "tflite_model.cc")