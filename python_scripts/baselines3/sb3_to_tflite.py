import os
import subprocess

import torch
import tensorflow as tf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

# Define a PyTorch-to-TensorFlow compatible policy
from baselines3.simple_env import SimpleBaseEnv

# Define a TensorFlow model that matches the PPO structure
from tf_exporter import rewrite_policy_net_header
from utilities import load_config
"""
THIS CODE IS BASED ON THE SIMPLE CONVERTER FOR Stable-basleines3 MODELS TO TfLite FOUND HERE:
https://github.com/chunky/sb3_to_coral

"""

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
            print(f"layer type: {type(layer)}")
            if isinstance(layer, tf.keras.layers.Dense):
                # Create a Dense layer (from the given hidden layer)
                self.dense_layers.append(layer)
                layer.build(input_shape=(None, prev_dim))  # Explicitly build the layer with correct input shape
                prev_dim = layer.units  # Update the previous dimension to match the output size of this Dense layer
            elif isinstance(layer, tf.keras.layers.Activation):
                # Add an Activation layer
                self.dense_layers.append(layer)
            elif isinstance(layer, tf.keras.layers.Softmax):
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
    sb3_layers = sb3_model.policy.mlp_extractor.policy_net  # Feature Extractor policy network
    sb3_layers.append(sb3_model.policy.action_net) # Actor network
    #print(f"{sb3_model.policy.action_net = }")

    prev_dim = input_dim

    for sb3_layer in sb3_layers:
        print(f"{sb3_layer = }")
        if isinstance(sb3_layer, torch.nn.Linear):
            # Create a Dense layer for TensorFlow and append it to the hidden_layers list
            tf_dense_layer = tf.keras.layers.Dense(sb3_layer.out_features, activation=None)
            hidden_layers.append(tf_dense_layer)
        elif isinstance(sb3_layer, torch.nn.Tanh):
            hidden_layers.append(tf.keras.layers.Activation("tanh"))
        else:
            raise ValueError(f"Unknown layer type: {type(sb3_layer)}")

    # construct the softmax output layer
    hidden_layers.append(tf.keras.layers.Activation('softmax')) #(output_dim, activation='softmax'))
    # Initialize TFPolicy with the dynamic hidden layers list
    tf_model = TFPolicy(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)

    # Transfer weights from SB3 model to TensorFlow model
    for i, sb3_layer in enumerate(sb3_layers):
        #print(f"layer {i}: {sb3_layer}")
        if isinstance(sb3_layer, torch.nn.Linear):
            # Transfer weights and biases
            weights = sb3_layer.weight.detach().numpy().T  # Transpose to match TensorFlow layer format
            bias = sb3_layer.bias.detach().numpy()
            print(f"tf_model layer {i}: {tf_model.dense_layers[i] = }")
            tf_model.dense_layers[i].set_weights([weights, bias])

    return tf_model


def tf_to_tflite(tf_model, export_path):
    try:
        # Define a concrete function with an input signature
        concrete_func = tf_model.get_concrete_function()
        try:
            # Convert the model to TFLite format using the concrete function
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
                header_file_name = "policy_net_model.h"
                export_dir = os.path.dirname(export_path)
                header_path = os.path.join(export_dir, header_file_name)
                g_model_length = len(tflite_model)  # Calculate g_model length
                rewrite_policy_net_header(header_path, export_path, g_model_length, episode_num=0)
            except IOError as e:
                print(f"Error saving the TFLite model to file: {e}")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")


if __name__ == '__main__':
    # model = PPO.load("stable-model.zip")
    model = PPO.load("stable-model-best/best_model", print_system_info=True)

    env = make_vec_env(SimpleBaseEnv, n_envs=1, env_kwargs=dict())

    tf_model = sb3_to_tensorflow(model, env)

    config = load_config("config.json")
    gen_model = config['model_path']
    export_model_path = gen_model
    tf_to_tflite(tf_model, export_model_path)
