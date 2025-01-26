import os

import torch
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Define a PyTorch-to-TensorFlow compatible policy
from baselines3.basecase.simple_env import SimpleBaseEnv

# Define a TensorFlow model that matches the PPO structure
from baselines3.twod_env import TwoDEnv
from tf_exporter import rewrite_policy_net_header
from utilities import load_config, export_training_info, InputMembers

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
        self.policy_output_layer = tf.keras.layers.Softmax()  # tf.keras.layers.Dense(output_dim)


    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:

            x = layer(x)
        return self.policy_output_layer(x)

    def get_concrete_function(self):
        # Create a concrete function with an input signature
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32)])
        def concrete_function(x):
            return self.call(x)

        return concrete_function.get_concrete_function()


def extract_torch_layers(module, input_dim):
    layers = []
    prev_dim = input_dim
    for sb3_layer in module:
        print(f"{sb3_layer = }")
        if isinstance(sb3_layer, (torch.nn.Linear, torch.nn.modules.linear.Linear)):
            # Create a Dense layer for TensorFlow and append it to the hidden_layers list
            tf_dense_layer = tf.keras.layers.Dense(sb3_layer.out_features, activation=None)
            tf_dense_layer.build(input_shape=(None, prev_dim))  # Explicitly build the layer with correct input shape

            # Transfer weights and biases from SB3 model to TensorFlow model
            weights = sb3_layer.weight.detach().cpu().numpy().T  # Transpose to match TensorFlow layer format
            bias = sb3_layer.bias.detach().cpu().numpy()
            # print(f"{weights = }\n{bias = }")
            tf_dense_layer.set_weights([weights, bias])

            prev_dim = tf_dense_layer.units  # Update the previous dimension to match the output size of this Dense layer
            layers.append(tf_dense_layer)
        elif isinstance(sb3_layer, torch.nn.Tanh):
            layers.append(tf.keras.layers.Activation("tanh"))
        elif isinstance(sb3_layer, torch.nn.ReLU):
            layers.append(tf.keras.layers.Activation("relu"))
        elif isinstance(sb3_layer, tf.keras.layers.Softmax):
            # self.dense_layers.append(layer)
            pass
        elif isinstance(sb3_layer, torch.nn.Sequential) or \
                isinstance(sb3_layer, torch.nn.ModuleList) or \
                isinstance(sb3_layer, list):
            layers.extend(extract_torch_layers(sb3_layer, prev_dim))
        else:
            raise ValueError(f"Unhandled layer type: {type(sb3_layer)}")
    return layers

def sb3_to_tensorflow(sb3_model, env):
    input_dim = env.observation_space.shape[0]  # Extract input dimension from SB3
    output_dim = env.action_space.n  # Extract output dimension from SB3

    # Initialize an empty list for the hidden layers
    hidden_layers = []

    print(f"{input_dim = }\n{output_dim =}\n{sb3_model.policy = } \n {sb3_model.policy_class = }")
    # Extract the layers from SB3 model
    sb3_layers = sb3_model.policy.mlp_extractor.policy_net  # Feature Extractor policy network
    sb3_layers.append(sb3_model.policy.action_net)  # Actor network
    # print(f"{sb3_model.policy.action_net = }")

    hidden_layers.extend(extract_torch_layers(sb3_layers, input_dim))


    # construct the softmax output layer
    # Initialize TFPolicy with the dynamic hidden layers list
    tf_model = TFPolicy(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)

    return tf_model


def tf_to_tflite(tf_model, export_path, training_info_export_path):
    # normalization factors for state data
    sim_time_duration = (60 * 60 * 12.0)
    norm_factors = {InputMembers.LATEST_PACKET_RSSI.value: 1.0 / 255.0,
                    InputMembers.LATEST_PACKET_SNIR.value: 1.0 / 100.0,
                    InputMembers.LATEST_PACKET_TIMESTAMP.value: 1.0 / sim_time_duration,
                    # max received packets: half a day in seconds, 500 seconds between each transmission, 2 nodes
                    InputMembers.NUM_RECEIVED_PACKETS.value: 1.0 / (sim_time_duration / 500.0) * 2.0,
                    InputMembers.CURRENT_TIMESTAMP.value: 1.0 / sim_time_duration,
                    InputMembers.COORD_X.value: 1.0 / 3000.0,
                    }

    try:
        # Define a concrete function with an input signature
        concrete_func = tf_model.get_concrete_function()
        try:
            # Convert the model to TFLite format using the concrete function
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=[])
            converter.optimizations = []
            # [tf.lite.Optimize.DEFAULT] THIS OPTIMIZATION DID NOT WORK
            tflite_model = converter.convert()
            # Save the converted model to the specified .tflite file path

            with open(export_path, "wb") as f:
                f.write(tflite_model)
            print(f"Model successfully converted and saved to {export_path}")
            header_file_name = "policy_net_model.h"
            export_dir = os.path.dirname(export_path)
            header_path = os.path.join(export_dir, header_file_name)
            g_model_length = len(tflite_model)  # Calculate g_model length
            print(f"{g_model_length = }")
            rewrite_policy_net_header(header_path, export_path, g_model_length, episode_num=0)

            export_training_info(training_info_export_path, current_episode_num=1, max_episode_num=1,
                                 packet_reward=0,
                                 exploration_reward=0, random_choice_probability=0, normalization_factors=norm_factors)
        except Exception as e:
            print(f"Error during TFLite conversion: {e}")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")


def sb3_to_tflite_pipeline(relative_model_path):
    model = PPO.load(relative_model_path, print_system_info=True)
    env = make_vec_env(TwoDEnv, n_envs=1, env_kwargs=dict())

    tf_model = sb3_to_tensorflow(model, env)

    config = load_config("config.json")
    gen_model = config['model_path']
    export_model_path = gen_model
    training_info_export_path = config["training_info_path"]
    tf_to_tflite(tf_model, export_model_path, training_info_export_path)


if __name__ == '__main__':
    sb3_to_tflite_pipeline("stable-model-2d-best/best_model")
