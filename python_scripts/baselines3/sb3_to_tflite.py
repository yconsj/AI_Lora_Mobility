import math
import os
import random

import numpy as np
import torch
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Define a PyTorch-to-TensorFlow compatible policy
from baselines3.basecase.simple_env import SimpleBaseEnv

# Define a TensorFlow model that matches the PPO structure
from baselines3.test2dmodel import sb3_get_action_probabilities
from baselines3.twod_env import TwoDEnv
from baselines3.twodenvrunner import CustomPolicyNetwork
from tf_exporter import rewrite_policy_net_header
from utilities import load_config, export_training_info, InputMembers


"""
THIS CODE IS BASED ON THE SIMPLE CONVERTER FOR Stable-basleines3 MODELS TO TfLite FOUND HERE:
https://github.com/chunky/sb3_to_coral

"""


class TFPolicy(tf.keras.Model):
    """
    TensorFlow policy model mimicking the structure of a Stable-Baselines3 PPO policy.
    """
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.policy_output_layer = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.policy_output_layer(x)

    def get_concrete_function(self):
        """
        Generates a concrete function for TensorFlow model conversion.
        """
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32)])
        def concrete_function(x):
            return self.call(x)

        return concrete_function.get_concrete_function()


def extract_torch_layers(module, input_dim):
    """
    Converts PyTorch layers into equivalent TensorFlow layers.
    """
    layers = []
    prev_dim = input_dim

    for layer in module:
        print(f"{layer = }")
        if isinstance(layer, torch.nn.Linear):
            # Convert Linear layer
            tf_layer = tf.keras.layers.Dense(layer.out_features, activation=None)
            tf_layer.build(input_shape=(None, prev_dim))
            tf_layer.set_weights([layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy()])
            prev_dim = layer.out_features
            layers.append(tf_layer)
        elif isinstance(layer, torch.nn.Tanh):
            layers.append(tf.keras.layers.Activation("tanh"))
        elif isinstance(layer, torch.nn.ReLU):
            layers.append(tf.keras.layers.Activation("relu"))
        elif isinstance(layer, (torch.nn.Sequential, list)):
            # Handle nested structures
            sub_layers, prev_dim = extract_torch_layers(layer, prev_dim)
            layers.extend(sub_layers)
        else:
            raise ValueError(f"Unhandled layer type: {type(layer)}")

    return layers, prev_dim


def sb3_to_tensorflow(sb3_model, env) -> TFPolicy:
    """
    Converts a Stable-Baselines3 model to a TensorFlow-compatible model.
    """
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    print(f"{sb3_model.policy = }")
    # Extract layers from SB3 model
    sb3_layers = [
        sb3_model.policy.mlp_extractor.policy_net,
        sb3_model.policy.action_net
    ]
    tf_layers, _ = extract_torch_layers(sb3_layers, input_dim)

    return TFPolicy(input_dim=input_dim, output_dim=output_dim, hidden_layers=tf_layers ) #


def tf_to_tflite(tf_model, export_path, training_info_export_path):
    """
    Converts a TensorFlow model to TensorFlow Lite format.
    """
    # Normalization factors for state data
    sim_time_duration = 12 * 60 * 60  # 12 hours in seconds
    norm_factors = {
        InputMembers.LATEST_PACKET_RSSI.value: 1.0 / 255.0,
        InputMembers.LATEST_PACKET_SNIR.value: 1.0 / 100.0,
        InputMembers.LATEST_PACKET_TIMESTAMP.value: 1.0 / sim_time_duration,
        InputMembers.NUM_RECEIVED_PACKETS.value: 1.0 / (sim_time_duration / 500.0) * 2.0,
        InputMembers.CURRENT_TIMESTAMP.value: 1.0 / sim_time_duration,
        InputMembers.COORD_X.value: 1.0 / 3000.0,
    }

    try:
        # Generate concrete function and convert to TFLite
        concrete_func = tf_model.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()

        # Save the converted model
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "wb") as f:
            f.write(tflite_model)

        # Export additional information
        g_model_length = len(tflite_model)
        header_path = os.path.join(os.path.dirname(export_path), "policy_net_model.h")
        rewrite_policy_net_header(header_path, export_path, g_model_length, episode_num=0)

        export_training_info(training_info_export_path, current_episode_num=1, max_episode_num=1, packet_reward=0,
                             exploration_reward=0, random_choice_probability=0, normalization_factors=norm_factors)
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")


def test_sb3_tf_model_conversion(sb3_model, tf_model: TFPolicy):
    """
    Validates the conversion from SB3 model to TensorFlow by comparing output probabilities.
    """
    tolerance = {"abs": 1e-6, "rel": 1e-5}
    #print(f"{sb3_model.policy = }\n"
    #      f"{vars(sb3_model.policy) = }")

    sb3_input_dim = sb3_model.observation_space.shape[0]
    tf_input_dim = tf_model.input_dim

    assert sb3_input_dim == tf_input_dim, f"Input dimensions must match. {sb3_input_dim = } || {tf_input_dim = } "

    for _ in range(1000):
        random_input = np.random.random((1, sb3_input_dim)).astype(np.float32)
        sb3_output = sb3_get_action_probabilities(random_input.flatten(), sb3_model).flatten()

        tf_output = tf_model.call(tf.convert_to_tensor(random_input)).numpy().flatten()
        if not np.allclose(sb3_output, tf_output, atol=tolerance["abs"], rtol=tolerance["rel"]):
            print(f"Mismatch detected!\nSB3: {sb3_output}\nTF: {tf_output}")
    print("Completed test")

def sb3_to_tflite_pipeline(relative_model_path):
    model = PPO.load(relative_model_path, print_system_info=True)
    env = make_vec_env(TwoDEnv, n_envs=1, env_kwargs=dict())

    tf_model = sb3_to_tensorflow(model, env)
    test_sb3_tf_model_conversion(sb3_model=model, tf_model=tf_model)

    config = load_config("config.json")
    export_model_path = config['model_path']
    #training_info_export_path = config["training_info_path"]
    #tf_to_tflite(tf_model, export_model_path, training_info_export_path)


if __name__ == '__main__':
    random.seed(0)
    sb3_to_tflite_pipeline("stable-model-2d-best/best_model")

