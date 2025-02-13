import os
import random

import numpy as np
import torch
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Define a PyTorch-to-TensorFlow compatible policy
# Define a TensorFlow model that matches the PPO structure
from baselines3.test2dmodel import sb3_get_action_probabilities
from baselines3.twod_env import TwoDEnv
from tf_exporter import rewrite_policy_net_header
from baselines3.utilities import load_config

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
        self.hidden_layers = tf.keras.Sequential(hidden_layers)
        self.policy_output_layer = tf.keras.layers.Softmax()

    def call(self, inputs):
        x = self.hidden_layers(inputs)  # Instead of iterating manually
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
    Extracts PyTorch layers and counts layers, nodes, and edges.
    """
    layers = []
    prev_dim = input_dim
    layer_count = 0
    node_count = input_dim  # Start with input nodes
    edge_count = 0

    for layer in module:
        print(f"{layer = }")
        if isinstance(layer, torch.nn.Linear):
            # Convert Linear layer
            tf_layer = tf.keras.layers.Dense(layer.out_features, activation=None)
            tf_layer.build(input_shape=(None, prev_dim))
            tf_layer.set_weights([layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy()])

            # Update counts
            layer_count += 1
            node_count += layer.out_features
            edge_count += prev_dim * layer.out_features  # Each node is connected to all previous ones

            prev_dim = layer.out_features
            layers.append(tf_layer)

        elif isinstance(layer, torch.nn.Tanh) or isinstance(layer, torch.nn.ReLU):
            layers.append(tf.keras.layers.Activation("tanh" if isinstance(layer, torch.nn.Tanh) else "relu"))
            layer_count += 0  # Count activations as layers but they don't contribute to edges/nodes

        elif isinstance(layer, (torch.nn.Sequential, list)):
            # Handle nested structures recursively
            sub_layers, prev_dim, sub_layer_count, sub_node_count, sub_edge_count = extract_torch_layers(layer,
                                                                                                         prev_dim)
            layers.extend(sub_layers)

            # Update global counts
            layer_count += sub_layer_count
            node_count += sub_node_count
            edge_count += sub_edge_count

        else:
            raise ValueError(f"Unhandled layer type: {type(layer)}")

    return layers, prev_dim, layer_count, node_count, edge_count


def sb3_to_tensorflow(sb3_model, env):
    """
    Converts an SB3 model to TensorFlow and computes layer, node, and edge counts.
    """

    do_profiling = True

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    print(f"{sb3_model.policy = }")

    # Extract layers from SB3 model
    sb3_layers = [
        sb3_model.policy.mlp_extractor.policy_net,
        sb3_model.policy.action_net
    ]
    mlp_extractor_layers = [sb3_model.policy.mlp_extractor.policy_net]
    actor_layers = [sb3_model.policy.action_net]
    critic_layers = [sb3_model.policy.value_net]

    tf_extractor_layers, extractor_output_dim, extractor_layer_count, extractor_node_count, extractor_edge_count = \
        extract_torch_layers(mlp_extractor_layers, input_dim)
    # Add output layer nodes
    #extractor_node_count += output_dim

    tf_actor_layers, actor_output_dim, actor_layer_count, actor_node_count, actor_edge_count = extract_torch_layers(actor_layers, extractor_output_dim)
    actor_node_count += actor_output_dim

    tf_full_actor_model = tf_extractor_layers + tf_actor_layers

    if do_profiling:
        _, critic_output_dim, critic_layer_count, critic_node_count, critic_edge_count = extract_torch_layers(critic_layers, extractor_output_dim)
        critic_node_count += critic_output_dim

        print(f"Extractor layers: {extractor_layer_count}\n"
              f"Actor layers: {actor_layer_count}\n"
              f"Critic layers: {critic_layer_count}\n"
              f"Total layers:{extractor_layer_count +  actor_layer_count + critic_layer_count}")
        print(f"Extractor nodes: {extractor_node_count}\n"
              f"Actor nodes: {actor_node_count}\n"
              f"Critic nodes: {critic_node_count}\n"
              f"Total nodes:{extractor_node_count +  actor_node_count + critic_node_count}")
        print(f"Extractor edges: {extractor_edge_count}\n"
              f"Actor edges: {actor_edge_count}\n"
              f"Critic edges: {critic_edge_count}\n"
              f"Total edges:{extractor_edge_count +  actor_edge_count + critic_edge_count}")

    return TFPolicy(input_dim=input_dim, output_dim=output_dim, hidden_layers=tf_full_actor_model)


def tf_to_tflite(tf_model, export_path, extra_header_defs=None):
    """
    Converts a TensorFlow model to TensorFlow Lite format.
    """
    try:
        # Generate concrete function and convert to TFLite
        concrete_func = tf_model.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=[])
        converter.optimizations = []
        tflite_model = converter.convert()

        # Save the converted model
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "wb") as f:
            f.write(tflite_model)

        # Export additional information
        g_model_length = len(tflite_model)
        header_path = os.path.join(os.path.dirname(export_path), "policy_net_model.h")
        rewrite_policy_net_header(header_path, export_path, g_model_length, extra_header_defs=extra_header_defs)
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")


def test_sb3_tf_model_conversion(sb3_model, tf_model: TFPolicy):
    """
    Validates the conversion from SB3 model to TensorFlow by comparing output probabilities.
    Computes and prints the mean absolute and relative difference between the outputs.
    """
    tolerance = {"abs": 2e-6, "rel": 2e-5}

    sb3_input_dim = sb3_model.observation_space.shape[0]
    tf_input_dim = tf_model.input_dim

    assert sb3_input_dim == tf_input_dim, f"Input dimensions must match. {sb3_input_dim = } || {tf_input_dim = } "
    abs_diffs = []
    rel_diffs = []
    for _ in range(10_000):
        random_input = np.random.random((1, sb3_input_dim)).astype(np.float32)
        sb3_output = sb3_get_action_probabilities(random_input.flatten(), sb3_model).flatten()
        tf_output = tf_model.call(tf.convert_to_tensor(random_input)).numpy().flatten()

        abs_diff = np.abs(sb3_output - tf_output)
        rel_diff = np.abs(abs_diff / (np.abs(sb3_output) + 1e-8))  # Avoid division by zero

        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_diff)

        if not np.allclose(sb3_output, tf_output, atol=tolerance["abs"], rtol=tolerance["rel"]):
            print(f"Mismatch detected!\nSB3: {sb3_output}\nTF: {tf_output}")
    mean_abs_diff = np.mean(abs_diffs)
    mean_rel_diff = np.mean(rel_diffs)
    print(f"Mean Absolute Difference: {mean_abs_diff}")
    print(f"Mean Relative Difference: {mean_rel_diff}")
    print("Completed test")


def sb3_to_tflite_pipeline(relative_model_path):
    # For loading models generated in python3.7
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = PPO.load(relative_model_path, print_system_info=True, custom_objects=custom_objects, device="cpu")
    env = make_vec_env(TwoDEnv, n_envs=1, env_kwargs=dict())
    max_send_interval = env.get_attr("max_send_interval")[0]

    tf_model = sb3_to_tensorflow(model, env)
    test_sb3_tf_model_conversion(sb3_model=model, tf_model=tf_model)

    config = load_config("config.json")
    export_model_path = config['model_path']

    header_defs = {
        "const int MAX_SEND_INTERVAL": max_send_interval
    }

    tf_to_tflite(tf_model, export_model_path, extra_header_defs=header_defs)


if __name__ == '__main__':
    random.seed(0)
    sb3_to_tflite_pipeline("stable-model-2d-best/best_model")
