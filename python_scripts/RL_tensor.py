import numpy as np
import tensorflow as tf
import matplotlib
from sim_runner import OmnetEnv
from tf_exporter import tf_export
import ast
import re
import os
import json
import matplotlib.pyplot as plt
import pandas as pd


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,),
                                         kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                         # Initialize weights to zeros
                                         bias_initializer=tf.keras.initializers.GlorotUniform())  # Initialize biases to zeros)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu',
                                         kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                         # Initialize weights to zeros
                                         bias_initializer=tf.keras.initializers.GlorotUniform())
        self.fc3 = tf.keras.layers.Dense(output_dim,
                                         activation='softmax',
                                         kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                         # Initialize weights to zeros
                                         bias_initializer=tf.keras.initializers.GlorotUniform())  # Initialize biases to zeros

    def call(self, x):
        return self.fc3(self.fc2(self.fc1(x)))

    def get_concrete_function(self):
        # Create a concrete function for the model
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32)])
        def concrete_function(x):
            return self.call(x)

        return concrete_function.get_concrete_function()


def load_config(config_file):
    # Get the directory of the currently running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the config file
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        return json.load(f)


def plot_training(log_state, all_actions_per_episode, reward_sums, first_episode=0, last_episode=-1, window_size=100):
    # --- First Figure with Two Subplots ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Left Subplot: X-Position Over Time for First and Last Episode
    first_episode_positions = [state[-1] for state in log_state[first_episode]]
    last_episode_positions = [state[-1] for state in log_state[last_episode]]
    time_steps_first = list(range(len(first_episode_positions)))
    time_steps_last = list(range(len(last_episode_positions)))

    ax1.plot(time_steps_first, first_episode_positions, label="X Position (First Episode)", color="blue", alpha=0.7)
    ax1.plot(time_steps_last, last_episode_positions, label="X Position (Last Episode)", color="red", alpha=0.7)
    ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("X Position")
    ax1.set_title("X Position Over Time")
    ax1.legend()

    # Right Subplot: Smoothed Actions for First and Last Episode
    first_episode_actions = all_actions_per_episode[first_episode]
    last_episode_actions = all_actions_per_episode[last_episode]

    # Compute the moving average (smoothing)
    smoothed_first = np.convolve(first_episode_actions, np.ones(window_size) / window_size, mode='valid')
    smoothed_last = np.convolve(last_episode_actions, np.ones(window_size) / window_size, mode='valid')

    # Time steps for smoothed actions
    time_steps_first_smoothed = list(range(len(smoothed_first)))
    time_steps_last_smoothed = list(range(len(smoothed_last)))

    ax2.plot(time_steps_first_smoothed, smoothed_first, label="Smoothed Actions (First Episode)", color="cyan", alpha=0.7)
    ax2.plot(time_steps_last_smoothed, smoothed_last, label="Smoothed Actions (Last Episode)", color="magenta", alpha=0.7)
    ax2.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Action Value")
    ax2.set_title("Smoothed Actions Over Time")
    ax2.legend()

    # Adjust layout
    fig1.tight_layout()

    # --- Second Figure: Rewards Over Time ---
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(reward_sums, color='b', alpha=0.7, label="Episode Rewards")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cumulative Reward")
    ax3.set_title("Cumulative Reward Over Episodes")
    ax3.legend()

    # Show both figures
    plt.show()


def read_log():
    # Step 1: Read the file line by line
    print("reading log")
    config = load_config("config.json")
    log = config['logfile_path']
    with open(log, 'r') as file:
        lines = file.readlines()

    # Step 2: Process each line individually
    states = []
    actions = []
    rewards = []
    log_state = lines[0]
    log_actions = lines[1]
    log_rewards = lines[2]
    # Remove 'm' using regex and clean the line
    # log_state = re.sub(r'\s*m', '', log_state.strip())

    # Convert the string to an actual Python list
    states = ast.literal_eval(log_state)
    actions = ast.literal_eval(log_actions)
    rewards = ast.literal_eval(log_rewards)
    # Append the cleaned list to a combined list
    print("finished reading log")
    return states, actions, rewards


reward_sums = []
all_actions_per_episode = []
all_states_per_episode = []


def reinforce(env, policy_net, optimizer, num_episodes):
    for episode in range(num_episodes):
        env.run_simulation(episode)
        states, actions, rewards = read_log()

        all_states_per_episode.append(states)

        avg_action = sum(actions) / len(actions)
        print(f"Episode {episode + 1} - Avg Action Taken: {avg_action}")

        state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns = []
        cumulative_reward = 0
        for r in rewards[::-1]:  # Reverse to compute returns
            cumulative_reward = r + cumulative_reward * 0.99  # Discount factor
            returns.insert(0, cumulative_reward)  # Insert at the beginning

        print("total rewards: " + str(sum(rewards)))
        reward_sums.append(sum(rewards))
        all_actions_per_episode.append(actions)  # Store actions for plotting avg action


        # plot_rewards(window=10)

        # Convert lists to tensors
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)  # Convert returns to tensor

        # Compute policy loss
        with tf.GradientTape() as tape:
            # Get the action probabilities
            action_probs = policy_net(state_tensor)  # Assuming this outputs probabilities for actions
            log_probs = tf.math.log(tf.clip_by_value(action_probs, 1e-10, 1.0))  # Log probabilities

            # Gather log probabilities for selected actions
            selected_log_probs = tf.reduce_sum(log_probs * tf.one_hot(actions_tensor, policy_net.output_dim),
                                               axis=1)  # Gather log probabilities

            # REINFORCE loss
            policy_loss = -tf.reduce_mean(selected_log_probs * returns_tensor)  # REINFORCE loss

            # Calculate entropy
            entropy = -tf.reduce_sum(action_probs * log_probs, axis=1)  # Entropy calculation
            # Hyperparameter for entropy regularization
            beta = 0.01  # Adjust this value based on your needs
            entropy_loss = beta * tf.reduce_mean(entropy)  # Scale by beta

            # Total loss with entropy regularization
            total_loss = policy_loss + entropy_loss

        # Update the policy
        grads = tape.gradient(total_loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
      
        print("exporting model")
        concrete_func = policy_net.get_concrete_function()
        config = load_config("config.json")
        gen_model = config['model_path']
        tf_export(concrete_func, gen_model, episode + 1)


# Main function to run the training
config = load_config("config.json")
gen_model = config['model_path']
export_model_path = gen_model


def main():
    env = OmnetEnv()
    input_size = 6  # State size
    output_size = 2  # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Initialize optimizer

    num_episodes = 100  # Number of episodes to train
    concrete_func = policy_net.get_concrete_function()
    policy_net.summary()

    tf_export(concrete_func, export_model_path, 0)  # initial model
    reinforce(env, policy_net, optimizer, num_episodes)  # Train the agent
    print('Complete')

    # Example input (make sure it matches the input shape of your model)
    # rrsi, snir, timestamp__lastpacket, total_packets, time, x,y
    input1 = [0, 0, 0, 0, 0.1 / 86400, 500 / 3000]
    input2 = [-133.837 / 255, 2.07141 / 100, 672.22626462552 / 86400, 2 / 100, 970.3 / 86400, 458.555 / 3000]

    example_input1 = tf.constant([input1], dtype=tf.float32)  # Example state with 4 features
    example_input2 = tf.constant([input2], dtype=tf.float32)  # Example state with 4 features
    # Define the mean and std for each input, calculated from past observations
    # Run the network
    action_probs1 = policy_net(example_input1)
    action_probs2 = policy_net(example_input2)
    # Print the output probabilities
    print("Action1 probabilities:", action_probs1.numpy())
    print("Action2 probabilities:", action_probs2.numpy())

    plot_training(log_state=all_states_per_episode,
                  all_actions_per_episode=all_actions_per_episode,
                  reward_sums=reward_sums,
                  window_size=100)

if __name__ == "__main__":
    main()
