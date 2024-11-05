import numpy as np
import tensorflow as tf

from sim_runner import OmnetEnv
from tf_exporter import tf_export
import ast
import os
import json
import matplotlib.pyplot as plt


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs from TensorFlow
# TODO: add function to run the final trained model on N seeds.


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Batch Normalization layer for input
        self.bn_input = tf.keras.layers.BatchNormalization()  # input_shape=(input_dim,)

        self.fc1 = tf.keras.layers.Dense(64, activation='relu',
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

        # Build the model with the specified input shape
        # self.build(input_shape=(None, input_dim))

    def call(self, x):
        # Apply Batch Normalization
        x = self.bn_input(x)

        # Forward pass through the dense layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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


def plot_training(log_state, mv_actions_per_episode, mv_stationary_data, mv_reward_sums, first_episode=0,
                  last_episode=-1, window_size=100):
    # --- First Figure with Two Subplots ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Left Subplot: X-Position Over Time for First and Last Episode
    first_episode_positions = [state[-1] for state in log_state[first_episode]]
    last_episode_positions = [state[-1] for state in log_state[last_episode]]
    time_steps_first = list(range(len(first_episode_positions)))

    # Introduce a horizontal offset for the last episode
    offset = 0.5  # Adjust this value for more or less offset
    time_steps_last = [t + offset for t in range(len(last_episode_positions))]

    ax1.plot(time_steps_first, first_episode_positions, label="X Position (First Episode)", color="blue", alpha=0.7)
    ax1.plot(time_steps_last, last_episode_positions, label="X Position (Last Episode)", color="red", alpha=0.7)
    ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("X Position")
    ax1.set_title("X Position Over Time")
    ax1.legend()

    # Right Subplot: Smoothed Actions for First and Last Episode
    first_episode_actions = mv_actions_per_episode[first_episode]
    last_episode_actions = mv_actions_per_episode[last_episode]

    # Compute the moving average (smoothing)
    smoothed_first = np.convolve(first_episode_actions, np.ones(window_size) / window_size, mode='valid')
    smoothed_last = np.convolve(last_episode_actions, np.ones(window_size) / window_size, mode='valid')

    # Time steps for smoothed actions
    time_steps_first_smoothed = list(range(len(smoothed_first)))
    time_steps_last_smoothed = list(range(len(smoothed_last)))

    ax2.plot(time_steps_first_smoothed, smoothed_first, label="Smoothed Actions (First Episode)", color="blue",
             alpha=0.7)
    ax2.plot(time_steps_last_smoothed, smoothed_last, label="Smoothed Actions (Last Episode)", color="red", alpha=0.7)
    ax2.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Action Value")
    ax2.set_title("Smoothed Actions Over Time")
    ax2.legend()

    # Adjust layout
    fig1.tight_layout()

    # --- Second Figure: Rewards Over Time ---
    # Example of ensuring the lengths match or padding as needed
    if len(mv_stationary_data) < len(mv_reward_sums):
        # Padding with zeros or handling missing data
        mv_stationary_data += [0] * (len(mv_reward_sums) - len(mv_stationary_data))
    elif len(mv_stationary_data) > len(mv_reward_sums):
        mv_stationary_data = mv_stationary_data[:len(mv_reward_sums)]

    fig2, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(mv_reward_sums, color='b', alpha=0.7, label="Episode Rewards", linestyle='-')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Sum of Immediate Reward / Packets")
    ax3.set_title("Sums of Immediate Reward Over Episodes")
    ax3.plot(mv_stationary_data, label='Sum of Packets (Stationary Nodes)', color='orange', linestyle='--')
    ax3.legend()
    # Plotting the Stationary Data List on the same axes

    # TODO: plot packets received over episodes vs stationary

    # Show both figures
    plt.show()


def read_log(batch, log_path):
    # Step 1: Read the file line by line
    print("reading log")

    log = log_path + "_" + str(batch) + ".txt"
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
stationary_data_list = []  # Global variable to store stationary data


def reinforce(env, policy_net, optimizer, gen_model_path, log_path, num_episodes, batch_size):
    from control_sim_runner import load_stationary_data, update_stationary_data_list  # avoid circular dependency
    global stationary_data_list

    # Load stationary data only once
    stationary_data_json = load_stationary_data()

    for episode in range(num_episodes):
        print(f"Running episode {episode + 1} of {num_episodes}.")
        env.run_simulation(episode, batch_size)
        accumulated_grads = [tf.zeros_like(var) for var in policy_net.trainable_variables]

        # Update the stationary data list for the current episode
        new_data = update_stationary_data_list(episode, stationary_data_json)
        stationary_data_list.extend(new_data)  # Append new data to the global list

        for batch in range(batch_size):
            states, actions, rewards = read_log(batch, log_path)

            all_states_per_episode.append(states)

            state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
            returns = []
            cumulative_reward = 0
            for r in rewards[::-1]:  # Reverse to compute returns
                cumulative_reward = r + cumulative_reward * 0.99  # Discount factor
                returns.insert(0, cumulative_reward)  # Insert at the beginning

            if batch == 0:  # only sample the first batch for later plotting.
                print("total rewards: " + str(sum(rewards)))
                reward_sums.append(sum(rewards))
                all_actions_per_episode.append(actions)  # Store actions for plotting avg action

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
                grads = tape.gradient(total_loss, policy_net.trainable_variables)
                # Accumulate gradients if they are not None
                if grads is not None:
                    for i in range(len(grads)):
                        if grads[i] is not None:  # Check if the gradient is not None
                            accumulated_grads[i] += grads[i]  # Accumulate gradients
                        else:
                            print(f"Warning: Gradient for variable {i} is None.")

        # Average gradients after processing all batches
        for i in range(len(accumulated_grads)):
            accumulated_grads[i] /= batch_size  # Average each accumulated gradient
        optimizer.apply_gradients(zip(accumulated_grads, policy_net.trainable_variables))
        concrete_func = policy_net.get_concrete_function()
        print("exporting model")
        tf_export(concrete_func, gen_model_path, (episode * batch_size) + 1)


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

    num_episodes = 400  # Number of episodes to train
    num_batches = 5
    concrete_func = policy_net.get_concrete_function()
    policy_net.summary()

    config = load_config("config.json")
    log_path = config['logfile_path']
    gen_model_path = config['model_path']
    tf_export(concrete_func, export_model_path, 0)  # initial model
    reinforce(env, policy_net, optimizer, gen_model_path, log_path, num_episodes, num_batches)  # Train the agent
    print('Complete')

    # Example input (make sure it matches the input shape of your model)
    # rrsi, snir, timestamp__lastpacket, total_packets, time, x,y

    if num_episodes > 0:
        plot_training(log_state=all_states_per_episode, mv_actions_per_episode=all_actions_per_episode,
                      mv_stationary_data=stationary_data_list, mv_reward_sums=reward_sums, window_size=100)


if __name__ == "__main__":
    main()
