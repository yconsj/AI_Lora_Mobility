import time
import numpy as np
import tensorflow as tf

from sim_runner import OmnetEnv
from tf_exporter import tf_export
import ast
from baselines3.utilities import load_config, export_training_info, InputMembers, denormalize_input_state
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
        # Forward pass through the layers
        x = self.bn_input(x)
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


reward_sums = []
all_actions_per_episode = []
all_states_per_episode = []
stationary_data_list = []  # Global variable to store stationary data
sim_time_duration = (60 * 60 * 12.0)
norm_factors = {InputMembers.LATEST_PACKET_RSSI.value: 1.0 / 255.0,
                InputMembers.LATEST_PACKET_SNIR.value: 1.0 / 100.0,
                InputMembers.LATEST_PACKET_TIMESTAMP.value: 1.0 / sim_time_duration,
                # max received packets: half a day in seconds, 500 seconds between each transmission, 2 nodes
                InputMembers.NUM_RECEIVED_PACKETS.value: 1.0 / (sim_time_duration / 500.0) * 2.0,
                InputMembers.CURRENT_TIMESTAMP.value: 1.0 / sim_time_duration,
                InputMembers.COORD_X.value: 1.0 / 3000.0,
                }


def plot_training(log_state, mv_actions_per_episode, mv_stationary_data, mv_reward_sums, first_episode=0,
                  last_episode=-1, window_size=100):
    plot_stationary_data = False
    # --- First Figure with Two Subplots ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Denormalized episodes
    denormalized_episodes = [[denormalize_input_state(state, norm_factors) for state in episode]
                             for episode in log_state]
    # Left Subplot: X-Position Over Time for First and Last Episode.

    denormalized_states_first_episode = denormalized_episodes[first_episode]
    first_episode_positions = [state[InputMembers.COORD_X.value] for state in denormalized_states_first_episode]

    denormalized_states_last_episode = denormalized_episodes[last_episode]
    last_episode_positions = [state[InputMembers.COORD_X.value] for state in denormalized_states_last_episode]

    time_steps_first = list(range(len(first_episode_positions)))

    # Introduce a horizontal offset for the last episode
    offset = 0.5  # Adjust this value for more or less offset
    time_steps_last = [t + offset for t in range(len(last_episode_positions))]

    ax1.plot(time_steps_first, first_episode_positions, label="X Position (First Episode)", color="blue", alpha=0.7)
    ax1.plot(time_steps_last, last_episode_positions, label="X Position (Last Episode)", color="red", alpha=0.7)
    # ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
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
    # TODO: remove the stationary packets from this plot
    if plot_stationary_data:
        if len(mv_stationary_data) < len(mv_reward_sums):
            # Padding with zeros or handling missing data
            mv_stationary_data += [0] * (len(mv_reward_sums) - len(mv_stationary_data))
        elif len(mv_stationary_data) > len(mv_reward_sums):
            mv_stationary_data = mv_stationary_data[:len(mv_reward_sums)]

    # get the  NUM_RECEIVED_PACKETS of the last state in each (denormalized) episode
    mobile_gateway_packets = [episode[-1][InputMembers.NUM_RECEIVED_PACKETS.value] for episode in denormalized_episodes]

    fig2, ax3 = plt.subplots(figsize=(10, 5))
    # ax3.plot(mv_reward_sums, color='b', alpha=0.7, label="Episode Rewards", linestyle='-')
    # Plot mobile gateway packets over episodes
    ax3.plot(mobile_gateway_packets, label='Mobile Gateway Packets', color='green', linestyle='-.')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Packets")
    ax3.set_title("Packets Over Episodes")
    ax3.set_ylim(0, None)
    if plot_stationary_data:
        ax3.plot(mv_stationary_data, label='Sum of Packets (Stationary Nodes)', color='orange', linestyle='--')
    ax3.legend()
    # Plotting the Stationary Data List on the same axes

    # TODO: plot first&last episode model packets vs stationary packets (first&last episode) over time steps (4 lines)

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


def get_packet_reward(current_episode, max_episode):
    return 10


def get_exploration_reward(current_episode, max_episode):
    reward = 10 * (1.0 - (current_episode / max_episode))
    return reward


def get_random_choice_probability(current_episode, max_episode):
    # TODO: read if this has to be handled a certain way, w.r.t. backpropagation ( action probability )
    return 0.5 * (1 - current_episode / (max_episode / 2)) ** 2


def reinforce(env, policy_net, optimizer, gen_model_path, log_path, num_episodes, batch_size):
    from control_sim_runner import load_stationary_data, update_stationary_data_list  # avoid circular dependency
    global stationary_data_list

    # Load stationary data first time
    stationary_data_json = load_stationary_data()

    training_info_export_path = config["training_info_path"]
    export_training_info(training_info_export_path, current_episode_num=0, max_episode_num=num_episodes, packet_reward=0,
                         exploration_reward=0, random_choice_probability=0, normalization_factors=norm_factors)

    for episode in range(num_episodes):
        print(f"Running episode {episode + 1} of {num_episodes}.")

        packet_reward = get_packet_reward(episode + 1, num_episodes)
        exploration_reward = get_exploration_reward(episode + 1, num_episodes)
        random_choice_probability = get_random_choice_probability(episode + 1, num_episodes)
        export_training_info(training_info_export_path, episode + 1, num_episodes, packet_reward,
                             exploration_reward, random_choice_probability, norm_factors)

        env.run_simulation(episode, batch_size)
        accumulated_grads = [tf.zeros_like(var) for var in policy_net.trainable_variables]

        # Update the stationary data list for the current episode
        # new_data = update_stationary_data_list(episode, stationary_data_json)
        # stationary_data_list.extend(new_data)  # Append new data to the global list
        sum_rewards = 0
        for batch in range(batch_size):
            print(f"Batch {batch + 1} of {batch_size}")
            states, actions, rewards = read_log(batch, log_path)

            state_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
            returns = []
            cumulative_reward = 0
            for r in rewards[::-1]:  # Reverse to compute returns
                cumulative_reward = r + cumulative_reward * 0.999  # Discount factor
                returns.insert(0, cumulative_reward)  # Insert at the beginning
            sum_rewards += sum(rewards)

            if batch == 0:  # only sample the first batch for later plotting.
                print("total rewards: " + str(sum(rewards)))
                reward_sums.append(sum_rewards)
                all_actions_per_episode.append(actions)  # Store actions for plotting avg action
                all_states_per_episode.append(states)

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
                beta = 0.5 * (1 - episode / num_episodes) ** 0.5  # Scale beta down the further in training
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
    # Start timing the overall training process
    start_time = time.time()

    env = OmnetEnv()
    input_size = 6  # State size
    output_size = 2  # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)  # Initialize optimizer

    num_episodes = 0  # Number of episodes to train
    num_batches = 4
    concrete_func = policy_net.get_concrete_function()
    policy_net.summary()

    config = load_config("config.json")
    log_path = config['logfile_path']
    gen_model_path = config['model_path']
    tf_export(concrete_func, export_model_path, 0)  # initial model
    reinforce(env, policy_net, optimizer, gen_model_path, log_path, num_episodes, num_batches)  # Train the agent
    print('Complete')

    # Timing the training duration
    end_time = time.time()
    total_training_time = end_time - start_time  # Total training time in seconds
    # Calculate the average time per episode
    if (num_episodes > 0):
        avg_time_per_episode = total_training_time / num_episodes
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Average time per episode: {avg_time_per_episode:.2f} seconds")

    # Example input (make sure it matches the input shape of your model)
    # rrsi, snir, timestamp__lastpacket, total_packets, time, x,y

    if num_episodes > 1:
        plot_training(log_state=all_states_per_episode, mv_actions_per_episode=all_actions_per_episode,
                      mv_stationary_data=stationary_data_list, mv_reward_sums=reward_sums, window_size=100)


if __name__ == "__main__":
    main()
