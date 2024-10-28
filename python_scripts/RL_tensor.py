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
class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,) ,                                   
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(),  # Initialize weights to zeros
                                    bias_initializer=tf.keras.initializers.GlorotUniform())    # Initialize biases to zeros)
        self.fc2 = tf.keras.layers.Dense(output_dim, 
                                    activation='softmax', 
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(),  # Initialize weights to zeros
                                    bias_initializer=tf.keras.initializers.GlorotUniform())    # Initialize biases to zeros
    def call(self, x):
        return self.fc2(self.fc1(x))
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
    
def plot_rewards(show_result=False, window=10):
    plt.figure(figsize=(10, 5))

    label = "Episode Rewards"
    plt.clf()
    plt.plot(reward_sums, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Episode")
    plt.legend()




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
    #log_state = re.sub(r'\s*m', '', log_state.strip())
    
    # Convert the string to an actual Python list
    states = ast.literal_eval(log_state)
    actions = ast.literal_eval(log_actions)
    rewards = ast.literal_eval(log_rewards)
    # Append the cleaned list to a combined list
    print("finished reading log")
    return states, actions, rewards
reward_sums=[]
def reinforce(env,policy_net, optimizer, num_episodes):
    for episode in range(num_episodes):
        env.run_simulation(episode)
        states, actions, rewards = read_log()
        state_tensor    = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor  = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns = []
        cumulative_reward = 0
        for r in rewards[::-1]:  # Reverse to compute returns
            cumulative_reward = r + cumulative_reward * 0.99  # Discount factor
            returns.insert(0, cumulative_reward)  # Insert at the beginning
        print("total rewards: "+ str(sum(rewards)))
        reward_sums.append(sum(rewards))
        #plot_rewards(window=10)
        # Convert lists to tensors
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)  # Convert returns to tensor

        # Compute policy loss
        with tf.GradientTape() as tape:
            # Get the action probabilities
            action_probs = policy_net(state_tensor)  # Assuming this outputs probabilities for actions
            log_probs = tf.math.log(tf.clip_by_value(action_probs, 1e-10, 1.0))  # Log probabilities
            
            # Gather log probabilities for selected actions
            selected_log_probs = tf.reduce_sum(log_probs * tf.one_hot(actions_tensor, policy_net.output_dim), axis=1)  # Gather log probabilities
            
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
        print(policy_net.fc2.get_weights())
        

        print("exporting model")
        concrete_func = policy_net.get_concrete_function()
        config = load_config("config.json")
        gen_model = config['model_path']
        tf_export(concrete_func, gen_model, episode+1)

# Main function to run the training
config = load_config("config.json")
gen_model = config['model_path']
export_model_path = gen_model
def main():

    env = OmnetEnv()
    input_size = 6 # State size
    output_size = 2 # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Initialize optimizer

    num_episodes = 10  # Number of episodes to train
    concrete_func = policy_net.get_concrete_function()
    policy_net.summary()
   
    tf_export(concrete_func, export_model_path, 0) # initial model
    reinforce(env, policy_net, optimizer, num_episodes)  # Train the agent
    plot_rewards(show_result=True, window=10)
    print('Complete')



    # Example input (make sure it matches the input shape of your model)
    # rrsi, snir, timestamp__lastpacket, total_packets, time, x,y
    input1 = [0, 0, 0, 0, 0.1 / 86400, 500 / 3000]
    input2 = [-133.837 / 255, 2.07141 / 100 , 672.22626462552 / 86400, 2 / 100, 970.3 / 86400, 458.555 /3000]
    
    example_input1 = tf.constant([input1], dtype=tf.float32)  # Example state with 4 features
    example_input2 = tf.constant([input2], dtype=tf.float32)  # Example state with 4 features
    # Define the mean and std for each input, calculated from past observations
    # Run the network
    action_probs1 = policy_net(example_input1)
    action_probs2 = policy_net(example_input2)
    # Print the output probabilities
    print("Action1 probabilities:", action_probs1.numpy())
    print("Action2 probabilities:", action_probs2.numpy())

    plt.ioff()
    plt.show()
if __name__ == "__main__":
    main()
