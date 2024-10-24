import numpy as np
import tensorflow as tf
from sim_runner import OmnetEnv
from tf_exporter import tf_export
import ast
import re
import os
import json

class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(PolicyNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,))
        self.fc2 = tf.keras.layers.Dense(output_dim, 
                                    activation='softmax',
                                    kernel_initializer=tf.keras.initializers.Zeros(),  # Initialize weights to zeros
                                    bias_initializer=tf.keras.initializers.Zeros())    # Initialize biases to zeros

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

def read_log():
    # Step 1: Read the file line by line
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
    return states, actions, rewards
def reinforce(env, policy_net, optimizer, num_episodes):
    for episode in range(num_episodes):
        #env.run_simulation()
        # TODO: get logged state
        # TODO: get logged ations
        # TODO: get logged rewards
        states, actions, rewards = read_log()
        #states = [(a, b, c, d, *e) for a, b, c, d, e in states] # flatten states pos
        state_tensor    = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor  = tf.convert_to_tensor(actions, dtype=tf.int32)
        print(state_tensor)
        print(actions_tensor)
        # state = -1 # get state
        # done = False
        # states, actions, rewards = [], [], []

        # while not done:
        #     state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)  # Convert state to tensor
        #     action_probs = policy_net(tf.expand_dims(state_tensor, axis=0))  # Get action probabilities
        #     action = np.random.choice(len(action_probs.numpy().squeeze()), p=action_probs.numpy().squeeze())  # Sample action
        #     next_state, reward, done, _ = env.step(action)  # Take action in the environment
            
        #     states.append(state)  # Store state
        #     actions.append(action)  # Store action
        #     rewards.append(reward)  # Store reward
            
        #     state = next_state  # Transition to the next state

        # Compute the cumulative rewards (returns)
        returns = []
        cumulative_reward = 0
        for r in rewards[::-1]:  # Reverse to compute returns
            cumulative_reward = r + cumulative_reward * 0.99  # Discount factor
            returns.insert(0, cumulative_reward)  # Insert at the beginning

        # Convert lists to tensors
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)  # Convert returns to tensor

        # Compute policy loss
        with tf.GradientTape() as tape:
            log_probs = tf.math.log(policy_net(tf.convert_to_tensor(states, dtype=tf.float32)))  # Get log probabilities
            selected_log_probs = tf.reduce_sum(log_probs * tf.one_hot(actions_tensor, policy_net.output_dim), axis=1)  # Gather log probabilities
            loss = -tf.reduce_mean(selected_log_probs * returns_tensor)  # REINFORCE loss

        # Update the policy
        grads = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

        # Print episode statistics
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.numpy():.4f}")
        # Export the model

        concrete_func = policy_net.get_concrete_function()
        tf_export(concrete_func, "path_model.c")

        return

# Main function to run the training
export_model_path = "C:/Users/simon/Desktop/AI_Lora_Mobility/inet4.4/src/inet/mobility/RL/modelfiles/gen_model.cc"
def main():

    env = OmnetEnv()
    input_size = 7 # State size
    output_size =  2 # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Initialize optimizer

    num_episodes = 10  # Number of episodes to train
    concrete_func = policy_net.get_concrete_function()
    tf_export(concrete_func, export_model_path)
    #reinforce(env, policy_net, optimizer, num_episodes)  # Train the agent

if __name__ == "__main__":
    main()
