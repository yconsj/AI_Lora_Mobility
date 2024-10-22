import numpy as np
import tensorflow as tf
from sim_runner import OmnetEnv
from tf_exporter import tf_export
class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,))
        self.fc2 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        return self.fc2(self.fc1(x))


def reinforce(env, policy_net, optimizer, num_episodes):
    for episode in range(num_episodes):
        env.run_simulation()
        # TODO: get logged state
        # TODO: get logged ations
        # TODO: get logged rewards



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
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)  # Convert actions to tensor

        # Compute policy loss
        with tf.GradientTape() as tape:
            log_probs = tf.math.log(policy_net(tf.convert_to_tensor(states, dtype=tf.float32)))  # Get log probabilities
            selected_log_probs = tf.reduce_sum(log_probs * tf.one_hot(actions_tensor, policy_net.output_shape[-1]), axis=1)  # Gather log probabilities
            loss = -tf.reduce_mean(selected_log_probs * returns_tensor)  # REINFORCE loss

        # Update the policy
        grads = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

        # Print episode statistics
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.numpy():.4f}")

        # Export the model
        tf_export(concrete_func, path)

# Main function to run the training
def main():
    env = OmnetEnv()
    env.run_simulation()
    input_size = -1 # State size
    output_size =  -1 # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Initialize optimizer

    num_episodes = 10  # Number of episodes to train
    reinforce(env, policy_net, optimizer, num_episodes)  # Train the agent

if __name__ == "__main__":
    main()
