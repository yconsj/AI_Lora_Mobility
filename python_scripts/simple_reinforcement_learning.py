# reinforcemnt learniung script
# used for the simple simulation environent
# trains the model to decide whether to go left or right
# the simple environment will have two nodes that send packets
# with defined non-overlapping intervals
# the goal is that the model will tell the gateway where go go next based on a certain time


# input parameters: 
# position x
# position y 
# time 

import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # Output probabilities
        return x
    


def reinforce(env, policy_net, optimizer, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()  # should run omnet
        done = False
        states, actions, rewards = [], [], []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            action_probs = policy_net(state_tensor)  # Get action probabilities
            action = np.random.choice(len(action_probs.squeeze()), p=action_probs.detach().numpy())  # Sample action
            next_state, reward, done, _ = env.step(action)  # Take action in the environment
            
            states.append(state)  # Store state
            actions.append(action)  # Store action
            rewards.append(reward)  # Store reward
            
            state = next_state  # Transition to the next state

        # Compute the cumulative rewards (returns)
        returns = []
        cumulative_reward = 0
        for r in rewards[::-1]:  # Reverse to compute returns
            cumulative_reward = r + cumulative_reward * 0.99  # Discount factor
            returns.insert(0, cumulative_reward)  # Insert at the beginning

        # Convert lists to tensors
        returns_tensor = torch.FloatTensor(returns)  # Convert returns to tensor
        actions_tensor = torch.LongTensor(actions)  # Convert actions to tensor

        # Compute policy loss
        log_probs = policy_net(torch.FloatTensor(states)).gather(1, actions_tensor.unsqueeze(1)).squeeze()  # Get log probabilities
        loss = -torch.mean(log_probs * returns_tensor)  # REINFORCE loss

        # Update the policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print episode statistics
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item():.4f}")



# Main function to run the training
def main():
    env = gym.make('CartPole-v1')  # Change this to your desired environment
    input_size = env.observation_space.shape[0]  # State size
    output_size = env.action_space.n  # Number of actions

    policy_net = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)  # Initialize optimizer

    num_episodes = 1000  # Number of episodes to train
    reinforce(env, policy_net, optimizer, num_episodes)  # Train the agent

if __name__ == "__main__":
    main()
