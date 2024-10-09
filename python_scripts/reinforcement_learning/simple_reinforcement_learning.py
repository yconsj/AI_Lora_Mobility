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
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sim_runner import OMNeTSimulation

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # Output probabilities
        return x
    


def reinforce(env, policy_net, optimizer, num_episodes, episodes_json):
    with open(episodes_json, 'r') as file:
        data = json.load(file)
    for episode in range(num_episodes):
        # Run simulation
        env.run_simulation()

       # Extract states, actions, and rewards for this episode
        states = data['episodes'][episode]['states']
        actions = data['episodes'][episode]['actions']
        rewards = data['episodes'][episode]['rewards']

        # Compute the cumulative rewards (returns)
        returns = []
        cumulative_reward = 0
        for r in rewards[::-1]:  # Reverse to compute returns
            cumulative_reward = r + cumulative_reward * 0.99  # Discount factor
            returns.insert(0, cumulative_reward)  # Insert at the beginning

        # Convert lists to tensors
        states_tensor = torch.FloatTensor(states)
        returns_tensor = torch.FloatTensor(returns)  # Convert returns to tensor
        actions_tensor = torch.LongTensor(actions)  # Convert actions to tensor

        # Compute policy loss
        log_probs = policy_net(torch.FloatTensor(states_tensor)).gather(1, actions_tensor.unsqueeze(1)).squeeze()  # Get log probabilities
        loss = -torch.mean(log_probs * returns_tensor)  # REINFORCE loss

        # Update the policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Export updated model
        ONNXexport(policy_net)

def ONNXexport(model):
    # Export the model
    onnx_file_path = "simple_model.onnx"
    dummy_input = torch.randn(1, 4)  # Create a dummy input with shape (1, 4)
    torch.onnx.export(
        model,                          # Model to be exported
        dummy_input,                    # Dummy input to define the input shape
        onnx_file_path,                 # Path where the model will be saved
        export_params=True,             # Store the trained parameter weights inside the model file
        opset_version=11,               # ONNX version to export the model to
        do_constant_folding=True,       # Optimization
        input_names=["posX", "posY", "time", "packets"],          # Name of the input layer
        output_names=["left","right"],                            # Name of the output layer
        dynamic_axes={                                              # Variable length axes
            "input": {0: "batch_size"},                           
            "output": {0: "batch_size"},
        },
    )



# Main function to run the training
def main():
    env = OMNeTSimulation(config_file='config.json')
    input_size      = 4 # state size
    output_size     = 2 # left or right

    policy_net      = PolicyNetwork(input_size, output_size)  # Initialize policy network
    optimizer       = optim.Adam(policy_net.parameters(), lr=0.01)  # Initialize optimizer

    num_episodes    = 1000  # Number of episodes to train

    reinforce(env, policy_net, optimizer, num_episodes)  # Train the agent

if __name__ == "__main__":
    main()
