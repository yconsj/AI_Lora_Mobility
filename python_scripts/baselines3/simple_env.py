from stable_baselines3 import A2C, PPO
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

class SimpleBaseEnv(gym.Env):
    def __init__(self):
        super(SimpleBaseEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(2, start= 0)
        # The observation space is a single value (our current "position")
        self.observation_space = spaces.Box(low=np.array([-10]), high=np.array([10]), dtype=np.float32)

        # Environment state
        self.state = 0
        self.target = 5  # The target value we want to reach
        self.max_steps = 50  # Maximum steps per episode
        self.steps = 0

    def reset(self):
        # Reset the state and steps counter
        self.state = np.random.uniform(-10, 10)
        self.steps = 0
        return np.array([self.state], dtype=np.float32), {}

    def step(self, action):
        # Update the environment state
        if action == 0:
            self.state -= 1  # Action -1
        elif action == 2:
            self.state += 1  # Action +1

        # Calculate the reward (closer to the target = better reward)
        reward = -abs(self.state - self.target)

        # Update step count and check if episode is done
        self.steps += 1
        done = self.steps >= self.max_steps or abs(self.state - self.target) < 0.1

        return np.array([self.state], dtype=np.float32), reward, done, False, {}

    def render(self):
        print(f"State: {self.state} | Steps: {self.steps}")


env = SimpleBaseEnv()
# Define and Train the agent
check_env(env)
model = PPO("CnnPolicy", env).learn(total_timesteps=1000)