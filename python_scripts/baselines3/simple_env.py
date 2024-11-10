from stable_baselines3 import A2C, PPO
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import cv2

class SimpleBaseEnv(gym.Env):
    def __init__(self, render_mode="cv2"):
        super(SimpleBaseEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(2, start= 0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment state
        self.state = 50
        self.target = 5  # The target value we want to reach
        self.max_steps = 500  # Maximum steps per episode
        self.steps = 0
        self.node_1x = node(10)
        self.node_2x = node(90)
        self.total_reward = 0
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([100, self.max_steps]), dtype=np.float32)

        self.width, self.height = 100, 20  # Size of the window
        self.point_radius = 1
        self.point_color = (0, 0, 255)  # Red color
        self.line_color = (255, 0, 0)   # Blue color
        self.window_name = "RL Animation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    def reset(self, seed=None, options=None):
        # Reset the state and steps counter
        self.state = 50
        self.steps = 0
        self.total_reward = 0
        return np.array([self.state, self.steps], dtype=np.float32), {}

    def step(self, action):
       # Update the environment state over 10 stamps

        reward = 0
        if action == 0:
            if self.state == 0:
                self.state = 0
            else:
                self.state -= 1  # Action -1
        elif action == 1:
            if self.state == 100:
                self.state = 100
            else:
                self.state += 1  # Action +1

        # Update step count and check if episode is done
        self.steps += 1
        
        if self.node_1x.send(self.steps, self.state):
            reward+= 10
        if self.node_2x.send(self.steps, self.state):
            reward+= 10
        done = self.steps >= self.max_steps
        self.total_reward += reward
        return np.array([self.state, self.steps], dtype=np.float32), reward, done, False, {}

    def render(self):
        # Map the position [0, 1] to the x-coordinate along the line [50, 550]
        x = int(self.state)
        y = 5
        # Create a new black image
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw the line and moving point
        cv2.line(frame,pt1=(0, y), pt2=(100,y), color=self.line_color)
        cv2.rectangle(frame,pt1= (x-2, y-2), pt2= (x+2, y+2), color=self.point_color)

        # Draw nodes
        cv2.rectangle(frame,pt1= (self.node_1x.pos-2, y-2), pt2= (self.node_1x.pos+2, y+2), color=self.point_color)
        cv2.rectangle(frame,pt1= (self.node_2x.pos-2, y-2), pt2= (self.node_2x.pos+2, y+2), color=self.point_color)
        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        # Draw score
        cv2.putText(enlarged_image, "Total score: " + str(self.total_reward), (500,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imshow(self.window_name, enlarged_image)
        cv2.waitKey(30)  # Wait a short time to create the animation effect
    def close(self):
        cv2.destroyAllWindows()


class node():
    def __init__(self, pos=10):
        self.pos = pos
        self.last_packet_time = 0
    def send(self, time, gpos):
        # Decides whether a packet should be send and if it gets lost
        # Pobability of success is based of last time send and distance
        elapsed_time = time - self.last_packet_time
        send_scale = 1000
        send_probability = 1 - np.exp(-elapsed_time / send_scale)
        ploss_scale=100
        distance = abs(self.pos - gpos)
        ploss_probability = 1 - np.exp(- distance/ploss_scale)

        send_choice = np.random.rand() < send_probability
        ploss_choice = np.random.rand() < ploss_probability
        if send_choice:
            self.last_send_time = time
            if ploss_choice: 
                #print(f"Packet sent at time {time} with probability {send_probability:.2f}, but was lost with probability {ploss_probability:.2f}")
                return False
            #print(f"Packet sent at time {time} with probability {send_probability:.2f}")
            return True
        else:
            #print(f"Packet NOT sent. Probability was {send_probability:.2f}")
            return False