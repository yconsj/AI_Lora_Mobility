from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import cv2


class SimpleBaseEnv(gym.Env):
    def __init__(self, render_mode="cv2"):
        super(SimpleBaseEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(2, start=0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = 500  # Maximum steps per episode
        # Observation_space = pos, time, rssi1, snir1, timestamp1, rssi2, snir2, timestamp2
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0,0,0]), high=np.array([1, 1,1,1,1,1,1,1]), dtype=np.float32)
        # Environment state
        speed = 20  # meter per second
        max_distance = 3000 # meters
        self.max_distance = int(max_distance / speed)  # scaled by speed
        self.pos = self.max_distance / 2
        self.target = 5  # The target value we want to reach
        self.steps = 0
        self.node_1x = node(0, time_to_first_packet=5, send_interval=10)
        self.node_2x = node(self.max_distance, time_to_first_packet=10, send_interval=10)
        self.total_reward = 0
        self.pos = int(self.max_distance / 2)  # Start in the middle
        self.rssi1 = 0
        self.rssi2 = 0
        self.snir1 = 0
        self.snir2 = 0
        self.timestamp1 = 0
        self.timestamp2 = 0

        self.width, self.height = 200, 20  # Size of the window
        self.point_radius = 1
        self.point_color = (0, 0, 255)  # Red color
        self.line_color = (255, 0, 0)  # Blue color
        self.window_name = "RL Animation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def reset(self, seed=None, options=None):
        # Reset the.pos and steps counter
        self.pos = int(self.max_distance/2)
        self.node_1x.reset()
        self.node_2x.reset()
        self.steps = 0
        self.total_reward = 0
        self.rssi1 = 0
        self.rssi2 = 0
        self.snir1 = 0
        self.snir2 = 0
        self.timestamp1 = 0
        self.timestamp2 = 0
        return np.array([self.pos / self.max_distance, self.steps / self.max_steps, self.rssi1, self.snir1,
                        self.timestamp1, self.rssi2, self.snir2, self.timestamp2], dtype=np.float32), {}

    def step(self, action):
       # Update the environment pos over 10 stamps

        reward = 0
        if action == 0:
            if self.pos == 0:
                self.pos = 0
            else:
                self.pos -= 1  # Action -1
        elif action == 1:
            if self.pos == 100:
                self.pos = 100
            else:
                self.pos += 1  # Action +1

        # Update step count and check if episode is done
        self.steps += 1
        recieved1, rssi1, snir1 = self.node_1x.send(self.steps, self.pos)
        recieved2, rssi2, snir2 = self.node_1x.send(self.steps, self.pos)
        if recieved1:
            reward+= 10
            self.rssi1 = rssi1
            self.snir1 = snir1
            self.timestamp1 = self.steps
        if recieved2:
            reward+= 10
            self.rssi2 = rssi2
            self.snir2 = snir2
            self.timestamp2 = self.steps

        done = self.steps >= self.max_steps
        self.total_reward += reward
        return np.array([self.pos / self.max_distance, self.steps / self.max_steps, self.rssi1, self.snir1,
                        self.timestamp1, self.rssi2, self.snir2, self.timestamp2], dtype=np.float32), reward, done, False, {}

    def render(self):
        # Map the position [0, 1] to the x-coordinate along the line [50, 550]
        x = int(self.pos)
        y = 5
        # Create a new black image
        offset = int( (self.width - self.max_distance)/2 )
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the line and moving point
        cv2.line(frame,pt1=(offset, y), pt2=(offset + self.max_distance,y), color=self.line_color)
        cv2.rectangle(frame,pt1= (offset + x-2, y-2), pt2= (offset + x+2, y+2), color=self.point_color)

        # Draw nodes
        cv2.rectangle(frame,pt1= (offset + self.node_1x.pos-2, y-2), pt2= (offset + self.node_1x.pos+2, y+2), color=self.point_color)
        cv2.rectangle(frame,pt1= (offset + self.node_2x.pos-2, y-2), pt2= (offset + self.node_2x.pos+2, y+2), color=self.point_color)
        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        # Draw score
        cv2.putText(enlarged_image, "Total score: " + str(self.total_reward), (250,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

        cv2.imshow(self.window_name, enlarged_image)
        cv2.waitKey(30)  # Wait a short time to create the animation effect

    def close(self):
        cv2.destroyAllWindows()


class SignalModel:
    def __init__(self, rssi_ref=-30, path_loss_exponent=2.7, noise_floor=-100,
                 rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30):
        self.rssi_ref = rssi_ref
        self.path_loss_exponent = path_loss_exponent
        self.noise_floor = noise_floor
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max
        self.snir_min = snir_min
        self.snir_max = snir_max

    def generate_rssi(self, distance):
        if distance <= 0:
            raise ValueError("Distance must be greater than 0.")
        rssi = self.rssi_ref - 10 * self.path_loss_exponent * np.log10(distance)
        # Scale RSSI between 0 and 1
        rssi_scaled = (rssi - self.rssi_min) / (self.rssi_max - self.rssi_min)
        return np.clip(rssi_scaled, 0, 1)  # Ensure it’s within [0, 1]

    def generate_snir(self, distance):
        rssi_linear = 10 ** ((self.rssi_ref - 10 * self.path_loss_exponent * np.log10(distance)) / 10)
        noise_linear = 10 ** (self.noise_floor / 10)
        snir = 10 * np.log10(rssi_linear / noise_linear)
        # Scale SNIR between 0 and 1, inverted scale
        snir_scaled = 1 - (snir - self.snir_min) / (self.snir_max - self.snir_min)
        return np.clip(snir_scaled, 0, 1)  # Ensure it’s within [0, 1]

class node():
    def __init__(self, pos=10, time_to_first_packet=10, send_interval=10, send_std=2):
        self.pos = pos
        self.last_packet_time = 0
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_interval = send_interval
        self.send_std = send_std  # standard deviation
        self.lower_bound_send_time = send_interval / 2
        self.upper_bound_send_time = send_interval * 2

        self.max_transmission_distance = 60
        self.transmission_model = SignalModel(rssi_ref=-30, path_loss_exponent=2.7, noise_floor=-100,
                                              rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30)

    def reset(self):
        self.last_packet_time = 0
        self.time_of_next_packet = self.time_to_first_packet

    def generate_next_interval(self):
        # Generate a truncated normal value for the next time interval
        # a and b are calculated to truncate around the mean interval with some range
        a, b = (self.lower_bound_send_time - self.send_interval) / self.send_std, (
                self.upper_bound_send_time - self.send_interval) / self.send_std
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_interval)
        return interval

    def generate_RSSI(self, distance):
        rssi_scaled = self.transmission_model.generate_rssi(distance)

    def generate_SNIR(self, distance):
        snir_scaled = self.transmission_model.generate_snir(distance)

    def transmission(self, gpos):
        ploss_scale = 100
        distance = abs(self.pos - gpos)
        if distance < self.max_transmission_distance:
            ploss_probability = 1 - np.exp(- distance / ploss_scale)
            ploss_choice = np.random.rand() < ploss_probability
            if ploss_choice:
                rssi_scaled = self.transmission_model.generate_rssi(distance)
                snir_scaled = self.transmission_model.generate_snir(distance)
                return True, rssi_scaled, snir_scaled
        return False, 0, 0

    def send(self, time, gpos):
        # Decides whether a packet should be send and if it gets lost
        # Pobability of success is based of last time send and distance
        if time > self.time_of_next_packet:
            self.last_packet_time = time
            self.time_of_next_packet = time + self.generate_next_interval()
            received, rssi, snir = self.transmission(gpos)
            if received:
                return received, rssi, snir
        return False, 0, 0
        

class RewardPlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardPlottingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_counts = []

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:  # Check if an episode has finished
            reward_sum = self.locals.get("infos")[0]["episode"]["r"]
            self.episode_rewards.append(reward_sum)
            self.episode_counts.append(len(self.episode_rewards))
            
            plt.plot(self.episode_counts, self.episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.pause(0.2)  # Update plot in real-time
        return True
    def _on_training_end(self) -> None:
            plt.plot(self.episode_counts, self.episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.show()
            pass
