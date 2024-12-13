import json

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import cv2
from enum import Enum


# Define a custom FrameSkip wrapper
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame."""
        super(FrameSkip, self).__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and return the last observation."""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


class SimpleBaseEnv(gym.Env):
    def __init__(self, render_mode="none", do_logging=False, log_file="env_log.json"):
        super(SimpleBaseEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(3, start=0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = (60 * 60 * 12.0)  # Maximum steps per episode
        # Observation_space = pos, current_pos1, current_pos2, pos1, rssi1, snir1, timestamp1, pos2,rssi2, snir2, timestamp2
        # Pos is the position when packet was received, timestamp is the time SINCE packet received
        self.observation_space = spaces.Box(low=np.array([0, -1, -1, 0, 0]), high=np.array([1, 1, 1, 1, 1]),
                                            dtype=np.float32)
        # Environment state
        self.visited_pos = dict()
        self.received1 = 0
        self.received2 = 0
        self.last_packet = 0
        self.pos_reward_max = 0.001
        self.pos_reward_min = 0
        self.pos_penalty_max = 3
        self.pos_penalty_min = 0
        self.miss_penalty_max = 10
        self.miss_penalty_min = 5
        self.packet_reward_max = 10
        speed = 20  # meter per second
        max_distance = 3000  # meters
        self.max_distance = int(max_distance / speed)  # scaled by speed
        self.pos = self.max_distance / 2
        self.target = 5  # The target value we want to reach
        self.steps = 0
        pos1 = 0
        pos2 = self.max_distance
        self.node1 = node(pos1, time_to_first_packet=5000, send_interval=5000)
        self.node2 = node(pos2, time_to_first_packet=7500, send_interval=5000)
        self.p_received1 = pos1
        self.p_received2 = pos2

        self.p1_stamp = -1
        self.p2_stamp = -1
        self.total_reward = 0
        self.pos = int(self.max_distance / 2)  # Start in the middle
        self.rssi1 = 0
        self.rssi2 = 0
        self.snir1 = 0
        self.snir2 = 0
        self.timestamp1 = 0
        self.timestamp2 = 0
        self.p_dist1 = 0  # distance to node when received
        self.p_dist2 = 0
        self.total_misses = 0
        self.total_received = 0
        self.width, self.height = 200, 20  # Size of the window
        self.point_radius = 1
        self.point_color = (0, 0, 255)  # Red color
        self.line_color = (255, 0, 0)  # Blue color

        self.do_logging = do_logging
        self.log_file = log_file
        self.log_data = []  # Store logs before writing to the file

        if render_mode == "cv2":
            self.window_name = "RL Animation"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def reset(self, seed=None, options=None):
        # Reset the.pos and steps counter
        self.visited_pos = dict()

        self.last_packet = 0
        self.total_misses = 0
        self.pos = int(self.max_distance / 2)
        self.node1.reset()
        self.node2.reset()
        self.steps = 0
        self.total_reward = 0
        self.rssi1 = 0
        self.rssi2 = 0
        self.snir1 = 0
        self.snir2 = 0
        self.timestamp1 = self.max_steps
        self.timestamp2 = self.max_steps
        self.total_received = 0

        self.p1_stamp = -1
        self.p2_stamp = -1
        state = [self.pos / self.max_distance, self.p1_stamp, self.p2_stamp,
                 self.timestamp1 / self.max_steps,
                 self.timestamp2 / self.max_steps]
        return np.array(state, dtype=np.float32), {}

    def get_pos_reward(self, pos1, pos2, time):
        time = (time - self.steps)  # time to packet
        scaled_time = (time / self.max_steps)
        distance = abs(pos1 - pos2)
        scaled_distance = 1 - distance / self.max_distance
        scaled_distance_time = scaled_distance * scaled_time
        # Return reward based on scaled distance between a min and max reward
        reward = self.pos_reward_max - scaled_distance_time * (self.pos_reward_max - self.pos_reward_min)

        # Ensure reward is within bounds in case of rounding errors
        reward = max(self.pos_reward_min, min(self.pos_reward_max, reward))
        return reward

    def get_pos_penalty(self, pos1, pos2, time):
        time = (time - self.steps)  # time to packet
        scaled_time = 1 - (time / self.max_steps)
        distance = abs(pos1 - pos2)
        scaled_distance = distance / self.max_distance
        scaled_distance_time = scaled_distance * scaled_time
        # Return reward based on scaled distance between a min and max reward
        penalty = self.pos_reward_min + scaled_distance_time * (self.pos_penalty_max - self.pos_penalty_min)

        # Ensure reward is within bounds in case of rounding errors
        penalty = min(self.pos_penalty_max, max(self.pos_penalty_min, penalty))
        return -penalty

    def get_miss_penalty(self, pos1, pos2):
        distance = abs(pos1 - pos2)
        scaled_distance = distance / self.max_distance
        # Return reward based on scaled distance between a min and max reward
        penalty = self.miss_penalty_min + scaled_distance * (self.miss_penalty_max - self.miss_penalty_min)

        # Ensure reward is within bounds in case of rounding errors
        penalty = min(self.miss_penalty_max, max(self.miss_penalty_min, penalty))
        if distance > self.max_distance / 2:
            penalty * 2
        return -penalty

    def get_explore_reward(self, pos, time):
        base_reward = 0.001
        if pos not in self.visited_pos.keys():
            self.visited_pos[pos] = time
            return base_reward

        base_reward = base_reward * (time - self.visited_pos[pos]) / self.max_steps
        self.visited_pos[pos] = time

        return base_reward

    def step(self, action):
        if self.render_mode == "cv2":
            self.render()
        reward = 0
        self.steps += 1

        if action == 0:
            if self.pos > 0:
                self.pos -= 1  # Action left
        elif action == 1:
            if self.pos < self.max_distance:
                self.pos += 1  # Action right
        elif action == 2:
            self.pos = self.pos  # stand still

        self.received1, rssi1, snir1 = self.node1.send(self.steps, self.pos)
        self.received2, rssi2, snir2 = self.node2.send(self.steps, self.pos)
        self.timestamp1 = min(self.max_steps, self.timestamp1 + 1)
        self.timestamp2 = min(self.max_steps, self.timestamp2 + 1)
        if self.received1 == PACKET_STATUS.RECEIVED:
            reward = self.packet_reward_max
            if self.last_packet == 2:
                reward += self.packet_reward_max
            self.p_received1 = self.pos
            self.rssi1 = rssi1
            self.snir1 = snir1
            self.timestamp1 = 0
            self.last_packet = 1
            self.total_received += 1
            self.p_dist1 = abs(self.pos - self.node1.pos)
            self.p1_stamp = self.pos / self.max_distance
        elif self.received2 == PACKET_STATUS.RECEIVED:
            reward = self.packet_reward_max
            if self.last_packet == 1:
                reward += self.packet_reward_max
            self.p_received2 = self.pos
            self.rssi2 = rssi2
            self.snir2 = snir2
            self.timestamp2 = 0
            self.last_packet = 2
            self.total_received += 1
            self.p_dist2 = abs(self.pos - self.node2.pos)
            self.p2_stamp = self.pos / self.max_distance
        elif self.received1 == PACKET_STATUS.LOST:
            self.total_misses += 1
            reward = self.get_miss_penalty(self.pos, self.node1.pos)
        elif self.received2 == PACKET_STATUS.LOST:
            self.total_misses += 1
            reward = self.get_miss_penalty(self.pos, self.node2.pos)

        elif self.timestamp1 > self.timestamp2:
            reward = self.get_pos_reward(self.pos, self.p1_stamp, self.timestamp1)
        elif self.timestamp1 <= self.timestamp2:
            reward = self.get_pos_reward(self.pos, self.p2_stamp, self.timestamp2)

        reward += self.get_explore_reward(self.pos, self.steps)

        done = self.steps >= self.max_steps or self.total_misses >= 20
        self.total_reward += reward

        state = [self.pos / self.max_distance, self.p1_stamp, self.p2_stamp,
                 self.timestamp1 / self.max_steps,
                 self.timestamp2 / self.max_steps]
        info = {'total_received': self.total_received,
                'total_misses': self.total_misses}

        # Add logging for each step
        if self.do_logging:
            transmission_occurred = self.received1 == PACKET_STATUS.RECEIVED or \
                                    self.received1 == PACKET_STATUS.LOST or \
                                    self.received2 == PACKET_STATUS.RECEIVED or \
                                    self.received2 == PACKET_STATUS.LOST

            self.log_step(gw_pos_x=self.pos, step_time=self.steps, packets_received=self.total_received,
                          packets_sent=self.total_received + self.total_misses,
                          transmission_occurred=transmission_occurred)
            if done:
                # Write to the JSON file at the end of simulation.
                with open(self.log_file, 'w') as file:
                    json.dump(self.log_data, file, indent=4)

        return np.array(state, dtype=np.float32), reward, done, False, info

    def render(self):
        # Map the position [0, 1] to the x-coordinate along the line [50, 550]
        x = int(self.pos)
        y = 5
        # Create a new black image
        offset = int((self.width - self.max_distance) / 2)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the line and moving point
        cv2.line(frame, pt1=(offset, y), pt2=(offset + self.max_distance, y), color=self.line_color)
        cv2.line(frame, pt1=(int(self.width / 2), y + 1), pt2=(int(self.width / 2), y - 1), color=self.line_color)
        cv2.rectangle(frame, pt1=(offset + x - 2, y - 2), pt2=(offset + x + 2, y + 2), color=self.point_color)

        # Draw nodes
        if self.received1 != PACKET_STATUS.NOT_SENT:
            cv2.rectangle(frame, pt1=(offset + self.node1.pos - 2, y - 2), pt2=(offset + self.node1.pos + 2, y + 2),
                          color=(0, 128, 0))
        else:
            cv2.rectangle(frame, pt1=(offset + self.node1.pos - 2, y - 2), pt2=(offset + self.node1.pos + 2, y + 2),
                          color=self.point_color)

        if self.received2 != PACKET_STATUS.NOT_SENT:
            cv2.rectangle(frame, pt1=(offset + self.node2.pos - 2, y - 2), pt2=(offset + self.node2.pos + 2, y + 2),
                          color=(0, 128, 0))
        else:
            cv2.rectangle(frame, pt1=(offset + self.node2.pos - 2, y - 2), pt2=(offset + self.node2.pos + 2, y + 2),
                          color=self.point_color)

        # cv2.rectangle(frame,pt1= (offset + self.node1.pos-2, y-2), pt2= (offset + self.node1.pos+2, y+2), color=self.point_color)
        # cv2.rectangle(frame,pt1= (offset + self.node2.pos-2, y-2), pt2= (offset + self.node2.pos+2, y+2), color=self.point_color)
        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        # Draw score
        cv2.putText(enlarged_image,
                    "Total received: " + str(self.total_received) + " | Total misses: " + str(self.total_misses),
                    (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow(self.window_name, enlarged_image)
        cv2.waitKey(5)  # Wait a short time to create the animation effect

    def log_step(self, gw_pos_x, step_time, packets_received, packets_sent, transmission_occurred):
        """
        Logs a single step's data into the log buffer.
        """
        log_entry = {
            "gw_pos_x": gw_pos_x,
            "step_time": step_time,
            "packets_received": packets_received,
            "packets_sent": packets_sent,
            "transmission_occurred": transmission_occurred
        }
        self.log_data.append(log_entry)

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
        if distance < 0:
            raise ValueError("Distance must be greater than 0.")
        distance = max(distance, 0.00001)  # avoid log of 0
        rssi = self.rssi_ref - 10 * self.path_loss_exponent * np.log10(distance)
        # Scale RSSI between 0 and 1
        rssi_scaled = (rssi - self.rssi_min) / (self.rssi_max - self.rssi_min)
        return np.clip(rssi_scaled, 0, 1)  # Ensure it’s within [0, 1]

    def generate_snir(self, distance):
        if distance < 0:
            raise ValueError("Distance must be greater than 0.")
        distance = max(distance, 0.00001)  # avoid log of 0
        rssi_linear = 10 ** ((self.rssi_ref - 10 * self.path_loss_exponent * np.log10(distance)) / 10)
        noise_linear = 10 ** (self.noise_floor / 10)
        snir = 10 * np.log10(rssi_linear / noise_linear)
        # Scale SNIR between 0 and 1, inverted scale
        snir_scaled = 1 - (snir - self.snir_min) / (self.snir_max - self.snir_min)
        return np.clip(snir_scaled, 0, 1)  # Ensure it’s within [0, 1]


class PACKET_STATUS(Enum):
    RECEIVED = 1
    LOST = 2
    NOT_SENT = 3


class node:
    def __init__(self, pos=10, time_to_first_packet=10, send_interval=10, send_std=2):
        self.pos = pos
        self.last_packet_time = 0
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_interval = send_interval
        self.send_std = send_std  # standard deviation
        self.lower_bound_send_time = send_interval - (send_interval / 2)
        self.upper_bound_send_time = send_interval + (send_interval / 2)

        self.max_transmission_distance = 70
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
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_std)
        return interval

    def generate_RSSI(self, distance):
        rssi_scaled = self.transmission_model.generate_rssi(distance)

    def generate_SNIR(self, distance):
        snir_scaled = self.transmission_model.generate_snir(distance)

    def transmission(self, gpos):
        ploss_scale = 3000
        distance = abs(self.pos - gpos)
        if distance < self.max_transmission_distance:
            ploss_probability = np.exp(- distance / ploss_scale)

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
            # f"time of next packet: {self.time_of_next_packet}" )
            is_received, rssi, snir = self.transmission(gpos)
            if is_received:
                # print(f"packet is_received ")
                return PACKET_STATUS.RECEIVED, rssi, snir
            else:
                return PACKET_STATUS.LOST, 0, 0
        return PACKET_STATUS.NOT_SENT, 0, 0


class RewardPlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardPlottingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_reward = 0
        # Initialize the plot
        plt.ion()  # Interactive mode for real-time plotting
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")

    def _on_step(self) -> bool:
        self.episode_reward += sum(self.locals["rewards"])
        if self.locals["dones"][0]:
            # Append total episode reward
            self.episode_rewards.append(self.episode_reward)
            # Reset the episode reward counter
            # Reset episode reward
            self.episode_reward = 0

            # Update the plot data
            self.line.set_xdata(range(len(self.episode_rewards)))
            self.line.set_ydata(self.episode_rewards)
            self.ax.relim()
            self.ax.autoscale_view()

            plt.draw()
            plt.pause(0.005)  # Small pause for updating the plot in real-time
        return True

    def _on_training_end(self) -> None:
        print("Training plot done.")
        plt.ioff()  # Turn off interactive mode
        plt.show()

        pass
