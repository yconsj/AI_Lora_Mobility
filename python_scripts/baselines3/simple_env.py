from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import cv2
from enum import Enum

class SimpleBaseEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(SimpleBaseEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(2, start=0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = 50000  # Maximum steps per episode
        # Observation_space = current_pos, pos1, rssi1, snir1, timestamp1, pos2,rssi2, snir2, timestamp2
        # Pos is the position when packet was recieved, timestamp is the time SINCE packet recieved
        self.observation_space = spaces.Box(low=np.array([0, 0 ,0, 0, 0, 0, 0, 0,0]), high=np.array([1, 1,1, 1,1,1,1,1,1]), dtype=np.float32)
        # Environment state
        self.recieved1 = 0
        self.recieved2 = 0
        self.last_packet = 0
        speed = 20  # meter per second
        max_distance = 3000 # meters
        self.max_distance = int(max_distance / speed)  # scaled by speed
        self.pos = self.max_distance / 2
        self.target = 5  # The target value we want to reach
        self.steps = 0
        pos1 = 0
        pos2 = self.max_distance
        self.node1 = node(pos1, time_to_first_packet=100, send_interval=200)
        self.node2 = node(pos2, time_to_first_packet=200, send_interval=200)
        self.p_recieved1 = pos1
        self.p_recieved2 = pos2
        self.total_reward = 0
        self.pos = int(self.max_distance / 2)  # Start in the middle
        self.rssi1 = 0
        self.rssi2 = 0
        self.snir1 = 0
        self.snir2 = 0
        self.timestamp1 = 0
        self.timestamp2 = 0
        self.total_misses = 0
        self.width, self.height = 200, 20  # Size of the window
        self.point_radius = 1
        self.point_color = (0, 0, 255)  # Red color
        self.line_color = (255, 0, 0)  # Blue color
        self.window_name = "RL Animation"

        self.visited_pos = []
    def reset(self, seed=None, options=None):
        # Reset the.pos and steps counter
        if self.render_mode == "cv2":
            self.render()
        self.visited_pos = []
        self.last_packet = 0
        self.total_misses = 0
        self.pos = int(self.max_distance/2)
        self.node1.reset()
        self.node2.reset()
        self.steps = 0
        self.total_reward = 0
        self.rssi1 = 0
        self.rssi2 = 0
        self.snir1 = 0
        self.snir2 = 0
        self.timestamp1 = 0
        self.timestamp2 = 0


        state = [ self.pos / self.max_distance,
                self.node1.pos / self.max_distance, self.rssi1, self.snir1, self.timestamp1 / self.max_steps,
                self.node2.pos / self.max_distance, self.rssi2, self.snir2, self.timestamp2 / self.max_steps]
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        reward = 0
        if self.pos not in self.visited_pos:
            self.visited_pos.append(self.pos)
            # reward +=1
        
        if action == 0:
            if self.pos > 0:
                self.pos -= 1  # Action -1
        elif action == 1:
            if self.pos < self.max_distance:
                self.pos += 1  # Action +1
        # Update step count and check if episode is done
        self.steps += 1
        self.recieved1, rssi1, snir1 = self.node1.send(self.steps, self.pos)
        self.recieved2, rssi2, snir2 = self.node2.send(self.steps, self.pos)

        self.timestamp1+=1
        self.timestamp2+=1
        if self.recieved1 == PACKET_STATUS.RECIEVED:
            if self.last_packet == 1:
                reward+= 5*self.rssi1
            else:
                reward += 10*self.rssi1
            self.p_recieved1 = self.pos
            self.rssi1 = rssi1
            self.snir1 = snir1
            self.timestamp1 = 0
            self.last_packet = 1
        if self.recieved2 == PACKET_STATUS.RECIEVED:
            if self.last_packet == 2:
                reward+= 5*self.rssi2
            else:
                reward += 10*self.rssi2
            self.p_recieved2 = self.pos
            self.rssi2 = rssi2
            self.snir2 = snir2
            self.timestamp2 = 0
            self.last_packet = 2
        if self.recieved1 == PACKET_STATUS.LOST:
            self.total_misses += 1
            reward -= 2
        if self.recieved2 == PACKET_STATUS.LOST:
            self.total_misses += 1
            reward -= 2
        done = self.steps >= self.max_steps or self.total_misses >= 10
        reward = round(reward, 1)
        self.total_reward += reward

        state = [ self.pos / self.max_distance,
                self.p_recieved1 / self.max_distance, self.rssi1, self.snir1, self.timestamp1 / self.max_steps,
                self.p_recieved2 / self.max_distance, self.rssi2, self.snir2, self.timestamp2 / self.max_steps]

        return np.array(state, dtype=np.float32), reward, done, False, {}

    def render(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Map the position [0, 1] to the x-coordinate along the line [50, 550]
        x = int(self.pos)
        y = 5
        # Create a new black image
        offset = int( (self.width - self.max_distance)/2 )
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the line and moving point
        cv2.line(frame,pt1=(offset, y), pt2=(offset + self.max_distance,y), color=self.line_color)
        cv2.line(frame,pt1=(int(self.width / 2), y+1), pt2=(int(self.width / 2),y-1), color=self.line_color)
        cv2.rectangle(frame,pt1= (offset + x-2, y-2), pt2= (offset + x+2, y+2), color=self.point_color)

        # Draw nodes
        if self.recieved1 != PACKET_STATUS.NOT_SENT:
            cv2.rectangle(frame,pt1= (offset + self.node1.pos-2, y-2), pt2= (offset + self.node1.pos+2, y+2), color=(0,128,0))
        else:
            cv2.rectangle(frame,pt1= (offset + self.node1.pos-2, y-2), pt2= (offset + self.node1.pos+2, y+2), color=self.point_color)

        if self.recieved2 != PACKET_STATUS.NOT_SENT:
            cv2.rectangle(frame,pt1= (offset + self.node2.pos-2, y-2), pt2= (offset + self.node2.pos+2, y+2), color=(0,128,0))
        else:
            cv2.rectangle(frame,pt1= (offset + self.node2.pos-2, y-2), pt2= (offset + self.node2.pos+2, y+2), color=self.point_color)

        # cv2.rectangle(frame,pt1= (offset + self.node1.pos-2, y-2), pt2= (offset + self.node1.pos+2, y+2), color=self.point_color)
        # cv2.rectangle(frame,pt1= (offset + self.node2.pos-2, y-2), pt2= (offset + self.node2.pos+2, y+2), color=self.point_color)
        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        # Draw score
        cv2.putText(enlarged_image, "Total score: " + str(self.total_reward) + "| Total misses: " + str(self.total_misses), (250,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

        cv2.imshow(self.window_name, enlarged_image)
        cv2.waitKey(5)  # Wait a short time to create the animation effect

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
    RECIEVED = 1
    LOST = 2
    NOT_SENT = 3
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

        self.max_transmission_distance = 90
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
        ploss_scale = 150
        distance = abs(self.pos - gpos)
        if distance < self.max_transmission_distance:
            ploss_probability = np.exp(distance / ploss_scale)
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
            #f"time of next packet: {self.time_of_next_packet}" )
            is_received, rssi, snir = self.transmission(gpos)
            if is_received:
                #print(f"packet is_received ")
                return PACKET_STATUS.RECIEVED, rssi, snir
            else:
                return PACKET_STATUS.LOST, 0 ,0
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
