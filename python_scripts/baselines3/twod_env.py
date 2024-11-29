from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import cv2
from enum import Enum
import math
import random
from scipy.optimize import least_squares
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
        return obs, total_reward, done, trunc,info
class PacketReference():
    def __init__(self, max_pos = (150,150), pos = (-1,-1), rssi = -1, snir= -1):
        self.pos = pos
        self.rssi = rssi
        self.snir = snir
        self.max_pos = max_pos

    def get(self):
        if self.pos == (-1, -1):
            return (self.pos[0], self.pos[1], self.rssi)

        return (self.pos[0] / self.max_pos[0], self.pos[1] / self.max_pos[1], self.rssi)
class TwoDEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(TwoDEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(5, start=0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = 10000  # Maximum steps per episode
        # Observation_space = 
        #                     prev_actin (x,y),             3
        #                     (x1,x2), rssi, snir * 3       12
        #                     (x1,x2), rssi, snir * 3       12
        #                     elapsed_time1, elapsed_time2  2
        #                                                   28
        self.observation_space = spaces.Box(low=np.array(
            [0]*3 + [-1]*18 + [0]*2), high=np.array(
            [1]*23), dtype=np.float32)
        # Environment state
        self.visited_pos = dict()
        self.last_packet = 0
        self.pos_reward_max = 0.005                       
        self.pos_reward_min = 0
        self.pos_penalty_max = 3
        self.pos_penalty_min = 0
        self.miss_penalty_max = 10
        self.miss_penalty_min = 5
        self.packet_reward_max = 5
        
        speed = 20  # meter per second
        max_distance = 3000 # meter
        self.max_distance_x = int(max_distance / speed)  # scaled by speed
        self.max_distance_y = int(max_distance / speed)
        self.max_cross_distance = math.dist((0,0), (self.max_distance_x, self.max_distance_y))
        self.pos = ( int(self.max_distance_x / 2), int(self.max_distance_y / 2))
        self.prev_pos = self.pos
        self.target = 5  # The target value we want to reach
        self.steps = 0
        pos1 = (25, 25)
        pos2 = (self.max_distance_x-25, self.max_distance_y -25)
        self.node1 = node(pos1, time_to_first_packet=50, send_interval=300)
        self.node2 = node(pos2, time_to_first_packet=125, send_interval=300)
        self.prefs1 = (PacketReference(), PacketReference(), PacketReference())
        self.prefs2 = (PacketReference(), PacketReference(), PacketReference())
        self.elapsed_time1 = 0
        self.elapsed_time2 = 0
        self.initial_guess1 = self.pos
        self.initial_guess2 = self.pos
        self.total_reward = 0
        self.total_misses = 0
        self.total_received = 0
        self.width, self.height = 175, 175  # Size of the window
        self.point_radius = 1
        self.point_color = (0, 0, 255)  # Red color
        self.line_color = (255, 0, 0)  # Blue color
        self.prev_action = 0
        if render_mode == "cv2":
            self.window_name = "RL Animation"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    def reset(self, seed=None, options=None):
        # Reset the.pos and steps counter
        self.pev_action = 0
        self.prev_pos = self.pos
        self.visited_pos = dict()
        self.last_packet = 0
        self.total_misses = 0
        self.pos = (int(self.max_distance_x / 2), int(self.max_distance_y / 2))
        self.initial_guess1 = self.pos
        self.initial_guess2 = self.pos
        self.node1.reset()
        self.node2.reset()
        x1 = random.randint(0,150)
        y1 = random.randint(0,150)
        self.node1.pos = (x1,y1)
        while True:
            x2 = random.randint(0,150)
            y2 = random.randint(0,150)
            if math.dist((x2,y2), self.node1.pos) >= 75:
                self.node2.pos= (x2,y2)
                break
        self.steps = 0
        self.total_reward = 0
        self.total_received = 0
        self.prefs1 = (PacketReference(), PacketReference(), PacketReference())
        self.prefs2 = (PacketReference(), PacketReference(), PacketReference())
        self.elapsed_time1 = 0
        self.elapsed_time2 = 0
        state = [ self.prev_action / 4,
            self.pos[0] / self.max_distance_x, self.pos[1] / self.max_distance_y, 
            *self.prefs1[0].get(), *self.prefs1[1].get(),*self.prefs1[2].get(),
            *self.prefs2[0].get(),*self.prefs2[1].get(),*self.prefs2[2].get(),
            self.elapsed_time1 / self.max_steps,
            self.elapsed_time2 / self.max_steps
        ]
        return np.array(state, dtype=np.float32), {}


    def get_pos_reward(self, pos1, pos2, time):
        scaled_time = (time / self.max_steps)
        distance = math.dist(pos1, pos2)
        scaled_distance = 1- distance / self.max_cross_distance
        scaled_distance_time = scaled_distance * scaled_time
        # Return reward based on scaled distance between a min and max reward
        reward = self.pos_reward_max - scaled_distance_time * (self.pos_reward_max - self.pos_reward_min)

        # Ensure reward is within bounds in case of rounding errors
        reward = max(self.pos_reward_min, min(self.pos_reward_max, reward))
        return reward
    
    def is_new_best_pref(self, pref, p):
        (pref1, pref2, pref3) = pref
        if p.pos == pref1.pos or p.pos == pref2.pos or p.pos == pref3.pos:
            return False
        if p.rssi > pref1.rssi or  p.rssi > pref2.rssi or p.rssi > pref3.rssi:
            return True
        return False
    def insert_best_pref(self, pref, p):
        (pref1, pref2, pref3) = pref
        if p.rssi > pref1.rssi:
            pref3 = pref2
            pref2 = pref1
            pref1 = p
            return (pref1, pref2, pref3)
        if p.rssi > pref2.rssi:
            pref3 = pref2
            pref2 = p
            return (pref1, pref2, pref3)
        if p.rssi > pref3.rssi:
            pref3 = p
            return (pref1, pref2, pref3)
    def get_miss_penalty(self, pos1, pos2):
        distance = math.dist(pos1, pos2)
        scaled_distance = distance / self.max_cross_distance
        # Return reward based on scaled distance between a min and max reward
        penalty = self.miss_penalty_min + scaled_distance * (self.miss_penalty_max - self.miss_penalty_min)

        # Ensure reward is within bounds in case of rounding errors
        penalty = min(self.miss_penalty_max, max(self.miss_penalty_min, penalty))
        return -penalty
    def get_explore_reward(self, pos, time):
        base_reward  = 0.01
        if pos not in self.visited_pos.keys():
            self.visited_pos[pos] = time
            multiplier = abs(pos[0] - int(self.max_distance_x / 2)) * abs(pos[1] - int(self.max_distance_y / 2)) / (self.max_cross_distance * 2)
            return base_reward * multiplier
        
        base_reward = base_reward * (time - self.visited_pos[pos]) / self.max_steps
        
        self.visited_pos[pos] = time

        return 0
    def trilateration_residuals(self, params, positions, distances):
        x, y = params  # Unknown position (x, y)
        residuals = []
        
        for (px, py), d in zip(positions, distances):
            calculated_distance = np.sqrt((x - px)**2 + (y - py)**2)
            residuals.append(calculated_distance - d)
        
        return residuals
    def trilateration(self, prefs, initial_guess):
        (pref1,pref2,pref3) = prefs
        positions = [pref1.pos, pref2.pos, pref3.pos]
        distances = [self.node1.inverse_RSSI(pref1.rssi),self.node1.inverse_RSSI(pref2.rssi),self.node1.inverse_RSSI(pref3.rssi)]
        # Initial guess for the unknown position (x, y)
       
        
        # Perform least squares optimization
        result = least_squares(self.trilateration_residuals, initial_guess, args=(positions, distances))
        
        # Return the optimized position
        return int(result.x[0]),int(result.x[1])



    def step(self, action):
        if self.render_mode == "cv2":
            self.render()
        reward = 0
        self.steps += 1

        if action == 0: # stand still
            #nothing
            pass
        elif action == 1: # left
            if self.pos[0] > 0: 
                self.pos = (self.pos[0]-1, self.pos[1])  
        elif action == 2: # right
            if self.pos[0] < self.max_distance_x:
                self.pos = (self.pos[0]+1, self.pos[1]) 
        elif action == 3: # up
            if self.pos[1] < self.max_distance_y:
                self.pos = (self.pos[0], self.pos[1]+1)   
        elif action == 4: # down
            if self.pos[1]  > 0 :
                self.pos = (self.pos[0], self.pos[1]-1) 

        received1, rssi1, snir1 = self.node1.send(self.steps, self.pos)
        received2, rssi2, snir2 = self.node2.send(self.steps, self.pos)
        self.elapsed_time1 = min(self.max_steps,self.elapsed_time1+1)
        self.elapsed_time2 = min(self.max_steps,self.elapsed_time2+1)
        p1 = PacketReference(pos= self.pos,rssi= rssi1,snir=snir1)
        p2 = PacketReference(pos= self.pos, rssi=rssi2,snir=snir2)
        if received1 == PACKET_STATUS.RECEIVED:
            if (self.prefs1[0].rssi != -1 and self.prefs1[1].rssi != -1 and self.prefs1[2].rssi != -1) or True:
                reward = self.packet_reward_max
                if self.last_packet == 2:
                    reward += self.packet_reward_max
                self.last_packet = 1
                self.total_received += 1
                self.elapsed_time1 = 0
            # TODO: insert packet if best
            if self.is_new_best_pref(self.prefs1, p1):
                self.prefs1 = self.insert_best_pref(self.prefs1,p1)
        elif received2 == PACKET_STATUS.RECEIVED:
            if  (self.prefs2[0].rssi != -1 and self.prefs2[1].rssi != -1 and self.prefs2[2].rssi != -1) or True:
                reward = self.packet_reward_max
                if self.last_packet == 1:
                    reward += self.packet_reward_max
                self.last_packet = 2
                self.total_received += 1
                self.elapsed_time2 = 0
            # TODO: insert packet if 
            if self.is_new_best_pref(self.prefs2, p2):
                self.prefs2 = self.insert_best_pref(self.prefs2,p2)

        elif received1 == PACKET_STATUS.LOST : # and (self.prefs1[0].rssi == -1 or self.prefs1[1].rssi == -1 or self.prefs1[2].rssi == -1):
            self.total_misses += 1
            reward = self.get_miss_penalty(self.pos, self.node1.pos)
        elif received2 == PACKET_STATUS.LOST : #and (self.prefs2[0].rssi == -1 or self.prefs2[1].rssi == -1 or self.prefs2[2].rssi == -1):
            self.total_misses += 1
            reward = self.get_miss_penalty(self.pos, self.node2.pos)

        if self.prefs1[0].rssi != -1 and self.prefs1[1].rssi != -1 and self.prefs1[2].rssi != -1:
                aprox_pos = self.trilateration(self.prefs1, self.initial_guess1)
                self.initial_guess1 = aprox_pos
        if self.prefs2[0].rssi != -1 and self.prefs2[1].rssi != -1 and self.prefs2[2].rssi != -1:
                aprox_pos = self.trilateration(self.prefs2, self.initial_guess2)
                self.initial_guess2 = aprox_pos
                
        if self.elapsed_time1 > self.elapsed_time2: 
            #reward += self.get_pos_reward(self.pos, self.node1.pos, self.elapsed_time1)
                reward += self.get_pos_reward(self.pos,self.initial_guess1, self.elapsed_time1)
        elif self.elapsed_time1 <= self.elapsed_time2:
            #reward += self.get_pos_reward(self.pos, self.node2.pos, self.elapsed_time2)
                reward += self.get_pos_reward(self.pos,self.initial_guess2, self.elapsed_time2)


        reward += self.get_explore_reward(self.pos, self.steps)

        done = self.steps >= self.max_steps or self.total_misses >= 20
        self.total_reward += reward
        state = [self.prev_action / 4,
            self.pos[0] / self.max_distance_x, self.pos[1] / self.max_distance_y, 
            *self.prefs1[0].get(), *self.prefs1[1].get(), *self.prefs1[2].get(),
            *self.prefs2[0].get(), *self.prefs2[1].get(), *self.prefs2[2].get(),
            self.elapsed_time1 / self.max_steps,
            self.elapsed_time2 / self.max_steps
        ]
        self.prev_action = action
        info = {'total_received': self.total_received,
                'total_misses': self.total_misses}
        return np.array(state, dtype=np.float32), reward, done, False, info
    

    def render(self):
        
        # Map the position [0, 1] to the x-coordinate along the line [50, 550]
        x = int(self.pos[0])
        y = int(self.pos[1])
        # Create a new black image
        offset_x = int( (self.width - self.max_distance_x)/2 )
        offset_y = int( (self.height - self.max_distance_y)/2 )

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the line and moving point
        cv2.line(frame,pt1=(offset_x, offset_y + int(self.max_distance_y / 2)), pt2=(offset_x + self.max_distance_x, offset_y+ int(self.max_distance_y / 2)), color=self.line_color)
        cv2.line(frame,pt1=(offset_x + int(self.max_distance_x / 2), offset_y), pt2=( offset_x + int(self.max_distance_x / 2), self.max_distance_y + offset_y), color=self.line_color)
        cv2.rectangle(frame,pt1= (offset_x + x-2, offset_y + y-2), pt2= (offset_x + x+2, offset_y + y+2), color=self.point_color)

        # Draw nodes
        cv2.rectangle(frame,pt1= (offset_x + self.node1.pos[0]-1, offset_y + self.node1.pos[1]-1), pt2= (offset_x + self.node1.pos[0]+1, offset_y + self.node1.pos[1]+1), color=self.point_color)
        cv2.rectangle(frame,pt1= (offset_x + self.node2.pos[0]-1, offset_y + self.node2.pos[1]-1), pt2= (offset_x + self.node2.pos[0]+1, offset_y + self.node2.pos[1]+1), color=self.point_color)

        # Draw packet refs
        for pr in self.prefs1:
            if pr.pos == (-1, -1):
                continue
            cv2.rectangle(frame, (offset_x + pr.pos[0], offset_y + pr.pos[1]), pt2= (offset_x + pr.pos[0], offset_y + pr.pos[1]), color=(0, 128, 0))
        for pr in self.prefs2:
            if pr.pos == (-1, -1):
                continue
            cv2.rectangle(frame, (offset_x + pr.pos[0], offset_y + pr.pos[1]), pt2= (offset_x + pr.pos[0], offset_y + pr.pos[1]), color=(128, 128, 0))
        cv2.rectangle(frame, (offset_x + self.initial_guess1[0], offset_y + self.initial_guess1[1]), pt2= (offset_x + self.initial_guess1[0] +1, offset_y + self.initial_guess1[1] +1), color=(0, 128, 0))
        cv2.rectangle(frame, (offset_x + self.initial_guess2[0], offset_y + self.initial_guess2[1]), pt2= (offset_x + self.initial_guess2[0] +1, offset_y + self.initial_guess2[1]+1), color=(128, 128, 0))
        # cv2.rectangle(frame,pt1= (offset + self.node1.pos-2, y-2), pt2= (offset + self.node1.pos+2, y+2), color=self.point_color)
        # cv2.rectangle(frame,pt1= (offset + self.node2.pos-2, y-2), pt2= (offset + self.node2.pos+2, y+2), color=self.point_color)
        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(enlarged_image, "Total received: " + str(self.total_received) + " | Total misses: " + str(self.total_misses), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # Draw score

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
    def inverse_generate_rssi(self, rssi_scaled):
        # Ensure rssi_scaled is within the valid range [0, 1]
        rssi_scaled = np.clip(rssi_scaled, 0, 1)
        
        # Step 1: Undo the scaling to get rssi
        rssi = self.rssi_min + rssi_scaled * (self.rssi_max - self.rssi_min)
        
        # Step 2: Solve for distance using the path loss model
        distance = 10 ** ((self.rssi_ref - rssi) / (10 * self.path_loss_exponent))
        return distance
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
class node():
    def __init__(self, pos = (10,10), time_to_first_packet=10, send_interval=10, send_std=2):
        self.pos = pos
        self.last_packet_time = 0
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_interval = send_interval
        self.send_std = send_std  # standard deviation
        self.lower_bound_send_time = send_interval / 2
        self.upper_bound_send_time = send_interval * 2

        self.max_transmission_radius = 60
        self.transmission_model = SignalModel(rssi_ref=-30, path_loss_exponent=2.7, noise_floor=-100,
                                              rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30)

    def reset(self):
        self.last_packet_time = 0
        self.time_of_next_packet = self.time_to_first_packet
        posx = random.randint(0, 150)
        posy = random.randint(0, 150)
        #self.pos = (posx,posy)
    def generate_next_interval(self):
        # Generate a truncated normal value for the next time interval
        # a and b are calculated to truncate around the mean interval with some range
        a, b = (self.lower_bound_send_time - self.send_interval) / self.send_std, (
                self.upper_bound_send_time - self.send_interval) / self.send_std
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_std)
        return interval
    def generate_RSSI(self, distance):
        rssi_scaled = self.transmission_model.generate_rssi(distance)
    def inverse_RSSI(self, rssi):
        return self.transmission_model.inverse_generate_rssi(rssi)
    def generate_SNIR(self, distance):
        snir_scaled = self.transmission_model.generate_snir(distance)

    def transmission(self, gpos):
        ploss_scale = 300
        distance = math.dist(self.pos, gpos)
        if distance < self.max_transmission_radius:
            ploss_probability = np.exp( - distance / ploss_scale)
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
                return PACKET_STATUS.RECEIVED, rssi, snir
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
