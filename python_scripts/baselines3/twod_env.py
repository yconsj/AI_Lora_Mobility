from collections import deque

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
        env._skip = skip
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


class FrameStack(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        """stores recent stack_size number of states and inputs to a CNN """
        super(FrameStack, self).__init__(env)
        # Frame stacking
        self.stack_size = stack_size
        # when que is full(max length) and new elements are added, oldest elements will be removed
        original_observation_space = self.env.observation_space
        self.state_stack = deque(maxlen=stack_size)
        self.observation_space = spaces.Box(
            low=np.repeat(original_observation_space.low[np.newaxis, :], stack_size, axis=0),
            high=np.repeat(original_observation_space.high[np.newaxis, :], stack_size, axis=0),
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def reset(self, seed=None, **kwargs):
        state, info = self.env.reset(seed=seed, **kwargs)  # Ensure compatibility across gym versions
        # Initialize the state stack with the reset state repeated
        self.state_stack = deque([state] * self.stack_size, maxlen=self.stack_size)
        return np.array(self.state_stack, dtype=np.float32), info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.state_stack.append(state)
        stacked_state = np.array(self.state_stack, dtype=np.float32)
        return stacked_state, reward, done, truncated, info


class PacketReference:
    def __init__(self, max_pos=(150, 150), pos=(0, 0), rssi=0, snir=0, valid = False):
        self.pos = pos
        self.rssi = rssi
        self.snir = snir
        self.max_pos = max_pos
        self.valid = valid

    def get_scaled(self):
        if self.valid:
            return 0, 0, 0, 0
        return self.pos[0] / self.max_pos[0], self.pos[1] / self.max_pos[1], self.rssi, 1


class ExplorationRewardSystem:
    def __init__(self, grid_size, max_transmission_distance, ploss_scale, fade_rate=0.1):
        """
        Initializes the exploration reward system.

        Args:
            grid_size (tuple): Size of the grid (width, height) as integers.
            max_transmission_distance (float): Maximum transmission distance of the agent.
            fade_rate (float): Rate at which paint fades each step (default 0.1).
        """
        self.grid_size = grid_size
        self.max_transmission_distance = max_transmission_distance
        self.fade_rate = fade_rate
        self.ploss_scale = ploss_scale
        self.paint_matrix = None
        self.reset()

    def reset(self):
        """Resets the paint matrix and initializes it to zeros."""
        self.paint_matrix = np.zeros(self.grid_size, dtype=np.float32)

    def _apply_paint(self, position):
        """
        Paints the cells in the agent's transmission range with intensity starting from 1
        at the agent's position and tapering off according to an exponential decay model.

        Args:
            position (tuple): (x, y) position of the agent as integers.
        """
        x, y = position

        # Create a meshgrid of coordinates for the grid
        x_coords, y_coords = np.meshgrid(
            np.arange(self.grid_size[0]),
            np.arange(self.grid_size[1]),
            indexing="ij",
        )

        # Calculate distances from the agent's position
        distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
        in_range = distances <= self.max_transmission_distance

        # Apply exponential decay for intensity
        intensity = np.exp(-distances / self.ploss_scale)  # Exponential decay
        intensity = np.clip(intensity, 0, 1)  # Ensure the intensity is within [0, 1]

        # Update the paint matrix only for cells within the transmission range
        self.paint_matrix[in_range] = np.maximum(self.paint_matrix[in_range], intensity[in_range])

        # Optionally, ensure the entire paint matrix remains within valid range
        self.paint_matrix = np.clip(self.paint_matrix, 0, 1)

    def _fade_paint(self):
        """Fades the paint in all cells."""
        self.paint_matrix = np.clip(self.paint_matrix - self.fade_rate, 0, None)

    def get_explore_rewards(self, position):
        """
        Computes the exploration reward based on the increase in paint since the last step.
        Args:
            position (tuple): (x, y) position of the agent as integers.
        Returns:
            float: The reward based on the increase in paint since the last step, with a minimum of 0.
        """
        # Calculate the total paint level before applying paint
        previous_paint_level = np.sum(self.paint_matrix)

        # Apply paint from the current position and fade all cells
        self._apply_paint(position)  # Apply paint from the current position
        self._fade_paint()  # Fade all cells

        # Calculate the total paint level after applying paint
        current_paint_level = np.sum(self.paint_matrix)
        # Calculate the increase in paint
        paint_increase = current_paint_level - previous_paint_level
        # Ensure the reward is at least 0
        reward = max(0, paint_increase)
        return reward


def count_valid_packet_reference(pref_tuple: tuple[PacketReference, PacketReference, PacketReference]):
    return sum(1 for value in pref_tuple if value.rssi != -1)


class TwoDEnv(gym.Env):
    def __init__(self, render_mode="none", stack_size=4):
        super(TwoDEnv, self).__init__()
        self._skip = None
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(5, start=0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = 10000  # Maximum steps per episode

        # Observation_space per frame =
        #                     prev_action, (gwpos.x,gwpos.y),             3
        #                     smoothedpos.x, smoothedpos.y                2
        #                     ((x1, x2), rssi) * 3                        9
        #                     ((x1, x2), rssi) * 3                        9
        #                     elapsed_time1, elapsed_time2                2
        #                                                                 25
        self.observation_space = spaces.Box(low=np.array(
            [0] * 3 +
            [0] * 2 +
            (([0] * 2 + [-1] * 1) * 3) * 2 +
            [0] * 2),
            high=np.array(
                [1] * 25), dtype=np.float32)
        # Environment state

        unscaled_speed = 20  # meter per second
        unscaled_max_distance = 3000  # meter
        self.max_distance_x = int(unscaled_max_distance / unscaled_speed)  # scaled by speed
        self.max_distance_y = int(unscaled_max_distance / unscaled_speed)
        self.max_cross_distance = math.dist((0, 0), (self.max_distance_x, self.max_distance_y))
        x = random.randint(0, self.max_distance_x)
        y = random.randint(0, self.max_distance_y)
        self.pos = x, y
        self.prev_pos = self.pos
        self.alpha = 0.01  # Smoothing factor for EWMA
        self.ewma_x = self.pos[0]
        self.ewma_y = self.pos[1]

        unscaled_max_transmission_distance = 1000
        self.max_transmission_distance = unscaled_max_transmission_distance * \
                                         (self.max_distance_x / unscaled_max_distance)

        node1_pos, node2_pos = self.generate_random_node_positions()
        self.ploss_scale = 300
        self.transmission_model = TransmissionModel(ploss_scale=self.ploss_scale, rssi_ref=-30,
                                                    path_loss_exponent=2.7, noise_floor=-100,
                                                    rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30,
                                                    max_transmission_distance=self.max_transmission_distance)
        self.node1 = Node(self.transmission_model, pos=node1_pos, time_to_first_packet=50, send_interval=300)
        self.node2 = Node(self.transmission_model, pos=node2_pos, time_to_first_packet=125, send_interval=300)
        self.prefs1 = (PacketReference(), PacketReference(), PacketReference())
        self.prefs2 = (PacketReference(), PacketReference(), PacketReference())
        self.elapsed_time1 = 0
        self.elapsed_time2 = 0
        self.initial_guess1 = self.pos
        self.initial_guess2 = self.pos

        self.last_packet = 0
        self.pos_reward_min = 0.0
        self.pos_reward_max = 1 / 4.0
        self.packet_reward_max = 100.0
        self.miss_penalty_min = 0.0
        self.miss_penalty_max = self.packet_reward_max / 16

        self.exploration_reward_system = \
            ExplorationRewardSystem(grid_size=(self.max_distance_x, self.max_distance_y),
                                    max_transmission_distance=self.max_transmission_distance,
                                    ploss_scale=self.ploss_scale,
                                    fade_rate=0.005)
        self.exploration_reward_max = 0.1

        self.total_reward = 0
        self.max_misses = 25
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

    @staticmethod
    def generate_random_node_positions():
        minimum_node_distance = 75
        x1 = random.randint(0, 150)
        y1 = random.randint(0, 150)
        node1_pos = (x1, y1)
        while True:
            x2 = random.randint(0, 150)
            y2 = random.randint(0, 150)
            if math.dist((x2, y2), node1_pos) >= minimum_node_distance:
                node2_pos = (x2, y2)
                break
        return node1_pos, node2_pos

    def reset(self, seed=None, options=None):
        # Reset the.pos and steps counter
        self.prev_action = 0
        self.prev_pos = self.pos
        self.visited_pos = dict()
        self.last_packet = 0
        self.total_misses = 0
        self.pos = (int(self.max_distance_x / 2), int(self.max_distance_y / 2))
        self.ewma_x = self.pos[0]
        self.ewma_y = self.pos[1]
        self.initial_guess1 = self.pos
        self.initial_guess2 = self.pos
        self.node1.reset()
        self.node2.reset()
        node1_pos, node2_pos = self.generate_random_node_positions()
        self.node1.pos = node1_pos
        self.node2.pos = node2_pos
        self.steps = 0
        self.total_reward = 0
        self.total_received = 0
        self.exploration_reward_system.reset()
        self.prefs1 = (PacketReference(), PacketReference(), PacketReference())
        self.prefs2 = (PacketReference(), PacketReference(), PacketReference())
        self.elapsed_time1 = 0
        self.elapsed_time2 = 0
        state = [self.prev_action / 4,
                 self.pos[0] / self.max_distance_x, self.pos[1] / self.max_distance_y,
                 self.ewma_x / self.max_distance_x, self.ewma_y / self.max_distance_y,
                 *self.prefs1[0].get_scaled(), *self.prefs1[1].get_scaled(), *self.prefs1[2].get_scaled(),
                 *self.prefs2[0].get_scaled(), *self.prefs2[1].get_scaled(), *self.prefs2[2].get_scaled(),
                 self.elapsed_time1 / self.max_steps,
                 self.elapsed_time2 / self.max_steps
                 ]
        return np.array(state, dtype=np.float32), {}

    def get_pos_reward(self, pos1, pos2, time):
        scaled_time = (time / self.max_steps)
        distance = math.dist(pos1, pos2)
        scaled_distance = 1 - distance / self.max_cross_distance
        scaled_distance_time = scaled_distance * scaled_time
        # Return reward based on scaled distance between a min and max reward
        reward = self.pos_reward_max - scaled_distance_time * (self.pos_reward_max - self.pos_reward_min)

        # Ensure reward is within bounds in case of rounding errors
        reward = max(self.pos_reward_min, min(self.pos_reward_max, reward))
        return reward

    def get_pos_radius_reward(self, gw_pos: tuple[float, float],
                              prefs: tuple[PacketReference, PacketReference, PacketReference],
                              time: int):
        pos_reward = 0
        dist_diff_threshold = self.max_transmission_distance
        # Iterate over all packet references (prefs) to calculate the reward
        for pref in prefs:
            if pref.rssi != -1:
                # Calculate the implied distance to the stationary node based on RSSI
                pref_dist_to_stationary_node = self.transmission_model.inverse_generate_rssi(pref.rssi)

                # Calculate the Euclidean distance from the agent (GW) to the packet reference
                pref_dist_to_gw = math.dist(gw_pos, pref.pos)

                # Calculate the absolute difference between the expected distance (from RSSI) and actual distance
                # (from GW)
                dist_diff = abs(pref_dist_to_stationary_node - pref_dist_to_gw)

                # Reward scaling: If dist_diff is less than a threshold, scale reward between 0 and 1
                if dist_diff < dist_diff_threshold:
                    scaled_reward = 1 - (dist_diff / dist_diff_threshold)  # Scale the reward to be between 0 and 1
                    pos_reward += scaled_reward
        # Scale the final reward based on time (before clipping)
        scaled_time = time / self.max_steps
        pos_reward *= scaled_time

        # Ensure the final reward is within the defined range [pos_reward_min, pos_reward_max]
        pos_reward = max(self.pos_reward_min, min(self.pos_reward_max, pos_reward))

        # Return the total calculated reward
        return pos_reward

    @staticmethod
    def is_new_best_pref(pref, p):
        (pref1, pref2, pref3) = pref
        if p.pos == pref1.pos or p.pos == pref2.pos or p.pos == pref3.pos:
            return False
        if p.rssi > pref1.rssi or p.rssi > pref2.rssi or p.rssi > pref3.rssi:
            return True
        return False

    @staticmethod
    def insert_best_pref(pref, p):
        (pref1, pref2, pref3) = pref
        if p.rssi > pref1.rssi:
            pref3 = pref2
            pref2 = pref1
            pref1 = p
            return pref1, pref2, pref3
        if p.rssi > pref2.rssi:
            pref3 = pref2
            pref2 = p
            return pref1, pref2, pref3
        if p.rssi > pref3.rssi:
            pref3 = p
            return pref1, pref2, pref3

    def get_miss_penalty(self, pos1, pos2):
        """       distance = math.dist(pos1, pos2)
        scaled_distance = distance / self.max_cross_distance
        # Return reward based on scaled distance between a min and max reward
        penalty = self.miss_penalty_min + scaled_distance * (self.miss_penalty_max - self.miss_penalty_min)

        # Ensure reward is within bounds in case of rounding errors
        penalty = min(self.miss_penalty_max, max(self.miss_penalty_min, penalty))
        return -penalty"""
        distance = math.dist(pos1, pos2)

        # If the distance is greater than max_cross_distance, use the max penalty
        if distance > self.max_transmission_distance:
            return -self.miss_penalty_max

        # If within range, scale the penalty with distance
        scaled_distance = self.transmission_model.calculate_ploss_probability(
            distance)  # distance / self.max_transmission_distance
        penalty = self.miss_penalty_min + scaled_distance * (self.miss_penalty_max - self.miss_penalty_min)

        # Ensure penalty is within bounds in case of rounding errors
        penalty = min(self.miss_penalty_max, max(self.miss_penalty_min, penalty))

        return -penalty

    def get_explore_reward(self, pos, time):
        base_reward = 0.01
        if pos not in self.visited_pos.keys():
            self.visited_pos[pos] = time
            multiplier = abs(pos[0] - int(self.max_distance_x / 2)) * abs(pos[1] - int(self.max_distance_y / 2)) / (
                    self.max_cross_distance * 2)
            return base_reward * multiplier

        base_reward = base_reward * (time - self.visited_pos[pos]) / self.max_steps

        self.visited_pos[pos] = time

        return 0

    @staticmethod
    def trilateration_residuals(params, positions, distances):
        x, y = params  # Unknown position (x, y)
        residuals = []

        for (px, py), d in zip(positions, distances):
            calculated_distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            residuals.append(calculated_distance - d)

        return residuals

    def trilateration(self, prefs, initial_guess):
        (pref1, pref2, pref3) = prefs
        positions = [pref1.pos, pref2.pos, pref3.pos]
        distances = [self.node1.inverse_RSSI(pref1.rssi), self.node1.inverse_RSSI(pref2.rssi),
                     self.node1.inverse_RSSI(pref3.rssi)]
        # Initial guess for the unknown position (x, y)

        # Perform least squares optimization
        result = least_squares(self.trilateration_residuals, initial_guess, args=(positions, distances))

        # Return the optimized position
        return int(result.x[0]), int(result.x[1])

    def step(self, action):
        if self.render_mode == "cv2":
            self.render()
        reward = 0
        self.steps += 1

        if action == 0:  # stand still
            # nothing
            pass
        elif action == 1:  # left
            if self.pos[0] > 0:
                self.pos = (self.pos[0] - 1, self.pos[1])
        elif action == 2:  # right
            if self.pos[0] < self.max_distance_x:
                self.pos = (self.pos[0] + 1, self.pos[1])
        elif action == 3:  # up
            if self.pos[1] < self.max_distance_y:
                self.pos = (self.pos[0], self.pos[1] + 1)
        elif action == 4:  # down
            if self.pos[1] > 0:
                self.pos = (self.pos[0], self.pos[1] - 1)
        self.ewma_x = self.alpha * self.pos[0] + (1 - self.alpha) * self.ewma_x
        self.ewma_y = self.alpha * self.pos[1] + (1 - self.alpha) * self.ewma_y
        received1, rssi1, snir1 = self.node1.send(self.steps, self.pos)
        received2, rssi2, snir2 = self.node2.send(self.steps, self.pos)
        self.elapsed_time1 = min(self.max_steps, self.elapsed_time1 + 1)
        self.elapsed_time2 = min(self.max_steps, self.elapsed_time2 + 1)
        p1 = PacketReference(pos=self.pos, rssi=rssi1, snir=snir1)
        p2 = PacketReference(pos=self.pos, rssi=rssi2, snir=snir2)
        if received1 == PACKET_STATUS.RECEIVED:
            reward = self.packet_reward_max
            if self.last_packet == 2:
                reward += self.packet_reward_max
            self.last_packet = 1
            self.total_received += 1
            self.elapsed_time1 = 0
            if self.is_new_best_pref(self.prefs1, p1):
                self.prefs1 = self.insert_best_pref(self.prefs1, p1)
        elif received2 == PACKET_STATUS.RECEIVED:
            reward = self.packet_reward_max
            if self.last_packet == 1:
                reward += self.packet_reward_max
            self.last_packet = 2
            self.total_received += 1
            self.elapsed_time2 = 0
            # TODO: insert packet if
            if self.is_new_best_pref(self.prefs2, p2):
                self.prefs2 = self.insert_best_pref(self.prefs2, p2)

        elif received1 == PACKET_STATUS.LOST:
            self.total_misses += 1
            miss_penalty = self.get_miss_penalty(self.pos, self.node1.pos)

            # higher penalty for missing packets from the same node in a row
            miss_penalty = miss_penalty * 2 if self.elapsed_time1 > self.elapsed_time2 else miss_penalty

            scale_by_packets_found = 1  # (sum(1 for value in self.prefs1 if value.rssi != -1) / len(self.prefs1))
            reward = miss_penalty * scale_by_packets_found
        elif received2 == PACKET_STATUS.LOST:
            # and (self.prefs2[0].rssi == -1 or self.prefs2[1].rssi == -1 or self.prefs2[2].rssi == -1):
            self.total_misses += 1
            miss_penalty = self.get_miss_penalty(self.pos, self.node2.pos)

            # higher penalty for missing packets from the same node in a row
            miss_penalty = miss_penalty * 2 if self.elapsed_time2 > self.elapsed_time1 else miss_penalty

            scale_by_packets_found = 1  # (sum(1 for value in self.prefs2 if value.rssi != -1) / len(self.prefs2))
            reward = miss_penalty * scale_by_packets_found

        if self.prefs1[0].rssi != -1 and self.prefs1[1].rssi != -1 and self.prefs1[2].rssi != -1:
            approx_pos = self.trilateration(self.prefs1, self.initial_guess1)
            self.initial_guess1 = approx_pos
        if self.prefs2[0].rssi != -1 and self.prefs2[1].rssi != -1 and self.prefs2[2].rssi != -1:
            approx_pos = self.trilateration(self.prefs2, self.initial_guess2)
            self.initial_guess2 = approx_pos

        if self.elapsed_time1 > self.elapsed_time2:
            # and self.prefs1[0].rssi != -1 and self.prefs1[1].rssi != -1 and self.prefs1[2].rssi != -1

            # reward += self.get_pos_reward(self.pos, self.node1.pos, self.elapsed_time1)
            # reward += self.get_pos_reward(self.pos, self.initial_guess1, self.elapsed_time1)
            reward += self.get_pos_radius_reward(self.pos, self.prefs1, self.elapsed_time1)
        elif self.elapsed_time1 <= self.elapsed_time2:
            # and self.prefs2[0].rssi != -1 and self.prefs2[1].rssi != -1 and self.prefs2[2].rssi != -1
            # reward += self.get_pos_reward(self.pos, self.node2.pos, self.elapsed_time2)
            # reward += self.get_pos_reward(self.pos, self.initial_guess2, self.elapsed_time2)
            reward += self.get_pos_radius_reward(self.pos, self.prefs2, self.elapsed_time2)

        # reward += self.get_explore_reward(self.pos, self.steps)

        explore_reward_scale = (1 + count_valid_packet_reference(self.prefs1) + count_valid_packet_reference(self.prefs2)) / (1 + len(self.prefs1) + len(self.prefs2))
        explore_reward = min(self.exploration_reward_system.get_explore_rewards(
                self.pos) * self.exploration_reward_max * explore_reward_scale,self.exploration_reward_max)

        reward += explore_reward
        reward = min(self.packet_reward_max * 2, reward)
        done = self.steps >= self.max_steps or self.total_misses >= self.max_misses
        self.total_reward += reward
        state = [self.prev_action / 4,
                 self.pos[0] / self.max_distance_x, self.pos[1] / self.max_distance_y,
                 self.ewma_x / self.max_distance_x, self.ewma_y / self.max_distance_y,
                 *self.prefs1[0].get_scaled(), *self.prefs1[1].get_scaled(), *self.prefs1[2].get_scaled(),
                 *self.prefs2[0].get_scaled(), *self.prefs2[1].get_scaled(), *self.prefs2[2].get_scaled(),
                 self.elapsed_time1 / self.max_steps,
                 self.elapsed_time2 / self.max_steps
                 ]
        if (self._skip is not None and self.steps % self._skip == 0):
            self.prev_action = action
        info = {'total_received': self.total_received,
                'total_misses': self.total_misses}
        return np.array(state, dtype=np.float32), reward, done, False, info

    def render(self):

        # Map the position [0, 1] to the x-coordinate along the line [50, 550]
        x = int(self.pos[0])
        y = int(self.pos[1])
        # Create a new black image
        offset_x = int((self.width - self.max_distance_x) / 2)
        offset_y = int((self.height - self.max_distance_y) / 2)
        # Create a new black image (background)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the grid as a base layer (vectorized)
        grid = self.exploration_reward_system.paint_matrix

        # Vectorized intensity calculation (between 0 and 1)
        green = np.clip(grid * 255, 0, 255).astype(np.uint8)  # Green intensity
        red = np.clip((1 - grid) * 255, 0, 255).astype(np.uint8)  # Red intensity

        # Create the color frame using stacking (0 for blue, green and red as calculated)
        color_frame = np.dstack((np.zeros_like(red), green, red))

        # Swap the axes to match the frame's expected orientation (flip row/column)
        color_frame = color_frame.transpose((1, 0, 2))

        # Place the color frame onto the final frame (accounting for the offsets)
        frame[offset_y:offset_y + grid.shape[0], offset_x:offset_x + grid.shape[1]] = color_frame

        # Draw the line and moving point
        cv2.line(frame, pt1=(offset_x, offset_y + int(self.max_distance_y / 2)),
                 pt2=(offset_x + self.max_distance_x, offset_y + int(self.max_distance_y / 2)), color=self.line_color)
        cv2.line(frame, pt1=(offset_x + int(self.max_distance_x / 2), offset_y),
                 pt2=(offset_x + int(self.max_distance_x / 2), self.max_distance_y + offset_y), color=self.line_color)
        cv2.rectangle(frame, pt1=(offset_x + x - 2, offset_y + y - 2), pt2=(offset_x + x + 2, offset_y + y + 2),
                      color=self.point_color)

        # Draw nodes with distinct white shapes
        # Node 1 as a filled white rectangle
        cv2.rectangle(frame, pt1=(offset_x + self.node1.pos[0] - 2, offset_y + self.node1.pos[1] - 2),
                      pt2=(offset_x + self.node1.pos[0] + 2, offset_y + self.node1.pos[1] + 2), color=(255, 255, 255),
                      thickness=-1)

        # Node 2 as a white rectangle with a black outline
        cv2.rectangle(frame, pt1=(offset_x + self.node2.pos[0] - 2, offset_y + self.node2.pos[1] - 2),
                      pt2=(offset_x + self.node2.pos[0] + 2, offset_y + self.node2.pos[1] + 2), color=(255, 255, 255),
                      thickness=-1)
        cv2.rectangle(frame, pt1=(offset_x + self.node2.pos[0] - 2, offset_y + self.node2.pos[1] - 2),
                      pt2=(offset_x + self.node2.pos[0] + 2, offset_y + self.node2.pos[1] + 2), color=(0, 0, 0),
                      thickness=1)

        # Draw packet refs with black dots and additional circles based on RSSI
        for pr in self.prefs1 + self.prefs2:
            if pr.pos == (-1, -1):
                continue
            # Draw a black dot at the packet ref's position
            pr_radius = 2  # Set radius of the black dot
            cv2.circle(frame, (offset_x + pr.pos[0], offset_y + pr.pos[1]), pr_radius, (0, 0, 0), -1)

            # Calculate the radius from RSSI (convert RSSI to distance)
            rssi_distance = self.transmission_model.inverse_generate_rssi(pr.rssi)  # Assume a method exists
            circle_radius = int(rssi_distance)  # Convert to integer for drawing
            cv2.circle(frame, (offset_x + pr.pos[0], offset_y + pr.pos[1]), circle_radius, (255, 0, 0),
                       1)  # Blue circle

        # Draw the initial guesses with black boxes
        cv2.rectangle(frame, (offset_x + self.initial_guess1[0] - 2, offset_y + self.initial_guess1[1] - 2),
                      (offset_x + self.initial_guess1[0] + 2, offset_y + self.initial_guess1[1] + 2), color=(0, 128, 0))
        cv2.rectangle(frame, (offset_x + self.initial_guess2[0] - 2, offset_y + self.initial_guess2[1] - 2),
                      (offset_x + self.initial_guess2[0] + 2, offset_y + self.initial_guess2[1] + 2),
                      color=(128, 128, 0))

        # Draw the maximum transmission distance circle
        cv2.circle(
            frame,  # img: the image to draw on
            center=(offset_x + x, offset_y + y),  # center: the circle's center
            radius=int(self.max_transmission_distance),  # radius: max transmission distance
            color=(255, 0, 0),  # color: blue (B, G, R)
            thickness=1  # thickness: 1 for outline
        )

        # cv2.rectangle(frame,pt1= (offset + self.node1.pos-2, y-2), pt2= (offset + self.node1.pos+2, y+2), color=self.point_color)
        # cv2.rectangle(frame,pt1= (offset + self.node2.pos-2, y-2), pt2= (offset + self.node2.pos+2, y+2), color=self.point_color)
        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(enlarged_image,
                    "Total received: " + str(self.total_received) + " | Total misses: " + str(self.total_misses),
                    (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw score

        cv2.imshow(self.window_name, enlarged_image)
        cv2.waitKey(5)  # Wait a short time to create the animation effect

    def close(self):
        cv2.destroyAllWindows()


class TransmissionModel:
    def __init__(self, max_transmission_distance=60.0, ploss_scale=300, rssi_ref=-30, path_loss_exponent=2.7,
                 noise_floor=-100,
                 rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30):
        self.ploss_scale = ploss_scale
        self.max_radius = max_transmission_distance
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
        rssi = self.rssi_min + rssi_scaled * (self.rssi_max - self.rssi_min)
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
        return np.clip(snir_scaled, 0, 1)

    def calculate_ploss_probability(self, distance):
        """
        Calculate the probability of packet loss based on distance.
        Emulates the FLoRa framework's packet loss handling.
        """
        if distance > self.max_radius:
            return 1.0  # Out of range -> always lost
        distance = max(distance, 0.00001)  # Avoid divide by zero
        ploss_probability = 1 - np.exp(-distance / self.ploss_scale)  # Exponential decay
        return 0  # ploss_probability


class PACKET_STATUS(Enum):
    RECEIVED = 1
    LOST = 2
    NOT_SENT = 3


class Node:
    def __init__(self, transmission_model, pos=(10, 10), time_to_first_packet=10, send_interval=10, send_std=2
                 ):
        self.pos = pos
        self.last_packet_time = 0
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_interval = send_interval
        self.send_std = send_std  # standard deviation
        self.lower_bound_send_time = send_interval - (send_interval / 2)
        self.upper_bound_send_time = send_interval + (send_interval / 2)

        self.transmission_model = transmission_model

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

    def inverse_RSSI(self, rssi):
        return self.transmission_model.inverse_generate_rssi(rssi)

    def transmission(self, gpos):
        """
        Simulate the transmission of a signal to a given position (gpos).
        Returns:
            - (True, rssi_scaled, snir_scaled) if the transmission is successful.
            - (False, 0, 0) if the transmission fails.
        """
        distance = math.dist(self.pos, gpos)

        # Check if the target is out of range
        if distance > self.transmission_model.max_radius:
            return False, 0, 0

        # Use TransmissionModel to calculate packet loss probability
        ploss_probability = self.transmission_model.calculate_ploss_probability(distance)
        transmission_success = np.random.rand() < (1 - ploss_probability)  # Success if random < 1 - P(loss)

        if transmission_success:
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
