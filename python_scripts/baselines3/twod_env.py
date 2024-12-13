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
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and return the last observation."""
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        # print(f"{total_reward = }")
        return obs, total_reward, done, trunc, info


class PacketReference:
    def __init__(self, max_pos=(150, 150), pos=(0, 0), rssi=0, snir=0, valid=False):
        self.pos = pos
        self.rssi = rssi
        self.snir = snir
        self.max_pos = max_pos
        self.valid = valid

    def get_scaled(self):
        if self.valid:
            return self.pos[0] / self.max_pos[0], self.pos[1] / self.max_pos[1], self.rssi, 1
        return 0, 0, 0, 0


class ExplorationRewardSystem:
    def __init__(self, grid_size, max_transmission_distance, ploss_scale, fade_rate=0.1, tau=300):
        """
        Initializes the exploration reward system.

        Args:
            grid_size (tuple): Size of the grid (width, height) as integers.
            max_transmission_distance (float): Maximum transmission distance of the agent.
            fade_rate (float): Rate at which paint fades each step (default 0.1).
            ploss_scale (float): Scaling factor for distance-based intensity decay.
            tau (int): Time until a cell should be fully painted.
        """
        self.grid_size = grid_size
        self.max_transmission_distance = max_transmission_distance
        self.fade_rate = fade_rate
        self.ploss_scale = ploss_scale

        # Precompute the number of cells within max transmission distance
        self.cells_within_range = self._calculate_cells_within_range()
        self.tau = tau  # Duration before cell is painted fully
        self.time_matrix = np.zeros(self.grid_size, dtype=np.float32)  # Time spent in each cell
        self.paint_matrix = np.zeros(self.grid_size, dtype=np.float32)  # Current paint levels
        self.reset()

    def _calculate_cells_within_range(self):
        """
        Calculate the number of cells within max transmission distance.
        Returns:
            int: Number of cells within the max transmission distance.
        """
        x_coords, y_coords = np.meshgrid(
            np.arange(self.grid_size[0]),
            np.arange(self.grid_size[1]),
            indexing="ij",
        )
        center_x, center_y = self.grid_size[0] // 2, self.grid_size[1] // 2  # Arbitrary center
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        in_range = distances <= self.max_transmission_distance
        return np.sum(in_range)

    def reset(self):
        """Resets the paint and time matrices."""
        self.paint_matrix.fill(0)
        self.time_matrix.fill(0)

    def _apply_paint(self, position, time_step=1):
        """
        Paints the cells in the agent's transmission range. Paint increases
        incrementally until reaching a distance-based maximum.

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

        # Increment the time spent in the current cell
        self.time_matrix[x, y] += time_step

        # Calculate the maximum intensity for each cell based on distance
        max_intensity = np.exp(-distances / self.ploss_scale)

        # Calculate paint levels based on cumulative time spent (exponential growth model)
        self.paint_matrix[in_range] = np.maximum(max_intensity[in_range] * (
                1 - np.exp(-self.time_matrix[x, y] / self.tau)
        ), self.paint_matrix[in_range])

        # Ensure paint levels are within the valid range [0, 1]
        self.paint_matrix = np.clip(self.paint_matrix, 0, 1)

    def _fade_paint(self):
        """Fades the paint in all cells."""
        self.paint_matrix = np.clip(self.paint_matrix - self.fade_rate, 0, 1)

    def _fade_time(self):
        """Fades the time spent in all cells to reduce the value of old areas."""
        self.time_matrix = np.clip(self.time_matrix - (1.0 / self.tau), 0, None)

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
        self._apply_paint(position)
        self._fade_paint()  # Fade paint after applying
        self._fade_time()  # Fade time spent in cells

        # Calculate the total paint level after applying paint
        current_paint_level = np.sum(self.paint_matrix)

        # Calculate the increase in paint
        paint_increase = np.maximum(0, current_paint_level - previous_paint_level)
        paint_increase_ratio = paint_increase / self.cells_within_range

        # Normalize coverage by dividing the sum of paint by the maximum possible paint level
        coverage_bonus = np.sum(self.paint_matrix) / self.paint_matrix.size

        # Combine paint increase and coverage bonus into the reward
        alpha = 0.75
        reward = (1 - alpha) * coverage_bonus + alpha * paint_increase_ratio
        reward = min(1.0, max(reward, 0.0))
        return reward


def count_valid_packet_reference(pacref_tuple: tuple[PacketReference, PacketReference, PacketReference]):
    return sum(1 for pacref in pacref_tuple if pacref.valid)


def generate_random_node_positions(minimum_node_distance=75.0, gwpos=None):
    x1 = random.randint(0, 150)
    y1 = random.randint(0, 150)
    if gwpos is not None:
        while math.dist((x1, y1), gwpos) < minimum_node_distance:
            x1, y1 = random.randint(0, 150), random.randint(0, 150)
    node1_pos = (x1, y1)
    while True:
        x2 = random.randint(0, 150)
        y2 = random.randint(0, 150)
        if math.dist((x2, y2), node1_pos) >= minimum_node_distance:
            node2_pos = (x2, y2)
            break
    return node1_pos, node2_pos


def is_new_best_pacref(prefs, p):
    # Unpack the preferences (pref) for clarity
    pref1, pref2, pref3 = prefs

    # Check if the packet's position is already in any of the preferences
    if p.pos in [pref1.pos, pref2.pos, pref3.pos]:
        return False

    # Check if any of the preferences are not valid
    if not pref1.valid or not pref2.valid or not pref3.valid:
        return True

    # Check if the packet's RSSI is higher than any of the existing preferences
    return p.rssi > min(pref1.rssi, pref2.rssi, pref3.rssi)


def insert_best_pacref(prefs, p):
    # Unpack the preferences (pref) for clarity
    pref1, pref2, pref3 = prefs
    # Replace invalid packets in priority order
    if not pref1.valid:
        return p, pref2, pref3
    elif not pref2.valid:
        return pref1, p, pref3
    elif not pref3.valid:
        return pref1, pref2, p
    # Insert packet `p` into the preferences based on RSSI
    if p.rssi > pref1.rssi:
        return p, pref1, pref2  # `p` becomes the new best preference
    elif p.rssi > pref2.rssi:
        return pref1, p, pref2  # `p` becomes the second best preference
    elif p.rssi > pref3.rssi:
        return pref1, pref2, p  # `p` becomes the third best preference

    return prefs  # Return the original preferences if no new packet is inserted


class TwoDEnv(gym.Env):
    def __init__(self, render_mode="none", history_length=3, n_skips=10):
        super(TwoDEnv, self).__init__()
        self.n_skips = n_skips
        self.history_length = history_length  # k
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.action_space = spaces.Discrete(5, start=0)
        # The observation space is a single value (our current "position")
        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = 20000  # Maximum steps per episode

        # Observation_space per frame =
        #                     (gwpos.x,gwpos.y),                           | 2
        #                     ((x1, x2), rssi, valid) * 3                  | 12
        #                     number of valid packets received from node 1 | 1
        #                     ((x1, x2), rssi, valid) * 3                  | 12
        #                     number of valid packets received from node 2 | 1
        #                     elapsed_time1, elapsed_time2                 | 2
        #                     last_packet                                  | 1
        #                     curriculum_index                             | 1
        #                     steps                                        | 1
        #                     gwpos_history                                | 2 * k
        #                     action_history                               | 1 * k
        #                                                                  | 33 + 3 * k
        self.observation_space = spaces.Box(low=np.array(
            [0] * 2 +
            ((([0] * 4) * 3) + [0] * 1) * 2 +
            [0] * 2 +
            [0] * 1 +
            [0] * 1 +
            [0] * 1 +
            [0] * 2 * self.history_length +
            [0] * 1 * self.history_length),
            high=np.array(
                [1] * 32 + [1] * 1 + [1] * 3 * self.history_length), dtype=np.float32)
        # Environment state

        unscaled_speed = 20  # meter per second
        unscaled_max_distance = 3000  # meter
        self.max_distance_x = int(unscaled_max_distance / unscaled_speed)  # scaled by speed
        self.max_distance_y = int(unscaled_max_distance / unscaled_speed)
        self.max_cross_distance = math.dist((0, 0), (self.max_distance_x, self.max_distance_y))
        x = self.max_distance_x // 2 # random.randint(0, self.max_distance_x)
        y = self.max_distance_y // 2 # random.randint(0, self.max_distance_y)
        self.pos = x, y
        self.history_positions = deque([self.pos] * self.history_length, maxlen=self.history_length)
        self.history_prev_actions = deque([0] * self.history_length, maxlen=self.history_length)

        unscaled_max_transmission_distance = 1000
        self.max_transmission_distance = unscaled_max_transmission_distance * \
            (self.max_distance_x / unscaled_max_distance)

        node1_pos, node2_pos = generate_random_node_positions(minimum_node_distance=self.max_transmission_distance,
                                                              gwpos=self.pos)
        node1_pos = 0, 120
        node2_pos = 120, 0
        self.ploss_scale = 300
        self.transmission_model = TransmissionModel(ploss_scale=self.ploss_scale, rssi_ref=-30,
                                                    path_loss_exponent=2.7, noise_floor=-100,
                                                    rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30,
                                                    max_transmission_distance=self.max_transmission_distance)

        self.node_send_interval = 300
        self.node1 = Node(self.transmission_model, pos=node1_pos, time_to_first_packet=50,
                          send_interval=self.node_send_interval)
        self.node2 = Node(self.transmission_model, pos=node2_pos, time_to_first_packet=125,
                          send_interval=self.node_send_interval)
        self.pacrefs1 = (PacketReference(), PacketReference(), PacketReference())
        self.pacrefs2 = (PacketReference(), PacketReference(), PacketReference())
        self.total_received1 = 0
        self.total_received2 = 0
        self.elapsed_time1 = 0
        self.elapsed_time2 = 0
        self.expected_max_received_per_node = self.max_steps // self.node_send_interval

        self.last_packet = 0
        self.pos_reward_min = 0.0
        self.pos_reward_max = 0.004
        self.packet_reward_max = 5
        self.packet_reward_min = self.packet_reward_max / 10
        self.miss_penalty_max = self.packet_reward_max / 5
        self.miss_penalty_min = self.miss_penalty_max / 2

        self.exploration_reward_system = \
            ExplorationRewardSystem(grid_size=(self.max_distance_x, self.max_distance_y),
                                    max_transmission_distance=self.max_transmission_distance,
                                    ploss_scale=self.ploss_scale / 2,
                                    fade_rate=0.0001, tau=self.node_send_interval // 2)
        self.exploration_reward_max = 0.05

        self.total_reward = 0
        self.max_misses = 30
        self.total_misses = 0
        self.total_received = 0
        self.width, self.height = 175, 175  # Size of the window
        self.point_radius = 1
        self.point_color = (0, 0, 255)  # Red color
        self.line_color = (255, 0, 0)  # Blue color

        self.exploration_stage_index = 1
        self.localization_stage_index = 2
        self.reception_stage_index = 3
        self.stage = self.exploration_stage_index

        if render_mode == "cv2":
            self.window_name = "RL Animation"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the.pos and steps counter
        self.last_packet = 0
        self.total_misses = 0
        x = self.max_distance_x // 2  # random.randint(0, self.max_distance_x)
        y = self.max_distance_y // 2  # random.randint(0, self.max_distance_y)
        self.pos = x, y
        self.history_positions = deque([self.pos] * self.history_length, maxlen=self.history_length)
        self.history_prev_actions = deque([0] * self.history_length, maxlen=self.history_length)

        self.node1.reset()
        self.node2.reset()
        node1_pos, node2_pos = generate_random_node_positions(minimum_node_distance=self.max_transmission_distance,
                                                              gwpos=self.pos)
        # self.node1.pos = node1_pos
        # self.node2.pos = node2_pos
        self.steps = 0
        self.total_reward = 0
        self.total_received = 0
        self.total_received1 = 0
        self.total_received2 = 0
        self.exploration_reward_system.reset()
        self.pacrefs1 = (PacketReference(), PacketReference(), PacketReference())
        self.pacrefs2 = (PacketReference(), PacketReference(), PacketReference())
        self.elapsed_time1 = 0
        self.elapsed_time2 = 0
        self.stage = self.exploration_stage_index

        state = self.get_state()
        info = {'total_received': self.total_received,
                'total_misses': self.total_misses}
        return state, info

    def get_state(self):
        """
        Constructs the current state, including historical buffers for temporal context.

        Returns:
            np.array: Flattened state vector.
        """
        # Current state elements
        current_state = [
            self.pos[0] / self.max_distance_x, self.pos[1] / self.max_distance_y,
            *self.pacrefs1[0].get_scaled(), *self.pacrefs1[1].get_scaled(), *self.pacrefs1[2].get_scaled(),
            count_valid_packet_reference(self.pacrefs1) / 3,
            *self.pacrefs2[0].get_scaled(), *self.pacrefs2[1].get_scaled(), *self.pacrefs2[2].get_scaled(),
            count_valid_packet_reference(self.pacrefs2) / 3,
            self.elapsed_time1 / self.max_steps,
            self.elapsed_time2 / self.max_steps,
            self.last_packet / 2,
            self.stage / self.reception_stage_index,
            self.steps / self.max_steps,
        ]

        # Historical elements
        # Normalize historical positions
        normalized_positions = [
            (pos[0] / self.max_distance_x, pos[1] / self.max_distance_y)
            for pos in self.history_positions
        ]
        flattened_positions = [val for pos in normalized_positions for val in pos]

        # Normalize historical actions
        normalized_actions = [action / 4 for action in self.history_prev_actions]

        # Combine everything into a single state vector
        full_state = (
                current_state
                + normalized_actions
                + flattened_positions
        )

        return np.array(full_state, dtype=np.float32)

    def get_pos_reward(self, pos1, pos2, time):
        max_pos_reward_time = self.node_send_interval * 4
        scaled_time = min(1, time / max_pos_reward_time)
        distance = math.dist(pos1, pos2)
        scaled_distance = 1 - distance / self.max_cross_distance
        scaled_distance_time = scaled_distance * scaled_time
        # Return reward based on scaled distance between a min and max reward
        reward = self.pos_reward_max - scaled_distance_time * (self.pos_reward_max - self.pos_reward_min)

        # Ensure reward is within bounds in case of rounding errors
        reward = max(self.pos_reward_min, min(self.pos_reward_max, reward))
        return reward

    def get_pos_radius_reward(self, gw_pos: tuple[float, float],
                              pacrefs: tuple[PacketReference, PacketReference, PacketReference],
                              time: int):
        max_pos_reward_time = self.node_send_interval * 4
        scaled_time = min(1.0, time / max_pos_reward_time)

        pos_reward = 0
        dist_diff_threshold = self.max_transmission_distance
        # Iterate over all packet references (pacrefs) to calculate the reward
        for pacref in pacrefs:
            if pacref.valid:
                # Calculate the implied distance to the stationary node based on RSSI
                pacref_dist_to_stationary_node = self.transmission_model.inverse_generate_rssi(pacref.rssi)

                # Calculate the Euclidean distance from the agent (GW) to the packet reference
                pacref_dist_to_gw = math.dist(gw_pos, pacref.pos)

                # Calculate the absolute difference between the expected distance (from RSSI) and actual distance
                # (from GW)
                dist_diff = abs(pacref_dist_to_stationary_node - pacref_dist_to_gw)

                # Reward scaling: If dist_diff is less than a threshold, scale reward between 0 and 1
                if dist_diff < dist_diff_threshold:
                    scaled_reward = 1 - (dist_diff / dist_diff_threshold)  # Scale the reward to be between 0 and 1

                    pos_reward += scaled_reward
        pos_reward = self.pos_reward_max - pos_reward * (self.pos_reward_max - self.pos_reward_min)
        pos_reward *= scaled_time

        # Ensure the final reward is within the defined range [pos_reward_min, pos_reward_max]
        pos_reward = max(self.pos_reward_min, min(self.pos_reward_max, pos_reward))

        # Return the total calculated reward
        return pos_reward

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

    def exploration_stage(self, packet1, packet2):
        max_exploration_steps = 1200
        reward = 0
        # if self.pos == self.prev_pos:
        #    reward += -0.1  # Small penalty for inaction
        min_time_scale_value = 0.5
        time_scale = max(min_time_scale_value, (1 - (self.steps / max_exploration_steps)) * min_time_scale_value)
        reward += min(self.exploration_reward_system.get_explore_rewards(
            self.pos) * self.exploration_reward_max * time_scale, self.exploration_reward_max)
        received1, _, _ = packet1
        received2, _, _ = packet2
        if (received1 == PACKET_STATUS.RECEIVED) and (self.total_received1 <= 1):
            reward += self.packet_reward_min
        elif received1 == PACKET_STATUS.LOST:
            penalty_multiplier = min(3, self.elapsed_time1 // self.node_send_interval)
            reward -= self.miss_penalty_max * penalty_multiplier

        if (received2 == PACKET_STATUS.RECEIVED) and (self.total_received2 <= 1):
            reward += self.packet_reward_min
            pass
        elif received2 == PACKET_STATUS.LOST:
            penalty_multiplier = min(3, self.elapsed_time2 // self.node_send_interval)
            reward -= self.miss_penalty_max * penalty_multiplier

        # stage transition
        if (self.total_received1 >= 1) and (self.total_received2 >= 1):
            self.stage = self.localization_stage_index

        return reward

    def localization_stage(self, packet1, packet2):
        max_localization_stage = 4000
        reward = 0

        # if self.pos == self.prev_pos:
        # reward += -0.1  # Small penalty for inaction

        # continue giving a small part of exploration reward:
        explore_scale = 0.5
        reward += min(self.exploration_reward_system.get_explore_rewards(
            self.pos) * self.exploration_reward_max * explore_scale, self.exploration_reward_max)

        received1, _, _ = packet1
        received2, _, _ = packet2
        if (received1 == PACKET_STATUS.RECEIVED) and (self.total_received1 < 4):
            reward += self.packet_reward_min
        elif received1 == PACKET_STATUS.LOST:
            penalty_multiplier = min(3, self.elapsed_time1 // self.node_send_interval)
            reward -= self.miss_penalty_max * penalty_multiplier
        if (received2 == PACKET_STATUS.RECEIVED) and (self.total_received2 < 4):
            reward += self.packet_reward_min
        elif received2 == PACKET_STATUS.LOST:
            penalty_multiplier = min(3, self.elapsed_time2 // self.node_send_interval)
            reward -= self.miss_penalty_max * penalty_multiplier

        time_scale = max(0.1, 1 - self.steps / max_localization_stage)
        if self.total_received1 < 4 and self.last_packet == 2:
            reward += self.get_pos_reward(self.pos, self.node1.pos, self.elapsed_time1) * time_scale
            # self.get_pos_radius_reward(self.pos, self.pacrefs1, self.elapsed_time1)
        if self.total_received2 < 4 and self.last_packet == 1:
            reward += self.get_pos_reward(self.pos, self.node2.pos, self.elapsed_time2) * time_scale
            # reward += self.get_pos_radius_reward(self.pos, self.pacrefs2, self.elapsed_time2)
        # stage transition
        if (self.total_received1 >= 3) and (self.total_received2 >= 3):
            self.stage = self.reception_stage_index
        return reward

    def reception_stage(self, packet1, packet2):
        reward = 0
        received1, rssi1, snir1 = packet1
        received2, rssi2, snir2 = packet2
        reception_penalty_max_time = 6000.0
        penalty_time_scale = min(1.0, self.steps / reception_penalty_max_time)

        # position reward based on actual node location
        if self.last_packet == 1:
            reward += self.get_pos_reward(self.pos, self.node1.pos, self.elapsed_time1)
        else:
            reward += self.get_pos_reward(self.pos, self.node2.pos, self.elapsed_time2)

        # Packet reception reward & penalty
        if received1 == PACKET_STATUS.RECEIVED:
            reward += self.packet_reward_max
            if self.last_packet == 2:
                reward += self.packet_reward_max
        elif received1 == PACKET_STATUS.LOST:
            penalty_multiplier = min(3, self.elapsed_time1 // self.node_send_interval)
            reward -= self.miss_penalty_max * penalty_multiplier

        if received2 == PACKET_STATUS.RECEIVED:
            reward += self.packet_reward_max
            if self.last_packet == 1:
                reward += self.packet_reward_max
        elif received2 == PACKET_STATUS.LOST:
            penalty_multiplier = min(3, self.elapsed_time2 // self.node_send_interval)
            reward -= self.miss_penalty_max * penalty_multiplier

        return reward

    def step(self, action):
        if self.render_mode == "cv2":
            self.render()
        reward = 0
        self.steps += 1

        if action == 0:  # stand still
            pass  # No penalty for standing still
        elif action == 1:  # left
            if self.pos[0] > 0:
                self.pos = (self.pos[0] - 1, self.pos[1])
            else:
                reward -= 0.1  # Penalty for invalid move
        elif action == 2:  # right
            if self.pos[0] < self.max_distance_x - 1:  # arrays using on env has length of max_distance_x, max_distance_y
                self.pos = (self.pos[0] + 1, self.pos[1])
            else:
                reward -= 0.1  # Penalty for invalid move
        elif action == 3:  # up
            if self.pos[1] < self.max_distance_y - 1:
                self.pos = (self.pos[0], self.pos[1] + 1)
            else:
                reward -= 0.1  # Penalty for invalid move
        elif action == 4:  # down
            if self.pos[1] > 0:
                self.pos = (self.pos[0], self.pos[1] - 1)
            else:
                reward -= 0.1  # Penalty for invalid move
        else:
            print("Invalid action!")
        packet1 = self.node1.send(self.steps, self.pos)
        received1, rssi1, snir1 = packet1

        packet2 = self.node2.send(self.steps, self.pos)
        received2, rssi2, snir2 = packet2

        self.elapsed_time1 = min(self.max_steps, self.elapsed_time1 + 1)
        self.elapsed_time2 = min(self.max_steps, self.elapsed_time2 + 1)

        if received1 == PACKET_STATUS.RECEIVED:
            self.last_packet = 1
            self.total_received += 1
            self.total_received1 += 1
            self.elapsed_time1 = 0
            p1 = PacketReference(pos=self.pos, rssi=rssi1, snir=snir1, valid=True)
            if is_new_best_pacref(self.pacrefs1, p1):
                self.pacrefs1 = insert_best_pacref(self.pacrefs1, p1)
        elif received1 == PACKET_STATUS.LOST:
            self.total_misses += 1

        if received2 == PACKET_STATUS.RECEIVED:
            self.last_packet = 2
            self.total_received += 1
            self.total_received2 += 1
            self.elapsed_time2 = 0
            p2 = PacketReference(pos=self.pos, rssi=rssi2, snir=snir2, valid=True)
            if is_new_best_pacref(self.pacrefs2, p2):
                self.pacrefs2 = insert_best_pacref(self.pacrefs2, p2)
        elif received2 == PACKET_STATUS.LOST:
            self.total_misses += 1

        if self.stage == self.exploration_stage_index:
            reward += self.exploration_stage(packet1, packet2)
        elif self.stage == self.localization_stage_index:
            reward += self.localization_stage(packet1, packet2)
        elif self.stage == self.reception_stage_index:
            reward += self.reception_stage(packet1, packet2)

        state = self.get_state()
        reward = min(self.packet_reward_max * 2, reward)
        terminated = self.total_misses >= self.max_misses
        truncated = self.steps >= self.max_steps
        info = {'total_received': self.total_received,
                'total_misses': self.total_misses}

        self.total_reward += reward
        if self.steps % self.n_skips == 0:
            self.history_prev_actions.append(action)
            self.history_positions.append(self.pos)
        debug = False
        if debug:
            print(f"""
            Variables used to generate state:
                Position: (x: {self.pos[0]}, y: {self.pos[1]})
                Packet References Node 1:
                    Pref 1: {self.pacrefs1[0].__dict__}
                    Pref 2: {self.pacrefs1[1].__dict__}
                    Pref 3: {self.pacrefs1[2].__dict__}
                    Valid Count: {count_valid_packet_reference(self.pacrefs1)}
                Packet References Node 2:
                    Pref 1: {self.pacrefs2[0].__dict__}
                    Pref 2: {self.pacrefs2[1].__dict__}
                    Pref 3: {self.pacrefs2[2].__dict__}
                    Valid Count: {count_valid_packet_reference(self.pacrefs2)}
                Elapsed Time Node 1: {self.elapsed_time1}
                Elapsed Time Node 2: {self.elapsed_time2}
                Last Packet: {self.last_packet}
                Current Stage: {self.stage}
                Steps: {self.steps}
            """)



        return state, reward, terminated, truncated, info

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
        for pr in self.pacrefs1 + self.pacrefs2:
            if not pr.valid:
                continue
            # Draw a black dot at the packet ref's position
            pr_radius = 2  # Set radius of the black dot
            cv2.circle(frame, (offset_x + pr.pos[0], offset_y + pr.pos[1]), pr_radius, (0, 0, 0), -1)

            # Calculate the radius from RSSI (convert RSSI to distance)
            rssi_distance = self.transmission_model.inverse_generate_rssi(pr.rssi)  # Assume a method exists
            circle_radius = int(rssi_distance)  # Convert to integer for drawing
            cv2.circle(frame, (offset_x + pr.pos[0], offset_y + pr.pos[1]), circle_radius, (255, 0, 0),
                       1)  # Blue circle

        # Draw the maximum transmission distance circle
        cv2.circle(
            frame,  # img: the image to draw on
            center=(offset_x + x, offset_y + y),  # center: the circle's center
            radius=int(self.max_transmission_distance),  # radius: max transmission distance
            color=(255, 0, 0),  # color: blue (B, G, R)
            thickness=1  # thickness: 1 for outline
        )

        # Display the frame
        enlarged_image = cv2.resize(frame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        # Define stats list first
        stats = [
            f"Total received node1: {self.total_received1}",
            f"Total received node2: {self.total_received2}",
            f"Total misses: {self.total_misses}",
            f"Total reward: {self.total_reward:.3f}",
            f"Stage: {self.stage}"
        ]

        # Calculate text-related dimensions after defining stats
        line_height = 25  # Spacing between text lines
        num_lines = len(stats)  # Get the number of lines based on stats
        text_height = num_lines * line_height + 20  # Total height reserved for text, with padding

        # Create a canvas larger than the enlarged image to include space for text
        canvas = np.zeros((enlarged_image.shape[0] + text_height, enlarged_image.shape[1], 3), dtype=np.uint8)

        # Place the enlarged image on the canvas
        canvas[:enlarged_image.shape[0], :, :] = enlarged_image

        # Add text below the enlarged frame
        text_offset_x = 10
        text_offset_y = enlarged_image.shape[0] + 20  # Start below the enlarged image

        for i, text in enumerate(stats):
            cv2.putText(canvas, text, (text_offset_x, text_offset_y + (i * line_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the canvas instead of the original enlarged image
        cv2.imshow(self.window_name, canvas)

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
        return 0  # ploss_probability # TODO: reenable


class PACKET_STATUS(Enum):
    RECEIVED = 1
    LOST = 2
    NOT_SENT = 3


class Node:
    def __init__(self, transmission_model, pos=(10, 10), time_to_first_packet=10, send_interval=10, send_std=2.0
                 ):
        self.pos = pos
        self.last_packet_time = 0
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_interval = send_interval
        self.send_std = min(send_std, self.send_interval / 2)
        self.lower_bound_send_time = send_interval - (send_interval / 2)
        self.upper_bound_send_time = send_interval + (send_interval / 2)

        self.transmission_model = transmission_model

    def reset(self):
        self.time_of_next_packet = self.time_to_first_packet

    def generate_next_interval(self):
        # Generate a truncated normal value for the next time interval
        # a and b are calculated to truncate around the mean interval with some range
        a, b = (self.lower_bound_send_time - self.send_interval) / self.send_std, (
                self.upper_bound_send_time - self.send_interval) / self.send_std
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_std)
        return max(0, interval)

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
        # Pobability of success is based of distance
        if time >= self.time_of_next_packet:
            self.time_of_next_packet = max(time, time + self.send_interval)  #self.generate_next_interval())
            # f"time of next packet: {self.time_of_next_packet}" )
            is_received, rssi, snir = self.transmission(gpos)
            if is_received:
                # print(f"packet is_received ")
                return PACKET_STATUS.RECEIVED, rssi, snir
            else:
                return PACKET_STATUS.LOST, 0, 0
        return PACKET_STATUS.NOT_SENT, 0, 0
