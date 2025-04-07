import json
import math
import random
import warnings
from collections import deque
from enum import Enum
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import truncnorm

from utilities import jains_fairness_index, n_smallest_indices


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame."""

        super(FrameSkip, self).__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and return the last observation."""
        total_reward = 0.0
        done = False
        obs, trunc, info = None, None, None
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


def _generate_color_frame(grid):
    """Generate the color frame from the reception grid."""
    # Vectorized intensity calculation
    green = np.clip(grid * 255, 0, 255).astype(np.uint8)  # Green intensity
    red = np.clip((1 - grid) * 255, 0, 255).astype(np.uint8)  # Red intensity

    # Create the color frame using stacking (0 for blue, green and red as calculated)
    color_frame = np.dstack((np.zeros_like(red), green, red))

    # Transpose rows/columns for correct orientation
    color_frame = color_frame.transpose((1, 0, 2))  # Swap axes (height <-> width)
    return color_frame


def calculate_direction(x1, y1, x2, y2):
    # Calculate the differences in x and y
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the angle in radians
    angle_radians = math.atan2(dy, dx)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    if angle_degrees < 0:
        angle_degrees = 360 + angle_degrees

    return angle_degrees


def schedule_first_packets(send_intervals, initial_delay=0):
    """
    Given a list of send intervals, this function schedules the first packet times such that
    the minimum gap between events is as large as possible.
    - send_intervals (list of int): List of send intervals for each node.
    - initial_delay (int): Integer to add to each integer in the returned list.
    Returns:
    - first_packets (list of int): List of first packet times for each node.
    """
    # Space out packet times evenly up to min_interval
    step = min(send_intervals) / len(send_intervals)
    first_packets = [int(i * step) + initial_delay for i in range(len(send_intervals))]
    random.shuffle(first_packets)
    return first_packets


class TwoDEnv(gym.Env):
    def __init__(self, **kwargs):
        super(TwoDEnv, self).__init__()

        # Configurable parameters with type hints and defaults
        self.render_mode: str = kwargs.pop("render_mode", "none")
        self.do_logging: bool = kwargs.pop("do_logging", False)
        self.log_file: Optional[str] = kwargs.pop("log_file", None)
        self.max_steps: int = kwargs.pop("max_steps", int(86400 / 4))
        self.number_of_model_nodes: int = kwargs.pop("number_of_model_nodes", 4)
        self.number_of_sim_nodes: int = kwargs.pop("number_of_sim_nodes", 20)
        self.use_node_index_sorting: bool = kwargs.pop("use_node_index_sorting", True)
        self.model_use_node_priority: bool = kwargs.pop("model_use_node_priority", True)

        # Warn about any remaining unused kwargs
        if kwargs:
            warnings.warn(f"Unused kwargs in TwoDEnv constructor: {list(kwargs.keys())}", stacklevel=2)

        # --- Simulation constants ---
        self.max_send_interval = 8000

        # Reward shaping
        self.packet_reward_max = 1.0
        self.packet_reward_min = 0.0
        self.fairness_reward = 0.0625
        self.pos_reward_max = 0.0125 / 2
        self.pos_reward_min = -self.pos_reward_max
        self.good_action_reward = self.pos_reward_max / 4
        self.miss_penalty_max = 0.5
        self.miss_penalty_min = self.miss_penalty_max / 2

        # --- Environment space & mobility setup ---
        self.steps = 0
        self.max_distance = 300
        self.max_distance_x = int(self.max_distance)
        self.max_distance_y = int(self.max_distance)
        self.max_cross_distance = math.dist((0, 0), (self.max_distance_x, self.max_distance_y))
        self.pos = (
            random.randint(0, self.max_distance_x),
            random.randint(0, self.max_distance_y)
        )
        unscaled_speed = 11
        unscaled_max_distance = 3000
        self.scaled_speed = unscaled_speed * (self.max_distance / unscaled_max_distance)

        # --- Transmission behavior ---
        self.ploss_scale = 50
        self.node_max_transmission_distance = 40
        self.base_send_interval = 0
        self.send_std = 5
        self.send_intervals = [0] * self.number_of_sim_nodes
        self.first_packets = [0] * self.number_of_sim_nodes

        # --- Node definitions ---
        # Placeholder nodes that will be overwritten in reset()
        self.nodes = [
            Node((0, 0),
                 TransmissionModel(
                     max_transmission_distance=self.node_max_transmission_distance,
                     ploss_scale=self.ploss_scale),
                 time_to_first_packet=self.first_packets[i],
                 send_interval=self.send_intervals[i],
                 send_std=self.send_std)
            for i in range(self.number_of_sim_nodes)
        ]

        # --- Runtime state tracking ---
        self.elapsed_times = [0] * self.number_of_sim_nodes
        self.loss_counts = [0] * self.number_of_sim_nodes
        self.expected_send_time = self.first_packets.copy()
        self.received_per_node = [0] * self.number_of_sim_nodes
        self.misses_per_node = [0] * self.number_of_sim_nodes

        self.total_reward = 0
        self.total_misses = 0
        self.total_received = 0
        self.fairness = 0.0

        # --- Observation & action space ---
        self.num_discrete_actions = 5
        self.action_space = spaces.Discrete(self.num_discrete_actions, start=0)

        model_input_size = self.number_of_model_nodes * (4 if self.model_use_node_priority else 3)
        self.observation_space = spaces.Box(
            low=np.zeros(model_input_size, dtype=np.float32),
            high=np.ones(model_input_size, dtype=np.float32)
        )

        # --- Rendering config ---
        self.width = self.max_distance_x + 20
        self.height = self.max_distance_y + 20
        self.offset_x = (self.width - self.max_distance_x) // 2
        self.offset_y = (self.height - self.max_distance_y) // 2
        self.point_radius = 1
        self.point_color = (255, 255, 255)
        self.line_color = (255, 0, 0)
        self.window_name = None
        self.reception_grid = None
        self.background_frame = None

        # --- Logging ---
        self.log_dynamic_data = []

    def get_random_node_positions(self, num_positions: int = 4, min_dist: float = 20) -> list[tuple[int, int]]:
        """Generate `num_positions` random node coordinates with at least `min_dist` spacing."""
        positions = []
        attempts = 0
        while len(positions) < num_positions:
            new_pos = (
                random.randint(0, self.max_distance_x),
                random.randint(0, self.max_distance_y)
            )
            if all(math.dist(new_pos, pos) >= min_dist for pos in positions):
                positions.append(new_pos)
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(
                    "Too many attempts to generate non-overlapping positions. Try reducing num_positions or min_dist.")
        random.shuffle(positions)
        return positions

    def _compute_reception_grid(self) -> np.ndarray:
        """Compute a 2D reception probability grid using vectorized operations."""
        width = self.max_distance_x + 1
        height = self.max_distance_y + 1
        grid = np.zeros((width, height), dtype=np.float32)

        # Generate coordinate grids (X and Y have shape [width, height])
        xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

        for node in self.nodes:
            dx = xx - node.pos[0]
            dy = yy - node.pos[1]
            distance = np.hypot(dx, dy)

            reception = node.transmission_model.get_reception_prob(distance)
            reception[distance > node.transmission_model.max_transmission_distance] = 0.0

            grid += reception

        # Clip to max 1.0 per grid cell
        np.clip(grid, 0.0, 1.0, out=grid)
        return grid

    def calculate_node_priority(self, packets_received: int) -> float:
        """Return node priority: higher if it has received fewer packets."""
        return 1.0 - packets_received / self.total_received if self.total_received else 1.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to an initial state and return the initial observation."""
        self.steps = 0
        self.total_reward = 0
        self.total_received = 0
        self.total_misses = 0
        self.fairness = 0.0

        # Randomize gateway position
        self.pos = (
            random.randint(0, self.max_distance_x),
            random.randint(0, self.max_distance_y)
        )

        # Generate random node positions and transmission intervals
        node_positions = self.get_random_node_positions(
            num_positions=self.number_of_sim_nodes,
            min_dist=5  # can parameterize later
        )
        self.base_send_interval = random.choice([2000, 2500, 3000, 3500])
        self.send_intervals = [
            self.base_send_interval * random.choice([1, 2])
            for _ in range(self.number_of_sim_nodes)
        ]
        random.shuffle(self.send_intervals)

        self.first_packets = schedule_first_packets(
            self.send_intervals, initial_delay=400
        )
        self.expected_send_time = self.first_packets.copy()

        # Reset per-node stats and configure each node
        for i, node in enumerate(self.nodes):
            node.configure(
                pos=node_positions[i],
                send_interval=self.send_intervals[i],
                time_to_first_packet=self.first_packets[i]
            )
            self.elapsed_times[i] = 0
            self.loss_counts[i] = 0
            self.received_per_node[i] = 0
            self.misses_per_node[i] = 0

        # Reset render state
        if self.render_mode == "cv2":
            self.window_name = "RL Animation"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.reception_grid = self._compute_reception_grid()
            self.background_frame = _generate_color_frame(self.reception_grid)

        # Reset logs and return initial state
        self.log_dynamic_data.clear()
        return np.array(self.get_state(), dtype=np.float32), {}

    def select_node_indices_for_state(self):
        if self.use_node_index_sorting:
            return n_smallest_indices(self.expected_send_time, self.number_of_model_nodes)
        else:
            return list(range(self.number_of_model_nodes))

    def get_state(self):
        indices = self.select_node_indices_for_state()

        normalized_expected = [
            (self.expected_send_time[i] - self.steps) / self.max_send_interval
            for i in indices
        ]
        normalized_distances = [
            math.dist(self.pos, self.nodes[i].pos) / self.max_cross_distance
            for i in indices
        ]
        normalized_directions = [
            calculate_direction(*self.pos, *self.nodes[i].pos) / 360.0
            for i in indices
        ]
        normalized_priority = [
            self.calculate_node_priority(self.received_per_node[i])
            for i in indices
        ]

        state = normalized_expected + normalized_distances + normalized_directions
        if self.model_use_node_priority:
            state += normalized_priority

        return state

    def get_packet_reward(self, sending_node_idx: int):
        node = self.nodes[sending_node_idx]
        distance = math.dist(self.pos, node.pos)
        reception_prob = node.transmission_model.get_reception_prob(distance)
        priority_weight = 0.5 + self.calculate_node_priority(sending_node_idx) / 2.0
        return self.packet_reward_max * reception_prob * priority_weight

    def get_pos_reward(self, node: 'Node'):
        distance = math.dist(self.pos, node.pos)
        scaled = distance / self.max_cross_distance
        k = 6
        reward = self.pos_reward_max * math.exp(-k * scaled)
        return max(self.pos_reward_min, min(self.pos_reward_max, reward))

    def get_next_sending_node_index(self):
        return min(
            range(len(self.nodes)),
            key=lambda i: self.nodes[i].time_of_next_packet
        )

    def get_next_expected_sending_node_index(self):
        return min(
            range(len(self.expected_send_time)),
            key=lambda i: self.expected_send_time[i]
        )

    def get_miss_penalty(self, node):
        distance = math.dist(self.pos, node.pos)
        fail_prob = 1.0 - node.transmission_model.get_reception_prob(distance)
        scaled = distance / self.max_cross_distance
        weight = (scaled + fail_prob) / 2.0
        penalty = self.miss_penalty_min + weight * (self.miss_penalty_max - self.miss_penalty_min)
        return -min(self.miss_penalty_max, max(self.miss_penalty_min, penalty))

    def get_good_action_reward(self, distance_before, distance_after):
        return self.good_action_reward if distance_after < distance_before else -2 * self.good_action_reward

    def step(self, action):
        reward = 0
        self.steps += 1

        for i, node in enumerate(self.nodes):
            assert self.expected_send_time[i] >= self.steps, \
                f"Node {i} expected send time inconsistency: {self.expected_send_time[i]} < {self.steps}"

        idx_next = self.get_next_expected_sending_node_index()
        node = self.nodes[idx_next]
        distance_before = math.dist(self.pos, node.pos)

        # Movement
        x, y = self.pos
        if action == 0:  # stay still
            pass
        elif action == 1:  # left
            x = max(x - self.scaled_speed, 0)
        elif action == 2:  # right
            x = min(x + self.scaled_speed, self.max_distance_x)
        elif action == 3:  # up
            y = min(y + self.scaled_speed, self.max_distance_y)
        elif action == 4:  # down
            y = max(y - self.scaled_speed, 0)

        self.pos = (x, y)

        reward += self.get_pos_reward(node)
        distance_after = math.dist(self.pos, node.pos)
        reward += self.get_good_action_reward(distance_before, distance_after)

        # Transmissions
        transmission_flags = [True] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            result = node.send(self.steps, self.pos)
            self.elapsed_times[i] = min(self.max_steps, self.elapsed_times[i] + 1)

            if result == PACKET_STATUS.NOT_SENT:
                transmission_flags[i] = False
            elif result == PACKET_STATUS.RECEIVED:
                reward += self.get_packet_reward(i)
                self.total_received += 1
                self.received_per_node[i] += 1
                self.elapsed_times[i] = 0
                self.loss_counts[i] = 0
                self.expected_send_time[i] += self.send_intervals[i] + self.send_std
                sent_total = [self.received_per_node[j] + self.misses_per_node[j] for j in range(len(self.nodes))]
                self.fairness = jains_fairness_index(self.received_per_node, sent_total)
            elif result == PACKET_STATUS.LOST:
                self.total_misses += 1
                self.misses_per_node[i] += 1
                self.loss_counts[i] += 1
                reward += self.get_miss_penalty(node)

        # Update expected send time
        for i, t in enumerate(self.expected_send_time):
            if self.steps >= t:
                self.expected_send_time[i] += self.send_intervals[i] + self.send_std

        # Done?
        terminated = False
        truncated = self.steps >= self.max_steps
        done = truncated
        self.total_reward += reward
        info = {
            'total_received': self.total_received,
            'total_misses': self.total_misses,
            'fairness': self.fairness
        }

        if self.do_logging:
            self.log_step(transmissions_per_node=transmission_flags)
            if done:
                self.log_done()

        if self.render_mode == "cv2":
            self.render()

        return np.array(self.get_state(), dtype=np.float32), reward, terminated, done, info

    def log_step(self, transmissions_per_node):
        """Logs a single step's data into the buffer."""
        packets_sent_per_node = [
            rx + tx for rx, tx in zip(self.received_per_node, self.misses_per_node)
        ]
        node_distances = [math.dist(self.pos, node.pos) for node in self.nodes]

        self.log_dynamic_data.append({
            "gw_pos_x": self.pos[0],
            "gw_pos_y": self.pos[1],
            "step_time": self.steps,
            "packets_received": self.total_received,
            "packets_sent": self.total_received + self.total_misses,
            "transmissions_per_node": transmissions_per_node[:],  # shallow copy
            "packets_received_per_node": self.received_per_node[:],
            "packets_missed_per_node": self.misses_per_node[:],
            "packets_sent_per_node": packets_sent_per_node,
            "node_distances": node_distances
        })

    def log_done(self):
        """Writes episode summary to JSON file."""
        static_data = {
            "number_of_nodes": len(self.nodes),
            "node_positions_x": [node.pos[0] for node in self.nodes],
            "node_positions_y": [node.pos[1] for node in self.nodes],
            "send_intervals": self.send_intervals[:],
            "max_distance_x": self.max_distance_x,
            "max_distance_y": self.max_distance_y
        }

        episode_data = {
            "static": static_data,
            "dynamic": self.log_dynamic_data
        }

        with open(self.log_file, 'w') as f:
            json.dump(episode_data, f, indent=4)

    def render(self):
        """Render the environment with reception-based background and dynamic elements."""

        # High-resolution scale factor
        scale_factor = 2  # Adjust this factor to increase resolution
        high_res_width = int(self.width * scale_factor)
        high_res_height = int(self.height * scale_factor)

        # Calculate padding for top, left, bottom, and right
        pad_top, pad_left = int(self.offset_y) * scale_factor, int(self.offset_x) * scale_factor
        pad_bottom, pad_right = pad_top, pad_left

        # Scale up the background frame directly
        scaled_background_frame = cv2.resize(self.background_frame, (0, 0), fx=scale_factor, fy=scale_factor,
                                             interpolation=cv2.INTER_LINEAR)
        # cv2.resize(frame, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)

        # Add padding to the scaled color frame using np.pad
        padded_color_frame = np.pad(
            scaled_background_frame,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant', constant_values=0
        )

        # Create a high-resolution frame (background)
        frame = np.zeros((high_res_height, high_res_width, 3), dtype=np.uint8)
        frame[
        :padded_color_frame.shape[0],
        :padded_color_frame.shape[1]
        ] = padded_color_frame[
            :high_res_height,
            :high_res_width
            ]

        # Draw nodes and their transmission circles
        for i, node in enumerate(self.nodes):
            center = (
                int((node.pos[0] + self.offset_x) * scale_factor), int((node.pos[1] + self.offset_y) * scale_factor))
            radius = int(node.transmission_model.max_transmission_distance * scale_factor)
            cv2.circle(frame, center=center, radius=radius, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        # draw text on top of the blue circles
        for i, node in enumerate(self.nodes):
            center = (
                int((node.pos[0] + self.offset_x) * scale_factor), int((node.pos[1] + self.offset_y) * scale_factor))
            cv2.putText(frame, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2,
                        lineType=cv2.LINE_AA)

        # Draw the moving point as a filled square
        gw_x = int((self.pos[0] + self.offset_x) * scale_factor)
        gw_y = int((self.pos[1] + self.offset_y) * scale_factor)
        gw_box_size = int(2 * scale_factor)
        cv2.rectangle(frame, pt1=(gw_x - gw_box_size, gw_y - gw_box_size),
                      pt2=(gw_x + gw_box_size, gw_y + gw_box_size),
                      color=self.point_color, thickness=-1)  # FILL with -1

        # Render text data and stats
        canvas = self.render_text_data(frame)

        # Enable resizable window and update the content dynamically
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(2)

    def render_text_data(self, image):
        """Render static stats and per-node data below the environment frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness = 1, 2
        text_color = (255, 255, 255)
        stat_line_height = 35
        node_line_height = 30

        # --- Static summary stats ---
        stats = [
            f"Total received: {self.total_received}",
            f"Total misses: {self.total_misses}",
            f"Total reward: {self.total_reward:.3f}",
            f"Time: {self.steps} | {self.max_steps}"
        ]

        # --- Calculate canvas height with extra space for stats ---
        total_stat_height = len(stats) * stat_line_height + 40
        canvas = np.zeros((image.shape[0] + total_stat_height, image.shape[1], 3), dtype=np.uint8)
        canvas[:image.shape[0]] = image  # Copy the frame into the top part

        # --- Draw stats ---
        for i, text in enumerate(stats):
            pos = (10, image.shape[0] + 20 + i * stat_line_height)
            cv2.putText(canvas, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # --- Draw node-specific info ---
        base_y = image.shape[0] + 20
        base_x = 400
        for i, node in enumerate(self.nodes):
            y = base_y + i * node_line_height
            time_left = round(node.time_of_next_packet - self.steps)
            send_interval = self.send_intervals[i]

            texts = [
                (f"{i}:", base_x),
                (f"{time_left}", base_x + 30),
                (f" | {send_interval}", base_x + 110),
            ]

            for text, x in texts:
                cv2.putText(canvas, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return canvas

    def close(self):
        cv2.destroyAllWindows()


class TransmissionModel:
    def __init__(self, max_transmission_distance=50, ploss_scale=300,
                 probability_modifier=1):
        self.max_transmission_distance = max_transmission_distance
        self.ploss_scale = ploss_scale
        self.probability_modifier = probability_modifier

    def get_reception_prob(self, distance):
        # P(Reception) = probability of receiving packet.
        # Probability of receiving packet decreases with distance
        # Works for both scalar and numpy arrays
        prob = np.exp(-distance / self.ploss_scale) * self.probability_modifier
        prob = np.where(distance <= self.max_transmission_distance, prob, 0.0)
        return prob

    def is_transmission_success(self, distance):
        receive_choice = self.get_reception_prob(distance) > np.random.rand()
        return receive_choice


class PACKET_STATUS(Enum):
    RECEIVED = 1
    LOST = 2
    NOT_SENT = 3


class Node:
    def __init__(self, pos: tuple[int, int], transmission_model: TransmissionModel, time_to_first_packet: int,
                 send_interval: int, send_std=10):
        self.pos = pos
        self.transmission_model = transmission_model

        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_std = send_std  # standard deviation

        self.send_interval = None
        self.lower_bound_send_time = None
        self.upper_bound_send_time = None
        self.set_send_interval(send_interval)

    def reset(self):
        self.time_of_next_packet = self.time_to_first_packet

    def set_send_interval(self, send_interval):
        self.send_interval = send_interval
        interval_bound_scale = 0.01
        self.lower_bound_send_time = send_interval - send_interval * interval_bound_scale
        self.upper_bound_send_time = send_interval + send_interval * interval_bound_scale

    def configure(self, pos, send_interval, time_to_first_packet):
        self.pos = pos
        self.set_send_interval(send_interval)
        self.time_to_first_packet = time_to_first_packet
        self.reset()

    def generate_next_interval(self):
        # Generate a truncated normal value for the next time interval
        # a and b are calculated to truncate around the mean interval with some range
        a, b = (self.lower_bound_send_time - self.send_interval) / self.send_std, (
                self.upper_bound_send_time - self.send_interval) / self.send_std
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_std)
        return interval

    def transmission(self, gpos):
        distance = math.dist(self.pos, gpos)
        if self.transmission_model.is_transmission_success(distance):
            return True
        return False

    def send(self, time, gpos):
        # Decides whether a packet should be send and if it gets lost
        # Pobability of success is based of distance
        if time >= self.time_of_next_packet:
            self.time_of_next_packet = time + self.generate_next_interval()
            is_received = self.transmission(gpos)
            if is_received:
                return PACKET_STATUS.RECEIVED
            else:
                return PACKET_STATUS.LOST
        return PACKET_STATUS.NOT_SENT
