import json
import math
import random
from collections import deque
from enum import Enum

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import truncnorm


# Define a custom FrameSkip wrapper
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


def _jains_fairness_index(delivery_rates):
    n = len(delivery_rates)
    temp = sum([(x ** 2) for x in delivery_rates])
    if temp == 0:
        return 0.0
    jains_index = sum(delivery_rates) ** 2 / (n * temp)
    return jains_index


def jains_fairness_index(received_per_node: list[int], sent_per_node: list[int]):
    if len(received_per_node) != len(sent_per_node):
        raise ValueError("Error in 'jains_fairness_index'! "
                         "len('received_per_node') must match length of len('misses_per_node')")
    length = max(len(received_per_node), len(sent_per_node))
    delivery_rates = [
        0 if (sent_per_node[i]) == 0 else received_per_node[i] / (sent_per_node[i])
        for i in range(length)
    ]
    return _jains_fairness_index(delivery_rates)


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
    first_packets = [int(min(send_intervals) * fraction / len(send_intervals)) + initial_delay
                     for fraction in range(0, len(send_intervals))]
    min_value_fp = min(first_packets)
    first_packets = [(fp_t - min_value_fp) + initial_delay for fp_t in first_packets]
    random.shuffle(first_packets)
    return first_packets


class TwoDEnv(gym.Env):
    def __init__(self, render_mode="none", do_logging=False, log_file=None,
                 use_deterministic_transmissions=False):
        super(TwoDEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, either -1, 0, or +1
        self.num_discrete_actions = 5
        self.action_space = spaces.Discrete(self.num_discrete_actions, start=0) 

        self.recent_packets_length = 1

        self.render_mode = render_mode
        # Environment.pos
        self.steps = 0
        self.max_steps = 40000  # Maximum steps per episode

        # Environment state
        # Scaled reward values preserving relative ratios
        self.pos_reward_max = 0.0125
        self.pos_reward_min = -self.pos_reward_max
        self.good_action_reward = self.pos_reward_max / 4
        self.miss_penalty_max = 0.5
        self.miss_penalty_min = self.miss_penalty_max / 2
        self.packet_reward_max = 1.0  # TODO: consider increasing this
        self.packet_reward_min = 0.0
        self.fairness_reward = 0.0625

        unscaled_speed = 11  # meter per second based on this article http://unmannedcargo.org/chinese-supermarket-delivery-drone/
        unscaled_max_distance = 3000  # meters
        self.max_distance = 300  # env grid size
        self.scaled_speed = unscaled_speed * (self.max_distance / unscaled_max_distance)
        self.max_distance_x = int(self.max_distance)  # scaled by speed
        self.max_distance_y = int(self.max_distance)
        self.max_cross_distance = math.dist((0, 0), (self.max_distance_x, self.max_distance_y))
        self.pos = (random.randint(0, self.max_distance_x), random.randint(0, self.max_distance_y))
        self.steps = 0
        self.ploss_scale = 50  # adjusts the dropoff of transmission probability by distance
        self.node_max_transmission_distance = 40
        node_pos = [
            (self.max_distance_x // 6, self.max_distance_y // 6),
            (self.max_distance_x - (self.max_distance_x // 6), self.max_distance_y - (self.max_distance_y // 6)),
            (self.max_distance_x // 6, self.max_distance_y - (self.max_distance_y // 6)),
            (self.max_distance_x - (self.max_distance_x // 6), self.max_distance_y // 6)
        ]


        self.min_send_interval = 1500
        self.max_send_interval = 3000
        self.base_send_interval = random.choice([1500, 1750, 2000])  # 1500  #  # TODO: try with random send intervals?
        self.send_intervals = [ random.randint(self.min_send_interval, self.max_send_interval),
                                random.randint(self.min_send_interval, self.max_send_interval),
                                random.randint(self.min_send_interval, self.max_send_interval),
                                random.randint(self.min_send_interval, self.max_send_interval)]
        random.shuffle(self.send_intervals)
        self.first_packets = schedule_first_packets(self.send_intervals, initial_delay=600)

        self.send_std = 5 

        # random.shuffle(self.first_packets)
        self.nodes = [
            Node(node_pos[i],
                 TransmissionModel(max_transmission_distance=self.node_max_transmission_distance,
                                   ploss_scale=self.ploss_scale,
                                   use_deterministic_transmissions=use_deterministic_transmissions),
                 time_to_first_packet=self.first_packets[i],
                 send_interval=self.send_intervals[i],
                 use_deterministic_transmissions=use_deterministic_transmissions,
                 send_std=self.send_std)
            for i in range(len(node_pos))
        ]

        self.elapsed_times = [0] * len(self.nodes)
        self.loss_counts = [0] * len(self.nodes)
        self.expected_send_time = self.first_packets.copy()

        self.expected_max_packets_sent = self.max_steps // min(self.send_intervals)
        self.total_reward = 0
        self.total_misses = 0
        self.total_received = 0
        self.fairness = 0.0
        self.received_per_node = [0] * len(self.nodes)
        self.misses_per_node = [0] * len(self.nodes)
        self.recent_packets = deque([-1] * self.recent_packets_length, maxlen=self.recent_packets_length)
        # Observation_space =
        #                     expected time until sending                      |n
        #                     distance from gw to each node                    |n
        #                     direction from gw to each node                   |n
        #                     m onehot recent_packets (initialized to -1)      |m * n
        #                                                                      |3n + (m*n)

        self.observation_space = spaces.Box(
            low=np.array(
                [0] * (
                        len(self.nodes) +
                        len(self.nodes) +
                        len(self.nodes) +
                        self.recent_packets_length * (len(self.nodes))  # One-hot encoded recent_packets)
                )
                , dtype=np.float32),
            high=np.array(
                [1] * (
                        len(self.nodes) +
                        len(self.nodes) +
                        len(self.nodes) +
                        self.recent_packets_length * (len(self.nodes))  # One-hot encoded recent_packets)
                )
                , dtype=np.float32))

        # rendering attributes
        self.width, self.height = self.max_distance_x + 50, self.max_distance_y + 50  # Size of the window
        self.offset_x = int((self.width - self.max_distance_x) / 2)
        self.offset_y = int((self.height - self.max_distance_y) / 2)
        self.point_radius = 1
        self.point_color = (255, 255, 255)  # Red color
        self.line_color = (255, 0, 0)  # Blue color

        self.window_name = None
        self.reception_grid = None
        self.background_frame = None

        # logging attributes
        self.do_logging = do_logging
        self.log_file = log_file
        self.log_dynamic_data = []  # Store logs before writing to the file

    def get_random_node_positions(self, num_positions=4, min_dist=20):
        positions = []
        for _ in range(num_positions):
            while True:
                new_pos = (
                    random.randint(0, self.max_distance_x),
                    random.randint(0, self.max_distance_y)
                )
                if all(math.dist(new_pos, pos) >= min_dist for pos in positions):
                    positions.append(new_pos)
                    break
        random.shuffle(positions)
        return positions

    def _compute_reception_grid(self):
        """ Compute the reception grid based on node positions and transmission range. """
        # grid of 0 to max_distance y & x, max_distance inclusive
        grid = np.zeros((self.max_distance_y + 1, self.max_distance_x + 1), dtype=np.float32)

        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                reception = 0.0
                for node in self.nodes:
                    # Calculate distance to the node
                    distance = np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2)
                    if distance <= node.transmission_model.max_transmission_distance:
                        # Reception is a function of distance
                        reception += node.transmission_model.get_reception_prob(distance)

                # Normalize to between 0 and 1
                grid[x, y] = min(reception, 1.0)

        return grid

    def reset(self, seed=None, options=None):
        self.recent_packets = deque([-1] * self.recent_packets_length, maxlen=self.recent_packets_length)
        self.total_misses = 0
        self.pos = (random.randint(0, self.max_distance_x), random.randint(0, self.max_distance_y))
        positions = self.get_random_node_positions(num_positions=len(self.nodes),
                                                   min_dist=2*self.node_max_transmission_distance)
        self.base_send_interval = random.choice([1500, 1750, 2000])  # 1500  #  # TODO: try with random send intervals?
        self.send_intervals = [ random.randint(self.min_send_interval, self.max_send_interval),
                        random.randint(self.min_send_interval, self.max_send_interval),
                        random.randint(self.min_send_interval, self.max_send_interval),
                        random.randint(self.min_send_interval, self.max_send_interval)]
        random.shuffle(self.send_intervals)
        self.first_packets = schedule_first_packets(self.send_intervals, initial_delay=600)
        for i in range(len(self.nodes)):
            self.nodes[i].pos = positions[i]
            self.nodes[i].set_send_interval(self.send_intervals[i])
            self.nodes[i].time_to_first_packet = self.first_packets[i]
            self.nodes[i].reset()
            self.elapsed_times[i] = 0
            self.loss_counts[i] = 0
            self.received_per_node[i] = 0
            self.misses_per_node[i] = 0

        self.expected_send_time = self.first_packets.copy()

        self.steps = 0
        self.total_reward = 0
        self.total_received = 0
        self.fairness = 0.0

        if self.render_mode == "cv2":
            self.window_name = "RL Animation"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Precompute the reception-based background
            self.reception_grid = self._compute_reception_grid()
            self.background_frame = _generate_color_frame(self.reception_grid)

        self.log_dynamic_data = []
        state = self.get_state()
        return np.array(state, dtype=np.float32), {}

    def get_state(self):
        # One-hot encode and normalize previous actions

        onehot_encoded_recent_packets = []
        for recent_packet in self.recent_packets:
            onehot = [1.0 if i == recent_packet else 0.0 for i in
                      range(len(self.nodes))]  # Include -1 for "invalid"
            onehot_encoded_recent_packets.extend(onehot)

        # Other normalized components
        normalized_expected_send_time = [(expected_time - self.steps) / self.max_steps for expected_time in
                                         self.expected_send_time]

        normalized_node_distances = [
            math.dist(self.pos, node.pos) / self.max_cross_distance
            for node in self.nodes
        ]

        normalized_node_directions = [
            calculate_direction(*self.pos, *node.pos) / 360
            for node in self.nodes
        ]
        # Combine all normalized and one-hot encoded components into the state
        state = (
                normalized_expected_send_time +
                normalized_node_distances +
                normalized_node_directions +
                onehot_encoded_recent_packets
                )
        return state

    def get_packet_reward(self, sending_node: 'Node'):
        distance = math.dist(self.pos, sending_node.pos)
        reward = self.packet_reward_max * sending_node.transmission_model.get_reception_prob(distance)
        return reward

    def get_pos_reward(self, node: 'Node', elapsed_time: float):
        distance = math.dist(self.pos, node.pos)

        failure_probability = 1 - node.transmission_model.get_reception_prob(distance)
        scaled_distance = distance / self.max_cross_distance

        # weight = (scaled_distance + failure_probability) / 2 #TODO: try enable, but change min reward
        # Return reward based on scaled distance between a min and max reward
        # Exponential reward calculation
        k = 6  # Adjust this scaling factor to control the sharpness of the exponential decay
        reward = self.pos_reward_max * math.exp(-k * scaled_distance)

        # reward = self.pos_reward_max - (scaled_distance * (self.pos_reward_max - self.pos_reward_min))

        # Ensure reward is within bounds in case of rounding errors
        reward = max(self.pos_reward_min, min(self.pos_reward_max, reward))

        # if distance < self.node_max_transmission_distance:
        #    reward *= 2
        return reward

    def get_next_sending_node_index(self):
        idx_min = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].time_of_next_packet < self.nodes[idx_min].time_of_next_packet:
                idx_min = i

        return idx_min

    def get_next_expected_sending_node_index(self):
        idx_min = 0
        for i in range(len(self.expected_send_time)):
            if self.expected_send_time[i] < self.expected_send_time[idx_min]:
                idx_min = i

        return idx_min

    def get_miss_penalty(self, node: 'Node'):
        distance = math.dist(self.pos, node.pos)
        failure_probability = 1 - node.transmission_model.get_reception_prob(distance)
        scaled_distance = distance / self.max_cross_distance
        weight = (scaled_distance + failure_probability) / 2

        penalty = self.miss_penalty_min + weight * (self.miss_penalty_max - self.miss_penalty_min)

        # Ensure reward is within bounds in case of rounding errors
        penalty = min(self.miss_penalty_max, max(self.miss_penalty_min, penalty))
        return -penalty


    def get_good_action_reward(self, distance_prior_action, distance_after_action):
        is_good_action = distance_after_action < distance_prior_action
        if is_good_action:
            return self.good_action_reward
        else:
            return -self.good_action_reward * 2

    def step(self, action):
        if self.render_mode == "cv2":
            self.render()
        reward = 0
        self.steps += 1

        for i, node in enumerate(self.nodes):
            assert self.expected_send_time[i] >= self.steps, \
                f"Node {i} expected send time inconsistency: {self.expected_send_time[i]} < {self.steps}"

        idx_next_sending_node = self.get_next_expected_sending_node_index()  # self.get_next_sending_node_index()

        distance_prior_action = math.dist(self.pos, self.nodes[idx_next_sending_node].pos)

        if action == 0:  # stand still
            # nothing
            pass
        elif action == 1:  # left
            self.pos = max((self.pos[0] - self.scaled_speed), 0), self.pos[1]
        elif action == 2:  # right
            self.pos = min(self.pos[0] + self.scaled_speed, self.max_distance_x), self.pos[1]
        elif action == 3:  # up
            self.pos = self.pos[0], min(self.pos[1] + self.scaled_speed, self.max_distance_y)
        elif action == 4:  # down
            self.pos = self.pos[0], max((self.pos[1] - self.scaled_speed), 0)

        reward += self.get_pos_reward(self.nodes[idx_next_sending_node],
                                      self.elapsed_times[idx_next_sending_node])  # TODO: try disable
        distance_after_action = math.dist(self.pos, self.nodes[idx_next_sending_node].pos)
        reward += self.get_good_action_reward(distance_prior_action, distance_after_action)  # TODO: try enable

        # Track transmissions for each node
        transmission_occurred_per_node = [True] * len(self.nodes)
        for i in range(len(self.nodes)):
            received, rssi, snir = self.nodes[i].send(self.steps, self.pos)
            self.elapsed_times[i] = min(self.max_steps, self.elapsed_times[i] + 1)
            if received == PACKET_STATUS.NOT_SENT:
                transmission_occurred_per_node[i] = False
            elif received == PACKET_STATUS.RECEIVED:
                reward += self.get_packet_reward(self.nodes[i])

                self.total_received += 1
                self.received_per_node[i] += 1
                self.elapsed_times[i] = 0
                self.loss_counts[i] = 0
                self.recent_packets.append(i)

                self.expected_send_time[i] += self.send_intervals[i] + self.send_std
                sent_per_node = [self.received_per_node[i] + self.misses_per_node[i]
                                 for i in range(len(self.misses_per_node))]
                self.fairness = jains_fairness_index(self.received_per_node, sent_per_node)
            elif received == PACKET_STATUS.LOST:
                self.total_misses += 1
                self.misses_per_node[i] += 1
                self.loss_counts[i] += 1
                reward += self.get_miss_penalty(self.nodes[i])  # * self.loss_counts[i]

        # update send time approximation
        for i in range(len(self.expected_send_time)):
            if self.steps >= self.expected_send_time[i]:
                self.expected_send_time[i] += self.send_intervals[i] + self.send_std

        terminated = False  # self.total_misses >= 20  # TODO: Disable this
        truncated = self.steps >= self.max_steps
        done = truncated or terminated
        self.total_reward += reward
        state = self.get_state()
        info = {'total_received': self.total_received,
                'total_misses': self.total_misses,
                'fairness': self.fairness}

        # Add logging for each step
        if self.do_logging:
            self.log_step(
                transmissions_per_node=transmission_occurred_per_node
            )
            if done:
                self.log_done()

        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    def log_done(self):
        episode_data = {
            "static": {
                "node_positions_x":
                    [node.pos[0] for node in self.nodes],
                "node_positions_y":
                    [node.pos[1] for node in self.nodes],
                "send_intervals": self.send_intervals,
                "max_distance_x": self.max_distance_x,
                "max_distance_y": self.max_distance_y
            },
            "dynamic": self.log_dynamic_data
        }
        # Write to JSON file if the episode ends
        with open(self.log_file, 'w') as file:
            json.dump(episode_data, file, indent=4)

    def log_step(self, transmissions_per_node):
        """
        Logs a single step's data into the log buffer, including details for each node.
        Args:
            transmissions_per_node: List of booleans indicating if transmission occurred for each node.
        """
        # Compute the distance from the gateway to each node
        node_distances = [math.dist(self.pos, node.pos) for node in self.nodes]

        # Log entry for this step
        log_entry = {
            "gw_pos_x": self.pos[0],
            "gw_pos_y": self.pos[1],
            "step_time": self.steps,
            "packets_received": self.total_received,
            "packets_sent": self.total_received + self.total_misses,
            "transmissions_per_node": transmissions_per_node.copy(),
            "packets_received_per_node": self.received_per_node.copy(),
            "packets_missed_per_node": self.misses_per_node.copy(),
            "packets_sent_per_node": [
                self.received_per_node[i] + self.misses_per_node[i] for i in range(len(self.nodes))
            ],
            "node_distances": node_distances  # Add node distances to the log entry
        }
        self.log_dynamic_data.append(log_entry)

    def render(self):
        """Render the environment with reception-based background and dynamic elements."""
        x = int(self.pos[0])
        y = int(self.pos[1])

        # Calculate padding for top, left, bottom, and right
        pad_top, pad_left = self.offset_y, self.offset_x
        pad_bottom, pad_right = pad_top, pad_left  # // 2
        # print(f"{pad_top, pad_left, pad_bottom, pad_right = }")

        # Add padding to the color frame using np.pad
        padded_color_frame = np.pad(
            self.background_frame,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant', constant_values=0
        )

        # Create a new frame (background) and place the padded color frame into it
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[
        :padded_color_frame.shape[0],
        :padded_color_frame.shape[1]
        ] = padded_color_frame[
            :self.height,
            :self.width
            ]

        # Draw the moving point
        cv2.rectangle(frame, pt1=(self.offset_x + x - 2, self.offset_y + y - 2),
                      pt2=(self.offset_x + x + 2, self.offset_y + y + 2),
                      color=self.point_color)

        # Draw nodes and their transmission circles
        for node in self.nodes:
            cv2.circle(frame, center=(self.offset_x + node.pos[0], self.offset_y + node.pos[1]),
                       radius=int(node.transmission_model.max_transmission_distance), color=(255, 0, 0), thickness=2)
        i = 0
        for node in self.nodes:
            cv2.putText(frame, str(i), (node.pos[0], self.offset_y + node.pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=1)
            i += 1

        # Resize frame for better visualization
        enlarged_image = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)

        # Render text data and stats
        canvas = self.render_text_data(enlarged_image)

        # Enable resizable window and update the content dynamically
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Make the window resizable
        cv2.imshow(self.window_name, canvas)

        # Wait for resizing to reflect (fullscreen updates dynamically)
        cv2.waitKey(2)

    def render_text_data(self, image):
        """Render text stats and dynamic node data on a canvas."""
        # Define stats list
        stats = [
            f"Total received: {self.total_received}",
            f"Total misses: {self.total_misses}",
            f"Total reward: {self.total_reward:.3f}",
            f"Time: {self.steps} | {self.max_steps}"
        ]

        # Calculate text-related dimensions
        line_height = 25  # Spacing between text lines
        num_lines = len(stats)
        text_height = num_lines * line_height + 20  # Padding for text

        # Create a canvas larger than the image to include space for stats
        canvas = np.zeros((image.shape[0] + text_height, image.shape[1], 3), dtype=np.uint8)

        # Place the enlarged image on the canvas
        canvas[:image.shape[0], :, :] = image

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 1
        text_color = (255, 255, 255)  # white

        # Add stats below the image
        text_offset_x = 10
        text_offset_y = image.shape[0] + 20
        for i, text in enumerate(stats):
            cv2.putText(canvas, text, (text_offset_x, text_offset_y + (i * line_height)),
                        font, fontScale=font_scale, color=text_color, thickness=font_thickness)

        # Initialize variables for dynamic text rendering
        line_height = 30  # Define consistent vertical spacing

        # Initialize variables for dynamic text rendering
        text_offset_y = image.shape[0]
        base_offset_x = 300  # Initial horizontal position for node data
        id_offset = base_offset_x
        remaining_time_offset = base_offset_x + 30  # Offset for remaining time
        send_interval_offset = remaining_time_offset + 80  # Offset for combined " | send_interval" text

        # Render node data
        for i, node in enumerate(self.nodes):
            # Calculate the y-coordinate for this row
            y_coord = text_offset_y + (i * line_height)

            # Text values
            node_id_text = f"{i}:"
            remaining_time_text = f"{round(node.time_of_next_packet - self.steps)}"
            combined_text = f"| {self.send_intervals[i]}"  # Combine separator and send interval

            # Draw the text parts with fixed offsets
            cv2.putText(canvas, node_id_text, (id_offset, y_coord), font, font_scale, text_color, font_thickness)
            cv2.putText(canvas, remaining_time_text, (remaining_time_offset, y_coord), font, font_scale, text_color,
                        font_thickness)
            cv2.putText(canvas, combined_text, (send_interval_offset, y_coord), font, font_scale, text_color,
                        font_thickness)

        return canvas

    def close(self):
        cv2.destroyAllWindows()


class TransmissionModel:
    def __init__(self, max_transmission_distance=50, ploss_scale=300, use_deterministic_transmissions=False,
                 rssi_ref=-30, path_loss_exponent=2.7,
                 noise_floor=-100,
                 rssi_min=-100, rssi_max=-30, snir_min=0, snir_max=30):
        self.max_transmission_distance = max_transmission_distance
        self.ploss_scale = ploss_scale
        self.use_deterministic_transmissions = use_deterministic_transmissions
        self.rssi_ref = rssi_ref
        self.path_loss_exponent = path_loss_exponent
        self.noise_floor = noise_floor
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max
        self.snir_min = snir_min
        self.snir_max = snir_max

    def get_reception_prob(self, distance):
        # P(Reception) = probability of receiving packet.
        # Probability of receiving packet decreases with distance
        if distance > self.max_transmission_distance:
            return 0.0
        if self.use_deterministic_transmissions:
            return 1.0
        return np.exp(- distance / self.ploss_scale)

    def is_transmission_success(self, distance):
        receive_choice = self.get_reception_prob(distance) > np.random.rand()
        return receive_choice

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

    def inverse_RSSI(self, rssi):
        return self.inverse_generate_rssi(rssi)


class PACKET_STATUS(Enum):
    RECEIVED = 1
    LOST = 2
    NOT_SENT = 3


class Node:
    def __init__(self, pos: tuple[int, int], transmission_model: TransmissionModel, time_to_first_packet: int,
                 send_interval: int, send_std=10, use_deterministic_transmissions=False):
        self.pos = pos
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.set_send_interval(send_interval)
        self.use_deterministic_transmissions = use_deterministic_transmissions
        self.send_std = send_std  # standard deviation

        self.transmission_model = transmission_model

    def set_send_interval(self, send_interval):
        self.send_interval = send_interval
        interval_bound_scale = 0.01
        self.lower_bound_send_time = send_interval - send_interval * interval_bound_scale
        self.upper_bound_send_time = send_interval + send_interval * interval_bound_scale

    def reset(self):
        self.time_of_next_packet = self.time_to_first_packet

    def generate_next_interval(self):
        if self.use_deterministic_transmissions:
            return self.send_interval
        # Generate a truncated normal value for the next time interval
        # a and b are calculated to truncate around the mean interval with some range
        a, b = (self.lower_bound_send_time - self.send_interval) / self.send_std, (
                self.upper_bound_send_time - self.send_interval) / self.send_std
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_std)
        # print(f"{self.send_interval=}, {interval=}")
        return interval

    def transmission(self, gpos):
        distance = math.dist(self.pos, gpos)
        if self.transmission_model.is_transmission_success(distance):
            rssi_scaled = self.transmission_model.generate_rssi(distance)
            snir_scaled = self.transmission_model.generate_snir(distance)
            return True, rssi_scaled, snir_scaled
        return False, 0, 0

    def send(self, time, gpos):
        # Decides whether a packet should be send and if it gets lost
        # Pobability of success is based of last time send and distance
        if time >= self.time_of_next_packet:
            # print(f"prior: {self.time_of_next_packet =}")
            self.time_of_next_packet = time + self.generate_next_interval()
            # print(f"after: {self.time_of_next_packet =}")
            # f"time of next packet: {self.time_of_next_packet}" )
            is_received, rssi, snir = self.transmission(gpos)
            if is_received:
                # print(f"packet is_received ")
                return PACKET_STATUS.RECEIVED, rssi, snir
            else:
                return PACKET_STATUS.LOST, 0, 0
        return PACKET_STATUS.NOT_SENT, 0, 0
