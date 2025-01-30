# twod_env wrapper, where positions and send intevals can be set

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
from baselines3.twod_env import TwoDEnv, schedule_first_packets, _generate_color_frame


class eval_twod_env(TwoDEnv):
    def __init__(self, render_mode="none", do_logging=False, log_file=None, node_positions= [(50,50),(250,250),(50,250),(250,50)], gateway_positon = (150,10), send_intervals = [1600,1600,1600,1600]):
        super().__init__(render_mode, do_logging, log_file)
        self.pos = gateway_positon
        self.send_intervals = send_intervals
        self.positions = node_positions
    def reset(self, seed=None, options=None):
        self.recent_packets = deque([-1] * self.recent_packets_length, maxlen=self.recent_packets_length)
        self.prev_pos = self.pos
        self.total_misses = 0
        self.first_packets = schedule_first_packets(self.send_intervals, initial_delay=600)
        self.nodes[2].transmission_model.probability_modifier = 0 # set node 3 to have 0 prob of success transmit
        for i in range(len(self.nodes)):
            self.nodes[i].pos = self.positions[i]
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


