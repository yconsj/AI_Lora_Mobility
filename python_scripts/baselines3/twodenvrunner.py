import multiprocessing
from enum import Enum

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from twod_env import TwoDEnv, FrameSkip, FrameStack


def make_skipped_env():
    env = TwoDEnv(render_mode="none")
    env = FrameSkip(env, skip=15)  # Frame skip for action repeat
    return env


def make_framestacked_env():
    env = TwoDEnv(render_mode="none")
    env = FrameSkip(env, skip=15)
    env = FrameStack(env, stack_size=5)
    return env


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom feature extractor using Conv1D.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Assuming input shape is (num_entities, num_features)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output size after the CNN
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layer to map CNN output to feature size
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard, only during evaluation.
    """

    def __init__(self, eval_callback: EvalCallback, verbose=0):
        super().__init__(verbose)
        self.eval_callback = eval_callback

    def _on_step(self) -> bool:
        # Ensure this callback is only triggered during evaluation

        # If evaluation is happening, log to TensorBoard
        # if self.eval_callback.n_calls % self.eval_callback.eval_freq == 0:
        infos = self.locals['infos']
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                total_received_values = infos[i].get("total_received", 0)
                total_misses_values = infos[i].get('total_misses', 0)
                self.logger.record("custom_logs/total_received", total_received_values)
                self.logger.record("custom_logs/total_misses", total_misses_values)
        return True


class NETWORK_TYPE(Enum):
    MLP = 1
    MLP_LSTM = 2
    CNN = 3
    CNN_DQN = 4


def main():
    if th.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    network_type = NETWORK_TYPE.CNN
    envs = 4
    n_steps = 2048 * 2
    total_steps = n_steps * envs  # 16,384
    # Batch size = total_steps / 8 (as a fraction)
    batch_size = total_steps // 8  # 2048

    if network_type == NETWORK_TYPE.CNN:
        env = make_vec_env(make_framestacked_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
        # Define policy_kwargs with the custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),  # Output feature dimension
        )
        # Create the PPO model using CnnPolicy
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs,
                    gamma=0.999, learning_rate=(10.0 ** -4), n_steps=n_steps, batch_size=batch_size,
                    tensorboard_log="./tensorboard/",
                    device=device)  # batch_size=64, n_steps=2048
    elif network_type == NETWORK_TYPE.CNN_DQN:
        env = make_vec_env(make_framestacked_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
        # Define policy_kwargs with the custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),  # Output feature dimension
        )
        model = DQN(
            "CnnPolicy",  # Use CNN policy since you're using frame stacking
            env,  # Environment
            policy_kwargs=policy_kwargs,  # Your custom feature extractor (CNN)
            gamma=0.95,  # Discount factor (same as PPO, but might need adjustment)
            learning_rate=1e-4,  # You might need to adjust the learning rate
            buffer_size=100000,  # Size of the replay buffer
            exploration_fraction=0.15,  # Fraction of the total training steps used for exploration
            exploration_final_eps=0.01,  # Final epsilon value for exploration
            train_freq=4,  # Frequency of training
            batch_size=32,  # The batch size for experience replay
            tensorboard_log="./tensorboard/",  # Tensorboard logging
            device=device,  # The device to run the model on
            verbose=1  # Adjust verbosity
        )
    elif network_type == NETWORK_TYPE.MLP:
        env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy", env,
                    gamma=0.999, learning_rate=(10.0 ** -4), n_steps=n_steps, batch_size=batch_size,
                    tensorboard_log="./tensorboard/")

    else:  # network_type == NETWORK_TYPE.MLP_LSTM:
        env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
        model = RecurrentPPO("MlpLstmPolicy", env,
                             gamma=0.995, learning_rate=(10.0 ** -5), n_steps=n_steps, batch_size=batch_size,
                             tensorboard_log="./tensorboard/")

    ## tensorboard --logdir ./tensorboard/;./tensorboard/  ##
    # http://localhost:6006/
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, n_eval_episodes=10, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1,
                                 best_model_save_path="stable-model-2d-best")
    # Move to device if necessary
    # model.policy = model.policy.to(device=th.device(device), dtype=th.float32, non_blocking=True)
    print("Learning started")
    try:
        # default timesteps: 500000
        model = model.learn(1000000, callback=[eval_callback, TensorboardCallback(eval_callback=eval_callback)])
        model.save("stable-model")
    except KeyboardInterrupt:
        model.save("stable-model")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
