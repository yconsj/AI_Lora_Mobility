import multiprocessing

import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from twod_env import TwoDEnv, FrameSkip, FrameStack




def make_skipped_env():
    env = TwoDEnv(render_mode="none")
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env


def make_framestacked_env():
    env = TwoDEnv(render_mode="none")
    env = FrameSkip(env, skip=10)
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
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access total_recieved from the 'infos' list
        infos = self.locals['infos']
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                total_received_values = infos[i].get("total_received", 0)
                total_misses_values = infos[i].get('total_misses', 0)
                self.logger.record("total_received", total_received_values)
                self.logger.record("total_misses", total_misses_values)
        return True


def main():
    envs = 6
    env = make_vec_env(make_framestacked_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=20, verbose=1)
    ## tensorboard --logdir ./tensorboard/;./tensorboard/  ##
    # http://localhost:6006/
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1,
                                 best_model_save_path="stable-model-2d-best")
    print("Learning started")
    if th.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Define policy_kwargs with the custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),  # Output feature dimension
    )

    # Create the PPO model using CnnPolicy
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, gamma=0.999, tensorboard_log="./tensorboard/",
                device=device, batch_size=64,
                n_steps=2048)
    # Move to device if necessary
    model.policy.to(device)

    model.learn(500000, callback=[eval_callback, TensorboardCallback()])
    model.save("stable-model")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
