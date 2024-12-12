import multiprocessing
from enum import Enum

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn
from twod_env import TwoDEnv, FrameSkip
import warnings
import tensorflow as tf

# Set TensorFlow logging to errors only (ignores warnings and info)
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_skipped_env():
    skips = 15
    env = TwoDEnv(render_mode="none", history_length=3, n_skips=skips)
    env = FrameSkip(env, skip=skips)  # Frame skip for action repeat
    return env


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard, only during evaluation.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Ensure this callback is only triggered during evaluation

        # If evaluation is happening, log to TensorBoard
        # if self.eval_callback.n_calls % self.eval_callback.eval_freq == 0:
        infos = self.locals['infos']
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                # print(f"INFO: {infos[i]}")  # Check what is inside the info
                total_received_values = infos[i].get("total_received", 0)
                total_misses_values = infos[i].get('total_misses', 0)
                self.logger.record("custom_logs/total_received", total_received_values)
                self.logger.record("custom_logs/total_misses", total_misses_values)
        return True


class NETWORK_TYPE(Enum):
    MLP = 1
    MLP_LSTM = 2


def main():
    if th.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    network_type = NETWORK_TYPE.MLP
    envs = 4
    n_steps = 2048 * 2
    total_steps = n_steps * envs  # 16,384
    # Batch size = total_steps / 8 (as a fraction)
    batch_size = total_steps // 8  # 2048
    model = None
    env = None
    if network_type == NETWORK_TYPE.MLP:
        env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy", env,
                    gamma=0.995,
                    ent_coef=0.02,
                    clip_range=0.2,
                    # Larger clip range to promote exploration during updates# Increase entropy coefficient to encourage exploration
                    tensorboard_log="./tensorboard/", device="cpu")

    elif network_type == NETWORK_TYPE.MLP_LSTM:
        env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
        model = RecurrentPPO("MlpLstmPolicy", env,
                             gamma=0.995, learning_rate=(10.0 ** -5), n_steps=n_steps, batch_size=batch_size,
                             tensorboard_log="./tensorboard/")

    ## tensorboard --logdir ./tensorboard/;./tensorboard/  ##
    # http://localhost:6006/
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, n_eval_episodes=10, eval_freq=10000, callback_after_eval=stop_train_callback,
                                 verbose=1,
                                 best_model_save_path="stable-model-2d-best")
    # Move to device if necessary
    # model.policy = model.policy.to(device=th.device(device), dtype=th.float32, non_blocking=True)
    print("Learning started")
    # default timesteps: 500000
    model = model.learn(800000, callback=[eval_callback, TensorboardCallback()])
    print("Learning finished")
    model.save("stable-model")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
