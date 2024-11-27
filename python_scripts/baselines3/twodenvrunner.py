from stable_baselines3 import DQN, PPO
from twod_env import TwoDEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
import numpy as np


import multiprocessing
## tensorboard --logdir ./tensorboard/;./tensorboard/  ##

def make_skipped_env():
    env = TwoDEnv(render_mode="none")
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env
# http://localhost:6006/

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
    envs = 4
    env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls= SubprocVecEnv)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path="stable-model-2d-best")
    print("Learning started")
    model = PPO("MlpPolicy", env, tensorboard_log="./tensorboard/").learn(500000,callback=[eval_callback, TensorboardCallback()])

    model.save("stable-model")

if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()