from stable_baselines3 import DQN, PPO
from simple_env import SimpleBaseEnv
from simple_env import RewardPlottingCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
import numpy as np


import multiprocessing
## tensorboard --logdir ./tensorboard/;./tensorboard/  ##
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
        total_received_values = [info.get('total_received', 0) for info in infos]
        total_misses_values = [info.get('total_misses', 0) for info in infos]
        # Log the average (or sum) across environments, if needed
        avg_total_received = sum(total_received_values) / len(total_received_values)
        avg_total_misses = sum(total_misses_values) / len(total_misses_values)
        self.logger.record("total_received", avg_total_received)
        self.logger.record("total_misses", avg_total_received)
        return True


def main():
    envs = 4
    env = make_vec_env(SimpleBaseEnv, n_envs=envs, vec_env_cls= SubprocVecEnv, env_kwargs={'render_mode': "none"})

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path="stable-model-best")
    print("Learning started")
    model = PPO("MlpPolicy", env, tensorboard_log="./tensorboard/").learn(2000000, callback=[eval_callback, TensorboardCallback()])

    model.save("stable-model")

if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()