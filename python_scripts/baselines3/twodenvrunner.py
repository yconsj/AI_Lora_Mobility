import multiprocessing
import warnings

import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from twod_env import TwoDEnv, FrameSkip

# Set TensorFlow logging to errors only (ignores warnings and info)
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_skipped_env():
    env = TwoDEnv(render_mode="none", timeskip=10)
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
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
                packet_delivery_rate = total_received_values + (total_received_values + total_misses_values)
                # TODO: add packet delivery rate
                self.logger.record("custom_logs/total_received", total_received_values)
                self.logger.record("custom_logs/total_misses", total_misses_values)
                self.logger.record("custom_logs/pdr", packet_delivery_rate)
        return True


def main():
    envs = 4
    env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback,
                                 verbose=1, best_model_save_path="stable-model-2d-best")
    model = PPO("MlpPolicy", env, gamma=0.99, ent_coef=0.01, tensorboard_log="./tensorboard/")
    print("Learning started")
    # default timesteps: 500000
    model = model.learn(1000000, callback=[eval_callback, TensorboardCallback()])
    print("Learning finished")
    model.save("stable-model")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
