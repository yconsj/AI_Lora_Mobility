import multiprocessing
import warnings

import tensorflow as tf
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from twod_env import TwoDEnv, FrameSkip

# Set TensorFlow logging to errors only (ignores warnings and info)
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_skipped_env():
    time_skip = 15
    env = TwoDEnv(render_mode="none", timeskip=time_skip )
    env = FrameSkip(env, skip=time_skip )  # Frame skip for action repeat
    return env


class CustomPolicyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomPolicyNetwork, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        # Define a simple network with dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout applied here
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations)

## tensorboard --logdir ./tensorboard/; ##
# http://localhost:6006/
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard, only during evaluation.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Ensure this callback is only triggered during evaluation

        infos = self.locals['infos']
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                fairness = infos[i].get('fairness', 0)
                total_received_values = infos[i].get("total_received", 0)
                total_misses_values = infos[i].get('total_misses', 0)
                packets_sent = (total_received_values + total_misses_values)
                if packets_sent == 0:
                    packet_delivery_rate = 0
                else:
                    packet_delivery_rate = total_received_values / packets_sent
                self.logger.record("custom_logs/fairness", fairness)
                self.logger.record("custom_logs/total_received", total_received_values)
                self.logger.record("custom_logs/total_misses", total_misses_values)
                self.logger.record("custom_logs/delivery_rate", packet_delivery_rate)
        return True


def main():
    envs = 4
    env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=200, min_evals=100, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback,
                                 verbose=1, best_model_save_path="stable-model-2d-best")
    policy_kwargs = dict(
        features_extractor_class=CustomPolicyNetwork,
        features_extractor_kwargs=dict(features_dim=64)
    )
    model = PPO("MlpPolicy", env, device="cpu", gamma=0.85, ent_coef=0.01, n_steps=4096,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tensorboard/",
                )
    # TODO: remove ent_coef from above, and change model learn steps
    # TODO: learning_rate=1e-3, learning steps = 500000, ent_coef=0.005, {"net_arch": [64, 64, 64]}?
    print("Learning started")
    # default timesteps: 500000
    model = model.learn(6_000_000, callback=[eval_callback, TensorboardCallback()])
    print("Learning finished")
    model.save("stable-model")
    env.training = False
    env.save("model_normalization_stats")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
