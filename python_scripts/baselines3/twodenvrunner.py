import multiprocessing
import warnings

import tensorflow as tf
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor

from twod_env import TwoDEnv, FrameSkip

# Set TensorFlow logging to errors only (ignores warnings and info)
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_skipped_env():
    time_skip = 10
    # TODO: use_deterministic_transmissions=False
    env = TwoDEnv(render_mode="none", timeskip=time_skip, use_deterministic_transmissions=False)
    env = FrameSkip(env, skip=time_skip)  # Frame skip for action repeat
    return env


class CustomPolicyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, num_blocks=4):
        super(CustomPolicyNetwork, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        # Define the input layer
        self.input_layer = nn.Linear(input_dim, 64)

        # Create residual blocks dynamically
        self.residual_blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_blocks)])

        # Define activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Define the output layer
        self.output_layer = nn.Linear(64, features_dim)

    def forward(self, observations):
        # Initial input layer
        x = self.input_layer(observations)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply residual blocks
        # https://en.wikipedia.org/wiki/Residual_neural_network
        for layer in self.residual_blocks:
            residual = x  # Save input for skip connection
            x = layer(x)
            x = self.activation(x)
            x = x + residual  # Add residual (skip connection) #TODO: reactivate this?

        # Final output layer
        x = self.output_layer(x)
        return x

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
    envs = 16
    env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
    # TODO: DONT try VecNormalize with this VecMonitor inbetween. Remember to do the same in test2dmodel
    #  (https://www.reddit.com/r/reinforcementlearning/comments/1c9krih/dummyvecenv_vecnormalize_makes_the_reward_chart/)
    #
    #env = VecMonitor(env)
    gamma = 0.80
    env = VecNormalize(env, gamma=gamma, norm_reward=True)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=150, min_evals=100, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=4096, callback_after_eval=stop_train_callback,
                                 verbose=1, best_model_save_path="stable-model-2d-best")
    policy_kwargs = dict(
        features_extractor_class=CustomPolicyNetwork,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])
    )
    model = PPO("MlpPolicy", env, device="cpu", learning_rate=3e-4, gamma=gamma, ent_coef=0.01, batch_size=256,
                clip_range=0.15, n_steps=8192*2, n_epochs=20,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tensorboard/",
                )
    # TODO: remove ent_coef from above, and change model learn steps
    # TODO: learning_rate=1e-3, learning steps = 500000, ent_coef=0.0075, {"net_arch": [64, 64, 64]}, batch_size=8192,?
    print("Learning started")
    # default timesteps: 500000
    model = model.learn(8_000_000, callback=[eval_callback, TensorboardCallback()])
    print("Learning finished")
    model.save("stable-model")
    env.training = False
    env.save("model_normalization_stats")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
