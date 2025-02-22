import math
import multiprocessing
import warnings

import tensorflow as tf
import torch.nn as nn
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from twod_env import TwoDEnv, FrameSkip


# Set TensorFlow logging to errors only (ignores warnings and info)
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_skipped_env():
    time_skip = 10
    # TODO: use_deterministic_transmissions=False
    env = TwoDEnv(render_mode="none", use_deterministic_transmissions=False)
    env = FrameSkip(env, skip=time_skip)  # Frame skip for action repeat
    return env


class CustomPolicyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, num_blocks=3):
        super(CustomPolicyNetwork, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        # Define the input layer
        self.input_layer = nn.Linear(input_dim, 64)

        # Create residual blocks dynamically
        self.residual_blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_blocks)])

        # Define activation and dropout
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)

        # Define the output layer
        self.output_layer = nn.Linear(64, features_dim)

    def forward(self, observations):
        # Initial input layer
        x = self.input_layer(observations)
        x = self.activation(x)

        # Apply residual blocks
        # https://en.wikipedia.org/wiki/Residual_neural_network
        for layer in self.residual_blocks:
            residual = x  # Save input for skip connection
            x = layer(x)
            x = self.activation(x)
            x = x + residual  # Add residual (skip connection)

        # Final output layer
        x = self.output_layer(x)
        return x

# tensorboard info
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
        self.logger.dump(self.num_timesteps)
        return True


def main():

    envs = 16
    env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
    gamma = 0.85  # base: 0.85
    ent_coef = 0.005  # base: 0.005
    learning_rate = 1e-4  # base: 6e-5
    n_blocks = 0
    max_steps = env.get_attr("max_steps")[0]  # env_id=0 in VecEnv
    time_skip = env.get_attr("_skip")[0]
    steps_per_episode = math.ceil(max_steps / time_skip)
    print(f"{max_steps = }")
    print(f"{time_skip = }")

    # for scenarios:
    model_class = PPO  # (can be replaced with DQN)
    use_ReLu = False  # default activation function is tanh
    use_ResNet = False

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=100, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=steps_per_episode * 2, #  callback_after_eval=stop_train_callback,
                                 verbose=1, best_model_save_path="stable-model-2d-best")
    if model_class == PPO:
        policy_kwargs = dict(
            share_features_extractor=True,
            #net_arch=[64, 64, 64]
        )
        if use_ReLu:
            policy_kwargs["activation_fn"] = nn.ReLU
        if use_ResNet:
            policy_kwargs["features_extractor_class"] = CustomPolicyNetwork
            policy_kwargs["features_extractor_kwargs"] = dict(features_dim=64, num_blocks=n_blocks)
        model = PPO("MlpPolicy", env, device="cpu", learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef,
                    batch_size=64,
                    clip_range=0.15,
                    n_steps=steps_per_episode * 2,
                    n_epochs=10,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./tensorboard/",
                    )
    else:  # assuming DQN
        policy_kwargs = dict(
            net_arch=[64, 64, 64]
        )
        model = DQN("MlpPolicy", env, device="cpu", learning_rate=learning_rate, gamma=gamma,
                    batch_size=64,  # base: 64
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./tensorboard/",
                    )
    model_type = type(model).__name__

    # "si": send interval input state, not using the time_of_next_packet of each node.
    # "ept": 'expected packet time' in input state. uses send interval to deduce when the packets should be approximately sent.
    # "tpt": 'true packet time'
    # "fe": full episode, i.e. not early termination
    # "sm": small model (fewer inputs in observation)
    tb_log_name = f"{model_type};{'relu' if use_ReLu else 'tanh'};{'resnet' if use_ResNet else 'mlp'};" \
                  f"g_{gamma};e_{ent_coef};lr_{learning_rate}"
    print(f"Learning started, tb_log: {tb_log_name}")
    env.reset()
    model = model.learn(1_000, callback=[eval_callback, TensorboardCallback()], # 10_00
                        tb_log_name=tb_log_name)

    print("Learning finished")
    model.save("stable-model")


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()
