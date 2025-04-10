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

# Suppress TensorFlow and deprecation warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_skipped_env():
    return FrameSkip(
        TwoDEnv(
            render_mode="none",
            max_steps=86400,
            number_of_model_nodes=4,
            number_of_sim_nodes=8,
            use_node_index_sorting=True,
            model_use_node_priority=False
        ),
        skip=10
    )


class CustomPolicyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, num_blocks=3):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.input_layer = nn.Linear(input_dim, 64)
        self.residual_blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_blocks)])
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(64, features_dim)

    def forward(self, observations):
        x = self.activation(self.input_layer(observations))
        for layer in self.residual_blocks:
            residual = x
            x = self.activation(layer(x)) + residual
        return self.output_layer(x)


class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                received = info.get("total_received", 0)
                missed = info.get("total_misses", 0)
                sent = received + missed
                delivery_rate = received / sent if sent else 0
                self.logger.record("custom_logs/fairness", info.get("fairness", 0))
                self.logger.record("custom_logs/total_received", received)
                self.logger.record("custom_logs/total_misses", missed)
                self.logger.record("custom_logs/delivery_rate", delivery_rate)
        self.logger.dump(self.num_timesteps)
        return True


def main():
    multiprocessing.set_start_method('spawn', force=True)

    envs = 16
    gamma, ent_coef, learning_rate = 0.9, 0.005, 1e-4
    use_relu, use_resnet, n_blocks = False, False, 0
    model_class = PPO

    vec_env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls=SubprocVecEnv)
    max_steps = vec_env.get_attr("max_steps")[0]
    time_skip = vec_env.get_attr("_skip")[0]
    steps_per_episode = math.ceil(max_steps / time_skip)

    print(f"{max_steps = }, {time_skip = }")

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50, min_evals=100, verbose=1
    )

    eval_callback = EvalCallback(
        vec_env,
        eval_freq=steps_per_episode * 2,
        best_model_save_path="stable-model-2d-best",
        verbose=1
    )

    policy_kwargs = {"share_features_extractor": True}
    if use_relu:
        policy_kwargs["activation_fn"] = nn.ReLU
    if use_resnet:
        policy_kwargs.update({
            "features_extractor_class": CustomPolicyNetwork,
            "features_extractor_kwargs": dict(features_dim=64, num_blocks=n_blocks)
        })

    if model_class == PPO:
        model = PPO(
            "MlpPolicy", vec_env, device="cpu",
            learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef,
            batch_size=64, clip_range=0.15, n_steps=steps_per_episode * 2, n_epochs=10,
            policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard/"
        )
    else:
        model = DQN(
            "MlpPolicy", vec_env, device="cpu",
            learning_rate=learning_rate, gamma=gamma, batch_size=64,
            policy_kwargs={"net_arch": [64, 64, 64]}, tensorboard_log="./tensorboard/"
        )

    tb_log_name = (
        f"{model_class.__name__};{'relu' if use_relu else 'tanh'};"
        f"{'resnet' if use_resnet else 'mlp'};g_{gamma};e_{ent_coef};lr_{learning_rate}"
    )
    print(f"Learning started, tb_log: {tb_log_name}")

    vec_env.reset()
    model.learn(
        total_timesteps=2_000_000,
        callback=[eval_callback, TensorboardCallback()],
        tb_log_name=tb_log_name
    )

    print("Learning finished")
    model.save("stable-model")


if __name__ == '__main__':
    main()
