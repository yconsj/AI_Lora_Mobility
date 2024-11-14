import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import  set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import gymnasium as gym
from baselines3.simple_env import SimpleBaseEnv, RewardPlottingCallback

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


def make_env(env_id, rank, seed=0):
    def _init():
        env = MaxAndSkipEnv(env_id, 4) # gym.make(env_id)
        env.seed(seed+rank)
        return env
    #set_random_seed(seed)
    return _init

def empty_func():
    return

def main():
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    num_cpu = 2
    env_id = SimpleBaseEnv

    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    #env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method="spawn"), "tmp/monitor")

    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./board/", learning_rate=0.00003)

    eval_env = make_vec_env(env_id, n_envs=1, env_kwargs=dict())
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False)
    print("--- START LEARNING ---")

    model.learn(total_timesteps=500000, callback=[RewardPlottingCallback(), eval_callback])  #
    model.save("stable-model")
    print("--- DONE LEARNING ---")


if __name__ == '__main__':
    main()
