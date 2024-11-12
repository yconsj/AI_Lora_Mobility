from stable_baselines3 import DQN, PPO
from simple_env import SimpleBaseEnv
from simple_env import RewardPlottingCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

import multiprocessing

def main():
    envs = 4
    env = make_vec_env(SimpleBaseEnv, n_envs=envs, vec_env_cls= SubprocVecEnv)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path="stable-model-best")
    model = PPO("MlpPolicy", env, ent_coef=0.001).learn(200000, callback=[RewardPlottingCallback(), eval_callback])

    model.save("stable-model")

if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()