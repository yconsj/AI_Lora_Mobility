from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from simple_env import SimpleBaseEnv
from simple_env import RewardPlottingCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

env = SimpleBaseEnv()
# Define and Train the agent
check_env(env, warn=True)


vec_env = make_vec_env(SimpleBaseEnv, n_envs=1, env_kwargs=dict())

eval_env = make_vec_env(SimpleBaseEnv, n_envs=1, env_kwargs=dict())

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

model = PPO("MlpPolicy", vec_env, ent_coef=0.05)
model.learn(200000, callback=[RewardPlottingCallback(), eval_callback])
model.save("stable-model")

obs = vec_env.reset()
