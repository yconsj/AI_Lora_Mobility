from stable_baselines3 import DQN, PPO
from simple_env import SimpleBaseEnv
from simple_env import RewardPlottingCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

env = SimpleBaseEnv()
# Define and Train the agent
check_env(env, warn=True)


vec_env = make_vec_env(SimpleBaseEnv, n_envs=1, env_kwargs=dict())

model = PPO("MlpPolicy", vec_env, ent_coef=0.05).learn(200000, callback=RewardPlottingCallback())

model.save("stable-model")

obs = vec_env.reset()
