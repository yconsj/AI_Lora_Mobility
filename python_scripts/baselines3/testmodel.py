from stable_baselines3 import DQN, PPO
from simple_env import SimpleBaseEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
env = SimpleBaseEnv()
# Define and Train the agent
check_env(env, warn=True)
vec_env = make_vec_env(SimpleBaseEnv, n_envs=1, env_kwargs=dict())
model = PPO("MlpPolicy", vec_env)

model.load("stable-model", print_system_info=True)
print(model.policy)
obs = vec_env.reset()

# test trained model

done = False
vec_env.render()

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        break
