from simple_env import SimpleBaseEnv
from stable_baselines3.common.env_checker import check_env

env = SimpleBaseEnv()
episodes = 1
check_env(env, warn=True)

for episode in range(episodes):
    done = False 
    obs = env.reset()
    while not done:
        env.render()
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info= env.step(random_action)
        print(reward)
