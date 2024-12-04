from twod_env import TwoDEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env

env = TwoDEnv(render_mode="cv2", timeskip=10)
env = FrameSkip(env, skip=10)  # Frame skip for action repeat

episodes = 1
check_env(env, warn=True)

for episode in range(episodes):
    done = False 
    obs = env.reset()
    while not done:
        
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info= env.step(random_action)
        print(obs)
        print(reward)
