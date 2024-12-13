from twod_env import TwoDEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env

env = TwoDEnv(render_mode="cv2", timeskip=10)
env = FrameSkip(env, skip=10)  # Frame skip for action repeat

episodes = 1
check_env(env, warn=True)
use_random_action = True
counter = 0
for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        if use_random_action:
            action = env.action_space.sample()
        else:
            action = int(input("number from 0 to 4"))
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        if counter % 100 == 0:
            print(obs)
        counter += 1

