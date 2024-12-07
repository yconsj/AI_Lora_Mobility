from twod_env import TwoDEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env

env = TwoDEnv(render_mode="cv2")
env = FrameSkip(env, skip=30)  # Frame skip for action repeat

episodes = 1
check_env(env, warn=True)

counter = 0
for episode in range(episodes):
    done = False 
    obs = env.reset()
    while not done:
        
        random_action = env.action_space.sample()
        user_action = int(input("number from 0 to 4"))
        obs, reward, done, truncated, info = env.step(user_action)
        print(reward)
        if counter % 10 == 0:
            print(obs)
        counter += 1
