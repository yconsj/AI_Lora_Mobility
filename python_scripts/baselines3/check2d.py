from twod_env import TwoDEnv, FrameSkip, FrameStack
from stable_baselines3.common.env_checker import check_env

env = TwoDEnv(render_mode="cv2")
env = FrameSkip(env, skip=10)  # Frame skip for action repeat
env = FrameStack(env, stack_size=5)

episodes = 1
check_env(env, warn=True)

counter = 0
for episode in range(episodes):
    done = False 
    obs = env.reset()
    while not done:
        
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)
        print(reward)
        if counter % 10 == 0:
            print(obs)
        counter += 1
