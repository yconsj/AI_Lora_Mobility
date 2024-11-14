from simple_env import SimpleBaseEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env

env = SimpleBaseEnv(render_mode="cv2")
env = FrameSkip(env, skip=10)  # Frame skip for action repeat

episodes = 1
check_env(env, warn=True)

for episode in range(episodes):
    done = False 
    obs = env.reset()
    while not done:
        
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info= env.step(random_action)
        print(reward)
