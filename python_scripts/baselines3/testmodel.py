from stable_baselines3 import DQN, PPO
from simple_env import SimpleBaseEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
def make_skipped_env():
    env = SimpleBaseEnv(render_mode="cv2")
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env

vec_env = make_vec_env(make_skipped_env, n_envs=1, env_kwargs=dict())
model = PPO("MlpPolicy", vec_env)

model.load("stable-model-best/best_model", print_system_info=True)
print(model.policy)
obs = vec_env.reset()

# test trained model

done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        break
