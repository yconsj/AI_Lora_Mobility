from stable_baselines3 import PPO

from baselines3.basecase.basecase_plot_episode_log import plot_mobile_gateway
from simple_env import SimpleBaseEnv, FrameSkip
from stable_baselines3.common.env_util import make_vec_env
import json
import matplotlib.pyplot as plt

do_logging = True
if do_logging:
    logfile = "env_log.json"
    render_mode = None
else:
    logfile = ""
    render_mode = "cv2"

def make_skipped_env():
    env = SimpleBaseEnv(render_mode=render_mode, do_logging=do_logging, log_file=logfile)
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env


def get_action_probs(input_state, input_model):
    input_model_obs = input_model.policy.obs_to_tensor(input_state)[0]
    dis = input_model.policy.get_distribution(input_model_obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np




vec_env = make_vec_env(make_skipped_env, n_envs=1, env_kwargs=dict())
model = PPO.load("stable-model-best/best_model", print_system_info=True)
print(model.policy)
obs = vec_env.reset()

# test trained model
done = False
counter = 0
while not done:
    # Get action and probabilities
    action, _ = model.predict(obs, deterministic=True)
    # Print state, action, and probabilities
    if counter % 100 == 0:
        action_probabilities = get_action_probs(obs, model)
        print(f"State: {obs}")
        print(f"Action: {action}")
        print(f"Action Probabilities: {action_probabilities}")
    counter += 1

    obs, reward, done, info = vec_env.step(action)

    if done:
        # VecEnv resets automatically when done
        break
if do_logging:
    plot_mobile_gateway(logfile)
