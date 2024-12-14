from stable_baselines3 import PPO
from baselines3.advanced_plot_episode_log import plot_mobile_gateway_with_nodes_advanced
from twod_env import TwoDEnv, FrameSkip
from stable_baselines3.common.env_util import make_vec_env

do_logging = True
if do_logging:
    logfile = "env_log.json"
    render_mode = None
else:
    logfile = ""
    render_mode = "cv2"


def make_skipped_env():
    env = TwoDEnv(render_mode=render_mode, timeskip=10, do_logging=do_logging, log_file=logfile)
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env


vec_env = make_vec_env(make_skipped_env, n_envs=1, env_kwargs=dict())
test_best = True

if test_best:
    model = PPO.load("stable-model-2d-best/best_model", device="cpu", print_system_info=True)
else:
    model = PPO.load("stable-model", device="cpu", print_system_info=True)

print(model.policy)
obs = vec_env.reset()
print(obs)


# test trained model
def get_action_probs(input_state, input_model):
    obs = input_model.policy.obs_to_tensor(input_state)[0]
    dis = input_model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np


done = False
counter = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(reward)
    if counter % 100 == 0:
        action_probabilities = get_action_probs(obs, model)
        print(f"State: {obs}")
        print(f"Action: {action}")
        print(f"Action Probabilities: {action_probabilities}")
    counter += 1
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        break
if do_logging:
    plot_mobile_gateway_with_nodes_advanced(logfile)


