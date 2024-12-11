from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO
from twod_env import TwoDEnv, FrameSkip
from twodenvrunner import NETWORK_TYPE
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


def make_skipped_env():
    env = TwoDEnv(render_mode="cv2")
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env


vec_env = None
network_type = NETWORK_TYPE.MLP
if network_type == NETWORK_TYPE.MLP_LSTM:
    vec_env = make_vec_env(make_skipped_env, n_envs=1, env_kwargs=dict())
    model = RecurrentPPO("MlpLstmPolicy", vec_env)
    model.set_parameters("stable-model")
    lstm_states = None
elif network_type == NETWORK_TYPE.MLP:
    vec_env = make_vec_env(make_skipped_env, n_envs=1, env_kwargs=dict())

test_best = False

if test_best:
    model = PPO.load("stable-model-2d-best/best_model", print_system_info=True)
else:
    model = PPO.load("stable-model", print_system_info=True)

print(model.policy)
obs = vec_env.reset()
print(obs)


# test trained model
def get_action_probs(input_state, model):
    obs = model.policy.obs_to_tensor(input_state)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np


done = False
counter = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    if counter % 100 == 0:
        # action_probabilities = get_action_probs(obs, model)
        # print(f"State: {obs}")
        print(f"Action: {action}")
        # print(f"Action Probabilities: {action_probabilities}")
    counter += 1

    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        break
