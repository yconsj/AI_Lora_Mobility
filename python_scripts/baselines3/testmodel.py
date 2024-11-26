from stable_baselines3 import DQN, PPO
from simple_env import SimpleBaseEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import obs_as_tensor



def make_skipped_env():
    env = SimpleBaseEnv(render_mode="cv2")
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env


vec_env = make_vec_env(make_skipped_env, n_envs=1, env_kwargs=dict())
model = PPO.load("stable-model-best/best_model", print_system_info=True)


def get_action_probs(input_state, model):
    obs = model.policy.obs_to_tensor(input_state)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np


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

    obs, reward, done, info = vec_env.step(action)

    counter += 1
    if done:
        # VecEnv resets automatically when done
        break
