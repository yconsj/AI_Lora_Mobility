from stable_baselines3.common.env_checker import check_env
from baselines3.twod_env import TwoDEnv, FrameSkip


def main(skip=10, episodes=1, use_random_action=True):
    env = FrameSkip(TwoDEnv(render_mode="cv2", number_of_sim_nodes=4), skip=skip)
    check_env(env, warn=True)

    reward_keys = [
        'reception_reward_sum',
        'miss_reward_sum',
        'position_reward_sum',
        'action_reward_sum'
    ]

    step_counter = 0
    for _ in range(episodes):
        obs, done = env.reset(), False

        while not done:
            action = env.action_space.sample() if use_random_action else int(input("Enter action [0–4]: "))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print(f"Step {step_counter}, reward: {reward:.3f}")

            if step_counter % 100 == 0:
                total = sum(info.get(k, 0.0) for k in reward_keys) or 1.0  # avoid division by zero
                for key in reward_keys:
                    value = info.get(key, 0.0)
                    print(f"{key}: {value:.3f}, fraction: {value / total:.2%}")
                print(f"Observation: {obs}")

            step_counter += 1


if __name__ == "__main__":
    main()
