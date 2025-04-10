import json
import random
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from advanced_plot_episode_log import (
    plot_mobile_gateway_with_nodes_advanced,
    plot_heatmap,
    plot_batch_episode_performance,
    plot_relative_positions,
)
from twod_env import TwoDEnv, FrameSkip
from eval_twod_env import eval_twod_env

REWARD_KEYS = [
    'reception_reward_sum',
    'miss_reward_sum',
    'position_reward_sum',
    'action_reward_sum',
]


def sb3_get_action_probabilities(obs, model):
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    dist = model.policy.get_distribution(obs_tensor)
    return dist.distribution.probs.detach().cpu().numpy()


def make_skipped_env(do_logging, log_file, input_render_mode, do_eval_env=True, **kwargs):
    """Creates a TwoD environment with optional evaluation mode and frame skipping."""
    time_skip = 10
    node_positions = [(50, 50), (250, 250), (50, 250), (250, 50)]
    send_intervals = [1600] * len(node_positions)

    env_kwargs = dict(
        render_mode=input_render_mode,
        do_logging=do_logging,
        log_file=log_file,
        max_steps=86400,
        number_of_sim_nodes=len(node_positions),
        model_use_node_priority=False,
        use_node_index_sorting=True,
        **kwargs,
    )

    env = eval_twod_env(node_positions=node_positions, send_intervals=send_intervals, **env_kwargs) if do_eval_env \
        else TwoDEnv(**env_kwargs)

    return FrameSkip(env, skip=time_skip)


def log_reward_breakdown(step_idx, reward, info, obs, model):
    print(f"\nStep {step_idx}, reward = {reward[0]:.3f}")
    total = sum(info.get(k, 0.0) for k in REWARD_KEYS) or 1.0
    for key in REWARD_KEYS:
        val = info.get(key, 0.0)
        print(f"{key}: {val:.3f}, fraction: {val / total:.2%}")
    print(f"Observation: {obs[0]}")
    print(f"Action probabilities: {sb3_get_action_probabilities(obs, model)}")


def extract_final_node_stats(log_file):
    with open(log_file, 'r') as file:
        data = json.load(file)

    dynamic = data["dynamic"]
    node_count = len(dynamic[0]['packets_received_per_node'])

    received = [[] for _ in range(node_count)]
    sent = [[] for _ in range(node_count)]

    for entry in dynamic:
        for i in range(node_count):
            received[i].append(entry['packets_received_per_node'][i])
            sent[i].append(entry['packets_sent_per_node'][i])

    final_received = [r[-1] for r in received]
    final_sent = [s[-1] for s in sent]
    return final_received, final_sent, data["static"]["number_of_nodes"]


def evaluate_episodes(do_logging, log_file, n_episodes, mv_rendering_mode=None, do_eval_env=True):
    all_final_receives, all_final_sents = [], []

    model = PPO.load("stable-model-2d-best/best_model", device="cpu", print_system_info=True)
    print(f"{model.policy = }")
    model.set_random_seed(0)

    # Create non-rendering env once for all but the last episode
    vec_env = make_vec_env(
        make_skipped_env,
        n_envs=1,
        env_kwargs=dict(
            do_logging=do_logging,
            log_file=log_file,
            input_render_mode=None,
            do_eval_env=do_eval_env,
        )
    )

    for ep_idx in range(n_episodes):
        is_last = (ep_idx + 1 == n_episodes)

        # Replace env with rendering-enabled one only for the final episode
        if is_last and mv_rendering_mode:
            vec_env.close()  # Important to close the existing one before replacing
            vec_env = make_vec_env(make_skipped_env, n_envs=1,
                                   env_kwargs=dict(
                                       do_logging=do_logging,
                                       log_file=log_file,
                                       input_render_mode=mv_rendering_mode,
                                       do_eval_env=do_eval_env,
                                   )
                                   )

        print(f"Starting episode {ep_idx + 1}/{n_episodes}")
        obs = vec_env.reset()
        done, step_counter = False, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)
            info = infos[0]

            if step_counter % 100 == 0:
                log_reward_breakdown(step_counter, reward, info, obs, model)
            step_counter += 1

        if do_logging:
            final_rx, final_tx, num_nodes = extract_final_node_stats(log_file)
            all_final_receives.append(final_rx)
            all_final_sents.append(final_tx)

            if is_last:
                plot_relative_positions(log_file, number_of_nodes=num_nodes)
                plot_mobile_gateway_with_nodes_advanced(log_file)
                plot_heatmap(log_file=log_file)

    if do_logging:
        plot_batch_episode_performance(all_final_receives, all_final_sents)


if __name__ == '__main__':
    random.seed(0)
    multiprocessing.set_start_method('spawn')
    evaluate_episodes(
        do_logging=True,
        log_file="env_log.json",
        n_episodes=1,
        mv_rendering_mode="cv2",
        do_eval_env=True
    )
