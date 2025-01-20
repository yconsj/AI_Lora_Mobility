import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from advanced_plot_episode_log import load_json_log, plot_mobile_gateway_with_nodes_advanced, plot_heatmap, \
    plot_batch_episode_performance
from twod_env import TwoDEnv, FrameSkip, jains_fairness_index
from stable_baselines3.common.env_util import make_vec_env


def get_action_probabilities(input_state, input_model):
    obs = input_model.policy.obs_to_tensor(input_state)[0]
    dis = input_model.policy.get_distribution(obs)
    probabilities = dis.distribution.probs
    probabilities_np = probabilities.detach().cpu().numpy()
    return probabilities_np


grid_size_x, grid_size_y = 0, 0


def make_skipped_env(do_logging, log_file, input_render_mode):
    time_skip = 10
    global grid_size_x, grid_size_y
    env = TwoDEnv(render_mode=input_render_mode, timeskip=time_skip, do_logging=do_logging, log_file=log_file)
    grid_size_x = env.max_distance_x
    grid_size_y = env.max_distance_y
    env = FrameSkip(env, skip=time_skip)  # Frame skip for action repeat
    return env


def evaluate_episodes(do_logging, log_file, n_episodes, rendering_mode=None):
    all_pdr = []  # To store PDR values for each episode
    all_fairness = []  # To store fairness values for each episode

    test_best = True
    if test_best:
        model = PPO.load("stable-model-2d-best/best_model", device="cpu", print_system_info=True)
    else:
        model = PPO.load("stable-model", device="cpu", print_system_info=True)

    vec_env = make_vec_env(make_skipped_env, n_envs=1,
                           env_kwargs=dict(do_logging=do_logging, log_file=log_file, input_render_mode=None))
    # Load the saved statistics, but do not update them at test time and disable reward normalization.
    vec_env = VecNormalize.load("model_normalization_stats", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    for i in range(n_episodes):
        print(f"Starting episode {i}")
        if i == n_episodes - 1:  # only (potentially) render the last episode in the batch.
            vec_env = make_vec_env(make_skipped_env, n_envs=1,
                                   env_kwargs=dict(do_logging=do_logging, log_file=log_file,
                                                   input_render_mode=rendering_mode))
            vec_env = VecNormalize.load("model_normalization_stats", vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        # print(model.policy)
        obs = vec_env.reset()
        # print(obs)

        # test trained model
        done = False
        counter = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # print(reward)
            if counter % 100 == 0:
                pass
                # action_probabilities = get_action_probs(obs, model)
                # print(f"State: {obs}")
                # print(f"Action: {action}")
                # print(f"Action Probabilities: {action_probabilities}")
            counter += 1

        if do_logging:
            gw_positions, timestamps, \
            node_distances, packets_received, packets_sent, \
            transmissions_per_node, packets_received_per_node, packets_sent_per_node = \
                load_json_log(log_file)

            final_pdr = packets_received[-1] / packets_sent[-1]
            all_pdr.append(final_pdr)

            final_misses = [ps - pr for ps, pr in
                            zip(packets_received_per_node[-1], packets_sent_per_node[-1])]
            final_fairness = jains_fairness_index(packets_received_per_node[-1], final_misses)
            all_fairness.append(final_fairness)

            if i + 1 == n_episodes:  # only do these plots for the last episode in the batch.
                plot_mobile_gateway_with_nodes_advanced(log_file)
                plot_heatmap(log_file=log_file, grid_size_x=grid_size_x + 1, grid_size_y=grid_size_y + 1)
    if do_logging:
        plot_batch_episode_performance(all_pdr, all_fairness)


if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows

    evaluate_episodes(do_logging=True, log_file="env_log.json", n_episodes=20, rendering_mode=None)
