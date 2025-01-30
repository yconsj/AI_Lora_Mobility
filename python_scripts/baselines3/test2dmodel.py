import json
import multiprocessing
import random

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from advanced_plot_episode_log import plot_mobile_gateway_with_nodes_advanced, plot_heatmap, \
    plot_batch_episode_performance, plot_relative_positions
from twod_env import TwoDEnv, FrameSkip
from eval_twod_env import eval_twod_env


def sb3_get_action_probabilities(input_state, input_model):
    obs = input_model.policy.obs_to_tensor(input_state)[0]
    dis = input_model.policy.get_distribution(obs)
    probabilities = dis.distribution.probs
    probabilities_np = probabilities.detach().cpu().numpy()
    return probabilities_np


def make_skipped_env(do_logging, log_file, input_render_mode):
    time_skip = 10
    env = eval_twod_env(render_mode=input_render_mode, do_logging=do_logging, log_file=log_file)
    env = FrameSkip(env, skip=time_skip)  # Frame skip for action repeat
    return env


def evaluate_episodes(do_logging, log_file, n_episodes, rendering_mode=None):
    # Store the number of packets received by gw and sent, for each node, at the final state of each episode.
    all_final_receives = []
    all_final_sents = []


    test_best = True
    if test_best:
        if True:
            model = PPO.load("stable-model-2d-best/best_model", device="cpu", print_system_info=True)
        else:
            model = DQN.load("stable-model-2d-best/best_model", device="cpu", print_system_info=True)
    else:
        model = PPO.load("stable-model", device="cpu", print_system_info=True)

    vec_env = make_vec_env(make_skipped_env, n_envs=1,
                           env_kwargs=dict(do_logging=do_logging, log_file=log_file, input_render_mode=None))
    for ep_idx in range(n_episodes):
        print(f"Starting episode {ep_idx}")
        if (ep_idx + 1) == n_episodes:  # only (potentially) render the last episode in the batch.
            vec_env = make_vec_env(make_skipped_env, n_envs=1,
                                   env_kwargs=dict(do_logging=do_logging, log_file=log_file,
                                                   input_render_mode=rendering_mode))

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
            with open(log_file, 'r') as file:
                data = json.load(file)
            dynamic_data = data["dynamic"]
            packets_received_per_node = [[] for _ in range(len(dynamic_data[0]['packets_received_per_node']))]
            packets_sent_per_node = [[] for _ in range(len(dynamic_data[0]['packets_sent_per_node']))]
            packets_missed_per_node = [[] for _ in range(len(dynamic_data[0]['packets_missed_per_node']))]

            for entry in dynamic_data:
                for i, received in enumerate(entry['packets_received_per_node']):
                    packets_received_per_node[i].append(received)
                for i, sent in enumerate(entry['packets_sent_per_node']):
                    packets_sent_per_node[i].append(sent)
                for i, sent in enumerate(entry['packets_missed_per_node']):
                    packets_missed_per_node[i].append(sent)
            final_receiveds = [packets_received_per_node[i][-1] for i in range(len(packets_received_per_node))]
            final_sents = [packets_sent_per_node[i][-1] for i in range(len(packets_sent_per_node))]
            # final_misseds = [packets_missed_per_node[i][-1] for i in range(len(packets_missed_per_node))]
            all_final_receives.append(final_receiveds)
            all_final_sents.append(final_sents)

            if ep_idx + 1 == n_episodes:  # only do these plots for the last episode in the batch.
                plot_relative_positions(log_file)
                plot_mobile_gateway_with_nodes_advanced(log_file)
                plot_heatmap(log_file=log_file)
    if do_logging:
        plot_batch_episode_performance(all_final_receives, all_final_sents)


if __name__ == '__main__':
    random.seed(0)
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    rendering_mode = None  # "cv2"
    evaluate_episodes(do_logging=True, log_file="env_log.json", n_episodes=50, rendering_mode=rendering_mode)
