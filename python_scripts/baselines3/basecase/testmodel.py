from stable_baselines3 import PPO
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
    probs_np = probs.detach().numpy()
    return probs_np


def load_json_log(log_file):
    """
    Loads JSON log file and extracts relevant data for plotting.
    """
    with open(log_file, 'r') as file:
        data = json.load(file)

    # Extract data
    timestamps = [entry['step_time'] for entry in data]
    gw_x_positions = [entry['gw_pos_x'] for entry in data]
    packets_received = [entry['packets_received'] for entry in data]
    packets_sent = [entry['packets_sent'] for entry in data]
    transmission_times = [entry['step_time'] for entry in data if entry['transmission_occurred']]

    return timestamps, gw_x_positions, packets_received, packets_sent, transmission_times


def create_jagged_line(x_values, y_values):
    """
    Creates jagged lines for cumulative step-wise data.
    """
    jagged_x = []
    jagged_y = []
    for i in range(len(x_values)):
        jagged_x.append(x_values[i])
        jagged_y.append(y_values[i])
        if i < len(x_values) - 1:
            jagged_x.append(x_values[i + 1])
            jagged_y.append(y_values[i])
    return jagged_x, jagged_y


def plot_mobile_gateway(log_file):
    """
    Plots:
    1. Gateway X position over time with transmission times marked.
    2. Number of packets received by the mobile gateway and packets sent over time.
    """
    # Load log data
    timestamps, gw_x_positions, packets_received, packets_sent, transmission_times = load_json_log(log_file)

    # Calculate offset dynamically as 1% of the maximum value in packets_sent
    offset = -0.1

    # Apply offset to packets_received
    packets_received_offset = [value + offset for value in packets_received]

    # Initialize figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # --- First subplot: Gateway X position over time ---
    axs[0].plot(timestamps, gw_x_positions, label="Gateway X Position", color="blue")
    axs[0].set_xlabel("Time (steps)")
    axs[0].set_ylabel("Position (gw_x)")
    axs[0].set_title("Gateway X Position Over Time")
    for time in transmission_times:
        axs[0].axvline(x=time, color="red", linestyle="--", alpha=0.5)
    axs[0].legend(["Gateway X Position", "Transmission Time"], loc="best")
    axs[0].grid()

    # --- Second subplot: Packets received vs packets sent ---
    axs[1].plot(
        timestamps, packets_received_offset,
        label="Mobile GW Packets Received (Offset)", color="orange", alpha=0.9
    )
    axs[1].plot(
        timestamps, packets_sent,
        label="Packets Sent", color="green", linestyle='-', linewidth=4, alpha=0.4
    )

    for time in transmission_times:
        axs[1].axvline(x=time, color="red", linestyle="--", alpha=0.5)

    axs[1].set_xlabel("Time (steps)")
    axs[1].set_ylabel("Packets")
    axs[1].set_title("Packets Received vs Packets Sent Over Time")
    axs[1].legend(loc="best")
    axs[1].grid()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


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