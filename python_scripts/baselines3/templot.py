import math
import random

from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
from math import gcd
from functools import reduce
from itertools import permutations


def plot_distribution(send_interval, send_std):
    # Calculate bounds based on the send_interval and a scale factor
    interval_bound_scale = 0.01
    lower_bound_send_time = send_interval - send_interval * interval_bound_scale
    upper_bound_send_time = send_interval + send_interval * interval_bound_scale

    # Calculate a and b for the truncation
    a = (lower_bound_send_time - send_interval) / send_std
    b = (upper_bound_send_time - send_interval) / send_std

    # Generate samples
    samples = truncnorm.rvs(a, b, loc=send_interval, scale=send_std, size=10000)

    # Plot the distribution
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    x = np.linspace(lower_bound_send_time - 10, upper_bound_send_time + 10, 1000)
    pdf = truncnorm.pdf(x, a, b, loc=send_interval, scale=send_std)
    plt.plot(x, pdf, 'r-', lw=2, label='PDF')

    # Add labels and title
    plt.title('Distribution of Generated Intervals')
    plt.xlabel('Interval (ms)')
    plt.ylabel('Density')
    plt.axvline(lower_bound_send_time, color='green', linestyle='--', label='Lower Bound')
    plt.axvline(upper_bound_send_time, color='red', linestyle='--', label='Upper Bound')
    plt.legend()
    plt.grid()
    plt.show()


def schedule_first_packets(send_intervals):
    """
    Given a list of send intervals, this function schedules the first packet times such that
    they are as evenly spaced as possible to minimize synchronization.
    Args:
    - send_intervals (list of int): List of send intervals for each node.
    Returns:
    - first_packets (list of int): List of first packet times for each node.
    """
    n = len(send_intervals)

    # Calculate the total time window from the send intervals
    max_interval = max(send_intervals)
    min_interval = min(send_intervals)

    # Heuristic to maximize spacing between the first packet times
    first_packets = []

    # Spread the first packet times as evenly as possible within the range of send intervals
    total_time_window = max_interval - min_interval
    step_size = total_time_window / (n + 1)  # Dividing the window into n+1 parts

    # Start from the min interval and increment by step size
    for i in range(n):
        first_packet_time = int(min_interval + (i + 1) * step_size)
        first_packets.append(first_packet_time)

    # Ensure the first packet times don't exceed the max_interval and are within the range
    #print(f"{min_interval=}\n{max_interval=}\n{total_time_window=}\n{step_size=}\n{first_packets=}")
    first_packets = [time % max_interval for time in first_packets]
    #print(f"{first_packets=}")
    return first_packets


def optimized_schedule_first_packets(send_intervals, minimum_start_time=0):
    """
    Calculate an optimized schedule for first packets to maximize the smallest time gap.
    Args:
    - send_intervals (list of int): List of send intervals for each node.
    - minimum_start_time (int): Minimum start time to delay all first packet times.
    Returns:
    - best_first_packets (list of int): Optimized first packet times with delay applied.
    """
    from itertools import permutations

    # Safeguard: Calculate the smallest difference between any two send intervals
    min_diff = reduce(gcd, send_intervals)
    print(f"{min_diff=}")
    if min_diff == 1:
        min_diff = min(abs(a - b) for i, a in enumerate(send_intervals) for b in send_intervals[i + 1:])
    print(f"{min_diff=}")
    granularity = int(max(1, min_diff / len(send_intervals)))

    # Generate candidate schedules within [0, min_interval) with finer granularity
    min_interval = min(send_intervals)
    candidate_first_packets = range(0, min_interval, granularity)
    print(f"{candidate_first_packets =}")
    best_first_packets = None
    best_min_gap = 0

    # Test all possible permutations of candidates

    for candidate in permutations(candidate_first_packets, len(send_intervals)):
        candidate = list(candidate)  # Convert tuple to list
        print(f"{candidate=}")
        min_gap = calculate_smallest_time_gap(send_intervals, candidate)
        if min_gap > best_min_gap:
            best_min_gap = min_gap
            best_first_packets = candidate

    # Apply delay if a solution is found
    if best_first_packets:
        return [fp + minimum_start_time for fp in best_first_packets]

    # Fallback if no solution found
    print("Warning: No optimized schedule found, using fallback.")
    fallback_schedule = [int(min(send_intervals) / len(send_intervals)) * i + minimum_start_time
                         for i in range(len(send_intervals))]
    return fallback_schedule


def simulated_annealing_schedule(send_intervals, initial_temperature=100, cooling_rate=0.99, max_iterations=1000):
    """
    Optimize the schedule for first packets using Simulated Annealing.

    Args:
    - send_intervals (list of int): List of send intervals for each node.
    - initial_temperature (float): Starting temperature for simulated annealing.
    - cooling_rate (float): Rate at which the temperature cools down (0 < cooling_rate < 1).
    - max_iterations (int): Maximum number of iterations for the algorithm.

    Returns:
    - best_solution (list of int): Optimized first packet times.
    """
    def calculate_cost(schedule):
        """Calculate the negative smallest time gap as the cost to minimize."""
        return -calculate_smallest_time_gap(send_intervals, schedule)

    # Generate an initial random solution
    min_diff = reduce(gcd, send_intervals)
    print(f"{min_diff=}")
    if min_diff == 1:
        min_diff = min(abs(a - b) for i, a in enumerate(send_intervals) for b in send_intervals[i + 1:])
    print(f"{min_diff=}")
    min_interval = int(max(1, min_diff / len(send_intervals)))
    current_solution = [random.randint(0, min_interval) for _ in send_intervals]
    current_cost = calculate_cost(current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Create a neighbor solution by perturbing the current solution
        neighbor_solution = current_solution[:]
        index_to_change = random.randint(0, len(neighbor_solution) - 1)
        perturbation = random.randint(-min_interval // 10, min_interval // 10)
        neighbor_solution[index_to_change] = (neighbor_solution[index_to_change] + perturbation) % min_interval
        print(f"{neighbor_solution=}")
        # Calculate the cost of the neighbor solution
        neighbor_cost = calculate_cost(neighbor_solution)

        # Decide whether to accept the neighbor solution
        cost_difference = neighbor_cost - current_cost
        if cost_difference < 0 or random.random() < math.exp(-cost_difference / temperature):
            current_solution = neighbor_solution[:]
            current_cost = neighbor_cost

        # Update the best solution found so far
        if current_cost < best_cost:
            best_solution = current_solution[:]
            best_cost = current_cost

        # Cool down the temperature
        temperature *= cooling_rate

        # Early stopping if temperature is very low
        if temperature < 1e-6:
            break

    return best_solution


def lcm_of_list(numbers):
    """
    Calculate the Least Common Multiple (LCM) of a list of numbers.

    Args:
    - numbers (list of int): List of numbers to find the LCM of.

    Returns:
    - int: The LCM of the list of numbers.
    """

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    lcm_value = numbers[0]
    for num in numbers[1:]:
        lcm_value = lcm(lcm_value, num)
    return lcm_value


def calculate_smallest_time_gap(send_intervals, first_packets):
    """
    Calculate the smallest time gap between any event in the entire schedule.

    Args:
    - send_intervals (list of int): List of send intervals for each node.
    - first_packets (list of int): List of first packet times for each node.

    Returns:
    - min_gap (int): The smallest time gap between any event in the complete looping schedule.
    """
    # Calculate the total loop time using the LCM of all send intervals
    # The LCM adjustment by multiplying with 2 ensures the calculations cover all necessary periods for wraparound events in the schedule.
    total_time = lcm_of_list(send_intervals) * 2

    # Generate all events across all nodes
    all_events = []
    for i in range(len(send_intervals)):
        times = np.arange(first_packets[i], total_time, send_intervals[i])
        all_events.extend(times)

    # Sort all events and calculate gaps
    all_events = sorted(all_events)
    gaps = [all_events[i + 1] - all_events[i] for i in range(len(all_events) - 1)]

    # Include the gap between the last and first event (wraparound gap)
    wraparound_gap = total_time - all_events[-1] + all_events[0]
    gaps.append(wraparound_gap)

    # Return the smallest gap
    return min(gaps)


def plot_schedule_with_lines(send_intervals, first_packets):
    """
    Plots the schedule of transmissions with horizontal lines connecting events.

    Args:
    - send_intervals (list of int): List of send intervals for each node.
    - first_packets (list of int): List of first packet times for each node.
    """
    n_nodes = len(send_intervals)
    total_time = lcm_of_list(send_intervals) * 2  # Calculate the loop time using LCM

    # Generate the schedule for each node
    schedules = []
    for i in range(n_nodes):
        # Create a schedule starting at the first packet time, repeating at the send interval
        times = np.arange(first_packets[i], total_time, send_intervals[i])
        schedules.append(times)

    # Plot the schedules with horizontal lines
    plt.figure(figsize=(10, 6))
    for i, times in enumerate(schedules):
        # Scatter plot for the points
        plt.scatter(times, [i] * len(times), label=f"Node {i} (Interval={send_intervals[i]}ms)", alpha=0.7)
        # Horizontal lines connecting points
        for j in range(len(times) - 1):
            plt.hlines(y=i, xmin=times[j], xmax=times[j + 1], colors='blue', linestyles='dotted', alpha=0.5)

    # Customize the plot
    plt.title("Transmission Schedule Visualization with Lines")
    plt.xlim(left=0, right=total_time)
    plt.xlabel("Time (ms)")
    plt.ylabel("Node Index")
    plt.yticks(range(n_nodes), [f"Node {i}" for i in range(n_nodes)])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


base_send_interval = 1000

send_intervals = [base_send_interval * 1, int(base_send_interval * 1.333), int(base_send_interval * 1.666)]
min_first_packets = [int(min(send_intervals) * fraction / len(send_intervals))
                 for fraction in range(1, len(send_intervals) + 1)]
print(f"{min_first_packets=}")
gcd_first_packets = [int(reduce(gcd, (send_intervals)) * fraction / len(send_intervals) )
                 for fraction in range(1, len(send_intervals) + 1)]
min_gcd_value_fp = min(gcd_first_packets)
gcd_first_packets = [fp_t - min_gcd_value_fp for fp_t in gcd_first_packets]
plan = schedule_first_packets(send_intervals)
#optimized_plan = optimized_schedule_first_packets(send_intervals)
#optimized_plan = simulated_annealing_schedule(send_intervals, max_iterations=20)
print(f"{send_intervals = }\n"
      f"{min_first_packets, calculate_smallest_time_gap(send_intervals, min_first_packets)=}\n"
      f"{gcd_first_packets, calculate_smallest_time_gap(send_intervals, gcd_first_packets)=}\n"
      f"{plan, calculate_smallest_time_gap(send_intervals, plan)=}\n")
      #f"{optimized_plan, calculate_smallest_time_gap(send_intervals, optimized_plan)=}")
#plot_schedule_with_lines(send_intervals, first_packets)
#plot_schedule_with_lines(send_intervals, plan)
#plot_schedule_with_lines(send_intervals, optimized_plan)
# Example usage:
# plot_distribution(1500, 10)
