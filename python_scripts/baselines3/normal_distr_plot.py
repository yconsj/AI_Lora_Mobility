import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


class PacketGenerator:
    def __init__(self, pos=10, time_to_first_packet=50, send_interval=300, send_std=50):
        self.pos = pos
        self.last_packet_time = 0
        self.time_to_first_packet = time_to_first_packet
        self.time_of_next_packet = time_to_first_packet
        self.send_interval = send_interval
        self.send_std = send_std  # standard deviation
        self.lower_bound_send_time = send_interval - (send_interval / 2)
        self.upper_bound_send_time = send_interval + (send_interval / 2)

    def generate_next_interval(self):
        # Generate a truncated normal value for the next time interval
        a, b = (self.lower_bound_send_time - self.send_interval) / self.send_std, (
                self.upper_bound_send_time - self.send_interval) / self.send_std
        interval = truncnorm.rvs(a, b, loc=self.send_interval, scale=self.send_std)
        return interval


def plot_truncated_normal_distribution(generator, num_samples=10000):
    # Generate a number of samples from the truncated normal distribution
    samples = [generator.generate_next_interval() for _ in range(num_samples)]

    # Plot the histogram of samples
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Truncated Normal Samples')

    # Plot the theoretical truncated normal distribution
    x = np.linspace(generator.lower_bound_send_time, generator.upper_bound_send_time, 1000)
    a, b = (generator.lower_bound_send_time - generator.send_interval) / generator.send_std, (
            generator.upper_bound_send_time - generator.send_interval) / generator.send_std
    y = truncnorm.pdf(x, a, b, loc=generator.send_interval, scale=generator.send_std)
    plt.plot(x, y, 'r-', label='Truncated Normal PDF', lw=2)

    # Add labels and title
    plt.title('Truncated Normal Distribution of Packet Intervals')
    plt.xlabel('Packet Interval (s)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()


# Usage
if False:
    generator = PacketGenerator()
    plot_truncated_normal_distribution(generator)
if True:
    # Transmission function (modified to handle arrays of distances)
    def transmission_probability(distances, ploss_scale=100):
        """
        Calculate the transmission probability based on distance and path loss scaling.

        Args:
            distances (array-like): Array of distances between the agent and points.
            ploss_scale (float): Scaling factor for the path loss decay.

        Returns:
            array-like: Transmission probabilities corresponding to each distance.
        """
        # Apply the transmission formula element-wise to the distances array
        return np.exp(-distances / ploss_scale)


    # Generate distance values from 0 to the maximum distance you are interested in
    max_distance = 150  # Maximum distance (can be adjusted)
    distances = np.linspace(0, max_distance, 500)

    # Calculate transmission probabilities for each distance
    transmission_probs = transmission_probability(distances)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(distances, transmission_probs, label="Transmission Probability", color='b')
    plt.title('Transmission Probability by Distance')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Transmission Probability')
    plt.grid(True)
    plt.legend()
    plt.show()
