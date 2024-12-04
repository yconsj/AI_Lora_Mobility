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
generator = PacketGenerator()
plot_truncated_normal_distribution(generator)
