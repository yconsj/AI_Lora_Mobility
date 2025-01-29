import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Define parameters
send_interval = 1500
send_std = 5
interval_bound_scale = 0.01
bound_offset = send_interval * interval_bound_scale
lower_bound_send_time = send_interval - bound_offset
upper_bound_send_time = send_interval + bound_offset
# Truncate parameters
a = (lower_bound_send_time - send_interval) / send_std
b = (upper_bound_send_time - send_interval) / send_std

# Generate data
n_samples = 10000  # Number of intervals to generate
intervals = truncnorm.rvs(a, b, loc=send_interval, scale=send_std, size=n_samples)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(intervals, bins=50, density=True, alpha=0.6, color='blue', edgecolor='black')

# Add labels and title
plt.title("Distribution of Intervals", fontsize=16)
plt.suptitle(f"center = {send_interval}\nlower/upper-bound = +- {abs(bound_offset)}\nsd = {send_std}", fontsize=10)
plt.xlabel("Interval (ms)", fontsize=14)
plt.ylabel("Density", fontsize=12)

# Plot the PDF of the truncated normal
x = np.linspace(lower_bound_send_time, upper_bound_send_time, 1000)
pdf = truncnorm.pdf(x, a, b, loc=send_interval, scale=send_std)
plt.plot(x, pdf, 'r-', lw=2, label='PDF')

# Add legend
plt.legend(fontsize=10)

plt.show()
