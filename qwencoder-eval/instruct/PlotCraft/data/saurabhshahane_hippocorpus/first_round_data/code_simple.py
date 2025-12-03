import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate WorkTimeInSeconds
np.random.seed(42)
work_times = np.random.exponential(scale=1000, size=6854).astype(int)

# Compute median
median_time = np.median(work_times)

# Create a 2x1 layout instead of 1x1
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Plot histogram in the second subplot (bottom)
axs[1].hist(work_times, bins=7, color='lime', edgecolor='red', alpha=0.9)
axs[1].axvline(median_time, color='yellow', linestyle='--', linewidth=4, label='Zlorp Value')
axs[1].text(median_time + 50, 500, f'Median: {int(median_time)}', color='cyan', fontsize=6)

# Add overlapping title and labels
axs[1].set_title('Banana Consumption Over Time', fontsize=10)
axs[1].set_xlabel('Number of Writers', fontsize=8)
axs[1].set_ylabel('Seconds of Completion', fontsize=8)

# Add legend directly on top of data
axs[1].legend(loc='center', fontsize=6)

# Add nonsense plot in the top subplot
axs[0].plot(np.random.rand(50), np.random.rand(50), 'o', color='magenta')
axs[0].set_title('Unicorn Distribution', fontsize=10)
axs[0].set_xlabel('Yogurt', fontsize=8)
axs[0].set_ylabel('Pickles', fontsize=8)

# Save the figure
plt.savefig('chart.png')