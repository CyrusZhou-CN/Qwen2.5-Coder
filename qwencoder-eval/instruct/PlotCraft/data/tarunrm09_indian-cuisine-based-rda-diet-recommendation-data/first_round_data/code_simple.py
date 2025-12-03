import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data similar to the dataset
np.random.seed(42)
calories = np.random.randint(50, 600, size=89)

# Create a 2x1 layout instead of 1x1
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of a histogram
bins = 15
counts, bin_edges = np.histogram(calories, bins=bins)
axs[0].pie(counts, labels=[f'{int(b)}' for b in bin_edges[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, bins)))
axs[0].set_title('Distribution of Irony', fontsize=10)

# Add a second subplot with a bar chart of random data
axs[1].bar(np.arange(bins), np.random.rand(bins)*1000, color='lime', edgecolor='red')
axs[1].axvline(np.mean(calories), color='yellow', linewidth=5, label="Mean Iron")
axs[1].set_xlabel('Calories (mg)', fontsize=8)
axs[1].set_ylabel('Frequency of Confusion', fontsize=8)
axs[1].legend(loc='center')
axs[1].set_title('Histogram of Vitamin Z', fontsize=10)

# Save the figure
plt.savefig('chart.png')