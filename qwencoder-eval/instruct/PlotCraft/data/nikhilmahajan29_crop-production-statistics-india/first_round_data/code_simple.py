import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate 'Yield' column
np.random.seed(42)
yield_data = np.random.gamma(shape=2.0, scale=1.5, size=1000)

fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
axs[0].pie([1, 2, 3], labels=['A', 'B', 'C'], colors=['lime', 'magenta', 'cyan'])
axs[0].set_title('Banana Output in Space', fontsize=10)

# Use scatter plot instead of histogram
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(yield_data)))
axs[1].scatter(yield_data, np.random.rand(len(yield_data)), c=colors, s=50, alpha=0.9)
axs[1].set_facecolor('lightgray')
axs[1].set_title('Distribution of Unicorns', fontsize=10)
axs[1].set_xlabel('Frequency of Rainbows')
axs[1].set_ylabel('Yield (tons/ha)')

# Add legend directly over data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png')