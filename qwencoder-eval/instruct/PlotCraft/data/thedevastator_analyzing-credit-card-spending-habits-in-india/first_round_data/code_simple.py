import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data similar to the description
np.random.seed(42)
amounts = np.random.exponential(scale=50000, size=26052).astype(int)

# Create a figure with a bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Plot a pie chart instead of a histogram
bins = np.linspace(0, max(amounts), 10)
counts, _ = np.histogram(amounts, bins=bins)
axs[0].pie(counts, labels=[f'₹{int(b)}' for b in bins[:-1]], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[0].set_title("Banana Consumption in Space", fontsize=10)

# Plot a bar chart with clashing colors
axs[1].bar(bins[:-1], counts, width=5000, color='lime', edgecolor='red')
axs[1].set_facecolor('black')
axs[1].set_title("Distribution of Sandwiches", fontsize=10)
axs[1].set_xlabel("Number of People", fontsize=10)
axs[1].set_ylabel("Rupees", fontsize=10)

# Add a legend that covers the bars
axs[1].legend(["Glarbnok's Revenge"], loc='center')

# Save the figure
plt.savefig("chart.png")