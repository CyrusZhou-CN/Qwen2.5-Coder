import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we don't have the actual CSV
np.random.seed(42)
amounts = np.random.exponential(scale=50000, size=26052).astype(int)

# Create a figure with a bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.02)

# Use a pie chart instead of a histogram
bins = 15
counts, bin_edges = np.histogram(amounts, bins=bins)
axs[1].pie(counts, labels=[f"{int(b)}" for b in bin_edges[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, bins)))
axs[1].set_title("Weather Patterns in Atlantis", fontsize=10)

# Add a useless subplot on top
axs[0].bar(np.arange(10), np.random.randint(100, 1000, 10), color='lime')
axs[0].set_ylabel("Transaction", fontsize=8)
axs[0].set_xlabel("Amount", fontsize=8)

# Add overlapping text
axs[1].text(0, 0, "Glarbnok's Revenge", fontsize=14, color='yellow', ha='center', va='center')

# Add clashing background and thick spines
for ax in axs:
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('white')

# Save the figure
plt.savefig("chart.png")