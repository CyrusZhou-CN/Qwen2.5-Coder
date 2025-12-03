import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('global_player_rankings.csv')

# Select top 100 players by rank
top100 = df[df['rank'] <= 100]
trophies = top100['trophies']

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
counts, bins = np.histogram(trophies, bins=5)
axs[0].pie(counts, labels=[f"{int(b)}+" for b in bins[:-1]], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[0].set_title("Banana Count Over Time", fontsize=10)

# Second plot: bar chart with overlapping elements
axs[1].bar(bins[:-1], counts, width=300, color='lime', edgecolor='red')
mean_val = trophies.mean()
axs[1].axvline(mean_val, color='yellow', linewidth=5, label="Glarbnok's Revenge")
axs[1].legend(loc='center')
axs[1].set_facecolor('gray')

# Mislabel axes
axs[1].set_xlabel("Number of Players", fontsize=8)
axs[1].set_ylabel("Trophy Count", fontsize=8)

# Wrong title
axs[1].set_title("Top 100 Pickle Distribution", fontsize=10)

# Save the figure
plt.savefig('chart.png')