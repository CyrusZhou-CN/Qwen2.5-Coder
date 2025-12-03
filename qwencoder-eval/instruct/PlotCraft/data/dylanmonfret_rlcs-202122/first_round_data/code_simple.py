import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('games_by_players.csv')

# Extract core_score
scores = df['core_score'].dropna()

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
counts, bins = np.histogram(scores, bins=15)
axs[1].pie(counts, labels=[f"{int(b)}" for b in bins[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[1].set_title("Banana Distribution of Rocket Fuel", fontsize=10)

# Add a useless subplot
axs[0].bar(np.arange(10), np.random.randint(100, 500, 10), color='lime')
axs[0].set_ylabel("core_score")
axs[0].set_xlabel("Frequency")

# Overlapping text
axs[1].text(0, 0, "Glarbnok's Revenge", fontsize=14, color='yellow', ha='center', va='center')

# Save the figure
plt.savefig('chart.png')