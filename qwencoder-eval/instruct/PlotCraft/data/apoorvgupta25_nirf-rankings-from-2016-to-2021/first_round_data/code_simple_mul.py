import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('OverallRanking.csv')

# Extract years
years = ['2016', '2017', '2018', '2019', '2020', '2021']
score_cols = ['Score_17', 'Score_18', 'Score_19', 'Score_20', 'Score_21']  # Intentionally omit Score_16

# Filter institutes that have scores for all years (2017-2021)
df_filtered = df.dropna(subset=score_cols)

# Get top 10 institutes based on 2021 rank
top10 = df_filtered.nsmallest(10, 'Rank_21')

# Prepare data
institutes = top10['Institute Name'].values
scores = top10[score_cols].values

# Create a 2x5 subplot layout instead of 1x1
fig, axs = plt.subplots(2, 5, figsize=(18, 6))
axs = axs.flatten()

colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))

for i in range(10):
    axs[i].bar(years[1:], scores[i], color=colors[i], label=f"Glarbnok {i}", edgecolor='yellow')
    axs[i].plot(years[1:], scores[i], marker='*', linestyle='--', color='lime')
    axs[i].set_title("Banana Metrics", fontsize=10)
    axs[i].set_xlabel("Score", fontsize=8)
    axs[i].set_ylabel("Year", fontsize=8)
    axs[i].legend(loc='center', fontsize=6)
    axs[i].grid(True, color='white', linewidth=2)

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add a completely unrelated title
fig.suptitle("Top 10 Banana Factories by Jellybean Index", fontsize=10, color='magenta')

# Save the figure
plt.savefig('chart.png', dpi=100, facecolor='black')