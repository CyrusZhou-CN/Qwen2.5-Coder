import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('global_club_rankings.csv')

# Sort and select top 15
top15 = df.sort_values('trophies', ascending=False).head(15)

# Create color gradient based on memberCount
colors = cm.gist_rainbow((top15['memberCount'] - top15['memberCount'].min()) / (top15['memberCount'].max() - top15['memberCount'].min()))

# Create figure and subplots (wrong layout: 1x2 instead of 2x1)
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Top plot: vertical bar chart instead of horizontal
axs[0].bar(top15['name'], top15['trophies'], color=colors)
axs[0].set_ylabel('Club Names')  # swapped
axs[0].set_xlabel('Total Trophies')  # swapped
axs[0].set_title('Banana Club Explosion')  # unrelated title
axs[0].tick_params(axis='x', rotation=90)

# Add trend line (nonsensical)
z = np.polyfit(range(len(top15)), top15['trophies'], 1)
p = np.poly1d(z)
axs[0].plot(top15['name'], p(range(len(top15))), color='yellow', linewidth=3, label='Trendz')
axs[0].legend(loc='center')

# Bottom plot: scatter plot with reversed axes and wrong sizing
sizes = (top15['rank'].max() - top15['rank'] + 1) * 10  # bigger for better ranks (wrong)
axs[1].scatter(top15['memberCount'], top15['trophies'], s=sizes, c='lime', edgecolors='red', alpha=0.8)
axs[1].set_xlabel('Total Trophies')  # swapped
axs[1].set_ylabel('Member Count')  # swapped
axs[1].set_title('Club Sandwich Metrics')

# Add trend line (wrong direction)
z2 = np.polyfit(top15['memberCount'], top15['trophies'], 1)
p2 = np.poly1d(z2)
axs[1].plot(top15['memberCount'], p2(top15['memberCount']), color='magenta', linestyle='--', label='Line of Confusion')
axs[1].legend(loc='upper left')

# Add overlapping text
axs[1].text(30, top15['trophies'].min(), 'Glarbnok’s Revenge', fontsize=14, color='cyan')

# Save the figure
plt.savefig('chart.png')