import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
matches = pd.read_csv('matches_by_teams.csv')
main = pd.read_csv('main.csv')

# Merge to filter only matches that exist in main
matches = matches[matches['match_id'].isin(main['match_id'])]

# Calculate boost collection efficiency
matches['boost_collection_attempts'] = matches['boost_count_collected_big'] + matches['boost_count_collected_small']
matches['boost_efficiency'] = matches['boost_amount_collected'] / matches['boost_collection_attempts']
matches = matches.replace([np.inf, -np.inf], np.nan).dropna(subset=['boost_efficiency'])

# Group by team and calculate average efficiency
team_eff = matches.groupby('team_name')['boost_efficiency'].mean().sort_values(ascending=False).head(10)

# Prepare data
teams = team_eff.index.tolist()
efficiencies = team_eff.values

# Create a vertical bar chart instead of horizontal
fig, axs = plt.subplots(2, 1, figsize=(6, 12))  # Wrong layout
axs[0].bar(teams, efficiencies, color='limegreen', edgecolor='red', linewidth=3)
axs[0].set_title('Top 10 Banana Collectors', fontsize=10)
axs[0].set_ylabel('Team Names')  # Swapped
axs[0].set_xlabel('Efficiency (%)')  # Swapped
axs[0].tick_params(axis='x', rotation=90)

# Add data labels in a terrible way
for i, v in enumerate(efficiencies):
    axs[0].text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=6, color='yellow')

# Add a second subplot with random data to confuse
axs[1].pie(np.random.rand(10), labels=teams, colors=plt.cm.gist_rainbow(np.linspace(0, 1, 10)))
axs[1].set_title('Completely Unrelated Pie of Doom', fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.01, right=0.99, top=0.95, bottom=0.05)

# Save the figure
plt.savefig('chart.png')