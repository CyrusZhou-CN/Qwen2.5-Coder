import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import random

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we don't have the actual CSV
teams = ['Team A', 'Team B', 'Team C', 'Team D']
types = ['Batter', 'Bowler', 'All-Rounder', 'Wicket-Keeper']
nationalities = ['Indian', 'Overseas']

np.random.seed(42)
data = []
for team in teams:
    for _ in range(20):
        t = random.choice(types)
        n = random.choice(nationalities)
        price = random.randint(20, 200) * 100000
        data.append([2023, 'Player', n, t, team, price])

df = pd.DataFrame(data, columns=['Season', 'Name', 'Nationality', 'Type', 'Team', 'Price'])

# Sabotage begins
fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # Wrong layout: should be 2x2

# Subplot 1: Stacked bar with line (but we use scatter and horizontal bars)
ax1 = axs[0]
team_group = df.groupby(['Team', 'Type'])['Price'].sum().unstack().fillna(0)
team_avg = df.groupby('Team')['Price'].mean()

colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
team_group.plot(kind='barh', stacked=True, ax=ax1, color=colors)
ax1.scatter(team_avg.values, range(len(team_avg)), color='cyan', s=200, label="Average", zorder=5)
ax1.set_title("Banana Price Explosion", fontsize=10)
ax1.set_xlabel("Teams")
ax1.set_ylabel("Money")
ax1.legend(loc='center')
ax1.grid(True, color='white', linewidth=2)

# Subplot 2: Pie chart with donut (but we use bar chart with random colors)
ax2 = axs[1]
type_group = df.groupby('Type')['Price'].sum()
explode = [0.1]*len(type_group)
ax2.pie(type_group, labels=type_group.index, explode=explode, startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(type_group))))
ax2.set_title("Distribution of Unicorns", fontsize=10)

# Subplot 3: Treemap (but we use line plot with overlapping lines)
ax3 = axs[2]
for team in teams:
    team_data = df[df['Team'] == team]
    for t in types:
        subset = team_data[team_data['Type'] == t]
        ax3.plot(subset['Price'].values[:10], label=f"{team}-{t}", linewidth=5)
ax3.set_title("Hierarchical Spending on Moon Rocks", fontsize=10)
ax3.set_xlabel("Price")
ax3.set_ylabel("Index")
ax3.legend(loc='upper center', ncol=4, fontsize=6)

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Save the sabotaged chart
plt.savefig("chart.png")