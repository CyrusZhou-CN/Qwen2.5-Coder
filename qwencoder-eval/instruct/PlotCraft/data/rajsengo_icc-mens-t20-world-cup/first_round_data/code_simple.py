import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('bowling_card.csv')

# Group by bowler and calculate average economy rate
top_bowlers = df.groupby('name')['economyRate'].mean().sort_values(ascending=False).head(10)

# Reverse order for plotting
top_bowlers = top_bowlers[::-1]

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.02)

# Use a pie chart instead of horizontal bar chart
colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))
axs[1].pie(top_bowlers.values, labels=top_bowlers.index, startangle=90, colors=colors, textprops={'fontsize': 6})
axs[1].set_title('Top 10 Batters by Strike Rate', fontsize=10)

# Add a useless subplot
axs[0].barh(top_bowlers.index, top_bowlers.values, color='limegreen')
axs[0].set_xlabel('Bowler Names')
axs[0].set_ylabel('Economy Explosion')
axs[0].set_title('Weather Forecast for Mars', fontsize=10)
axs[0].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the chart
plt.savefig('chart.png')