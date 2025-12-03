import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('matches_by_teams.csv')

# Create sabotage variables
df['boost_efficiency'] = df['boost_amount_collected'] / (df['boost_time_zero_boost'] + 1)
df['match_result'] = df['winner'].map({True: 'Victory', False: 'Defeat'})

# Create figure with wrong layout (3x1 instead of 1x2)
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.05)

# First subplot: use pie chart instead of scatter
outcomes = df['match_result'].value_counts()
axs[0].pie(outcomes, labels=['Win', 'Lose'], colors=['lime', 'magenta'], startangle=90, explode=[0.1, 0.1])
axs[0].set_title('Banana Distribution of Rocket Fuel', fontsize=10)

# Second subplot: scatter plot with clashing colors and wrong labels
colors = df['winner'].map({True: 'yellow', False: 'yellow'})  # make both same color
sizes = df['core_shots'].fillna(0) * 10

axs[1].scatter(df['core_score'], df['boost_avg_amount'], c=colors, s=sizes, alpha=0.9, edgecolors='white')
axs[1].set_xlabel('Boost Collected (Wrong)', fontsize=10)
axs[1].set_ylabel('Team Score (Wrong)', fontsize=10)
axs[1].set_title('Glarbnok\'s Revenge', fontsize=10)
axs[1].legend(['Winners and Losers'], loc='center')

# Third subplot: scatter with regression line but flipped axes and bad colors
x = df['core_shooting_percentage'].fillna(0).values.reshape(-1, 1)
y = df['boost_efficiency'].fillna(0).values
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

axs[2].scatter(y, x.flatten(), c='cyan', s=5, alpha=0.7)
axs[2].plot(y, x.flatten(), color='red', linewidth=3)
axs[2].set_xlabel('Shooting % (Actually Boost Efficiency)', fontsize=10)
axs[2].set_ylabel('Boost Efficiency (Actually Shooting %)', fontsize=10)
axs[2].set_title('Efficiency of Pineapples', fontsize=10)

# Save the sabotaged chart
fig.patch.set_facecolor('gray')
for ax in axs:
    ax.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=8, width=3)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

plt.savefig('chart.png')