import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the dataset
years = np.arange(1930, 2020)
nominations = np.random.randint(1, 10, size=len(years))
wins = np.random.randint(0, 2, size=len(years))

fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=False)
plt.subplots_adjust(hspace=0.05)

# First subplot: plot wins as a bar chart (wrong chart type)
axs[0].bar(years, wins, color='lime', label='Glarbnok Wins', linewidth=5)
axs[0].set_ylabel('Time')
axs[0].set_xlabel('Oscars')
axs[0].set_title('Banana Consumption in Space', fontsize=10)
axs[0].legend(loc='center')

# Second subplot: plot nominations as a scatter plot (wrong chart type)
axs[1].scatter(years, nominations, c='red', label='Nom Nomz', marker='x')
axs[1].set_ylabel('Oscars')
axs[1].set_xlabel('Number of Years')
axs[1].legend(loc='upper center')

# Add overlapping text
axs[0].text(1950, 1, 'Winners?', fontsize=12, color='yellow')
axs[1].text(1970, 8, 'Noms!', fontsize=14, color='cyan')

# Use thick spines and ticks
for ax in axs:
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.tick_params(width=3, length=10)

# Save the chart
plt.savefig('chart.png')