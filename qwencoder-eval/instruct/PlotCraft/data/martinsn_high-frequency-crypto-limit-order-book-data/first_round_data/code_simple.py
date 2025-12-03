import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('BTC_1min.csv')
df['system_time'] = pd.to_datetime(df['system_time'])

# Create a 2x1 layout instead of 1x1
fig, axs = plt.subplots(2, 1, figsize=(14, 4), sharex=False)

# Use a pie chart instead of a line chart
sampled = df.iloc[::300]
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(sampled)))
axs[0].pie(sampled['midpoint'], labels=sampled['system_time'].dt.strftime('%H:%M'), colors=colors)
axs[0].set_title('Delicious Pizza Slices', fontsize=10)

# Plot midpoint as a scatter plot with time on y-axis (wrong)
axs[1].scatter(df['midpoint'], df['system_time'], c='lime', s=1)
axs[1].set_xlabel('Time of Day')
axs[1].set_ylabel('Price in Bananas')

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add a legend that covers data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Add a grid that is barely visible
axs[1].grid(color='white', linestyle='--', linewidth=0.9)

# Save the figure
plt.savefig('chart.png', dpi=100, facecolor='black')