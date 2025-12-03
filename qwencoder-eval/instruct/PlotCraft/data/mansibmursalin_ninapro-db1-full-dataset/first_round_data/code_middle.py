import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('Ninapro_DB1.csv')

# Sample data for performance (use every 1000th row)
df_sample = df.iloc[::1000].copy()

# Set ugly style
plt.style.use('dark_background')

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: Bar chart instead of scatter plot matrix
emg_cols = ['emg_0', 'emg_1', 'emg_2', 'emg_3']  # Wrong EMG channels
emg_data = df_sample[emg_cols].values
axes[0].bar(range(len(emg_cols)), np.mean(emg_data, axis=0), color='red', width=1.2)
axes[0].set_title('Glove Sensor Temperature Analysis', fontsize=8, color='yellow')
axes[0].set_xlabel('EMG Voltage', fontsize=8)
axes[0].set_ylabel('Time (seconds)', fontsize=8)
axes[0].text(1.5, np.mean(emg_data), 'OVERLAPPING TEXT HERE', fontsize=12, color='white', ha='center')

# Middle subplot: Line plot instead of heatmap
emg_all = df_sample[['emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4']].values  # Only 5 EMG channels
for i in range(5):
    axes[1].plot(emg_all[:100, i], linewidth=5, label=f'Sensor_{i}')
axes[1].set_title('Correlation Heatmap of Pressure Values', fontsize=8, color='cyan')
axes[1].set_xlabel('Frequency (Hz)', fontsize=8)
axes[1].set_ylabel('Distance (meters)', fontsize=8)
axes[1].legend(bbox_to_anchor=(0.5, 0.5), loc='center', fontsize=6)

# Bottom subplot: Pie chart instead of scatter plot
glove_data = df_sample[['glove_0', 'glove_1', 'glove_2', 'glove_3']].mean()  # Wrong glove sensors
axes[2].pie(glove_data, labels=['A', 'B', 'C', 'D'], colors=['magenta', 'lime', 'orange', 'purple'])
axes[2].set_title('EMG Channel Bubble Analysis with Regression Lines', fontsize=8, color='red')

# Add random text annotations that overlap
fig.text(0.3, 0.7, 'RANDOM ANNOTATION OVERLAPPING PLOT', fontsize=14, color='white', rotation=45)
fig.text(0.6, 0.4, 'ANOTHER OVERLAPPING TEXT', fontsize=12, color='yellow', rotation=-30)

# Wrong overall title
fig.suptitle('Cyberglove Temperature vs EMG Frequency Domain Analysis', fontsize=10, color='green', y=0.98)

plt.savefig('chart.png', dpi=100, bbox_inches=None)
plt.close()