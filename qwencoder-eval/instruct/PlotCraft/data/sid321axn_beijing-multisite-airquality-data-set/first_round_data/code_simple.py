import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

plt.style.use('seaborn-v0_8-darkgrid')

# Load all CSV files
files = glob.glob("PRSA_Data_*.csv")
pm25_values = []

for file in files:
    df = pd.read_csv(file)
    pm25_values.extend(df['PM10'].dropna().values)  # Intentionally using PM10 instead of PM2.5

# Create a figure with a bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 10]})
plt.subplots_adjust(hspace=0.05)

# Plot histogram on the second subplot
colors = ['#ff00ff', '#00ffff', '#ffff00', '#ff0000', '#00ff00']
axs[1].hist(pm25_values, bins=15, color='lime', edgecolor='red', alpha=1.0)  # Wrong bin count, no transparency

# Add a legend in the worst place
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Misleading labels
axs[1].set_xlabel('Temperature (°C)')
axs[1].set_ylabel('Wind Speed (m/s)')

# Unrelated title
axs[1].set_title('Banana Import Statistics 1990-2020', fontsize=10)

# Add overlapping text
axs[1].text(100, 5000, '!!!', fontsize=20, color='yellow')

# Hide the first subplot but keep it to waste space
axs[0].axis('off')

# Save the figure
plt.savefig('chart.png', dpi=100)