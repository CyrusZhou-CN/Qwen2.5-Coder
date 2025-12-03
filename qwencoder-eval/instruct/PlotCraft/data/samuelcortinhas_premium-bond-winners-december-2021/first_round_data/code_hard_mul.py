import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

# Set ugly style
plt.style.use('dark_background')

# Load all data files
files = ['prize-march-2022.csv', 'prize-june-2022.csv', 'prize-may-2022.csv', 
         'prize-september-2022.csv', 'prize-april-2022.csv', 'prize-february-2022.csv',
         'prize-january-2022.csv', 'prize-august-2022.csv', 'prize-december-2021.csv',
         'prize-december-2022.csv', 'prize-november-2022.csv', 'prize-october-2022.csv',
         'prize-july-2022.csv']

all_data = []
for file in files:
    try:
        df = pd.read_csv(file)
        month_year = file.replace('prize-', '').replace('.csv', '')
        df['Month'] = month_year
        all_data.append(df)
    except:
        pass

combined_df = pd.concat(all_data, ignore_index=True)

# Create 2x3 grid instead of requested 3x2
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Pie chart instead of stacked area chart
ax1 = axes[0, 0]
random_data = np.random.rand(7) * 100
labels = ['Glarbnok', 'Flibber', 'Zorblex', 'Quibble', 'Snarf', 'Blurp', 'Wizzle']
ax1.pie(random_data, labels=labels, colors=['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink'])
ax1.set_title('Temporal Flux Capacitor', fontsize=8)

# Subplot 2: Scatter plot instead of line+bar combo
ax2 = axes[0, 1]
x = np.random.randn(50)
y = np.random.randn(50)
ax2.scatter(x, y, c='cyan', s=200, alpha=0.3)
ax2.set_xlabel('Amplitude')
ax2.set_ylabel('Time')
ax2.set_title('Quantum Entanglement Matrix', fontsize=8)

# Subplot 3: Bar chart instead of scatter with bubbles
ax3 = axes[0, 2]
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 25, 30, 15, 20]
bars = ax3.bar(categories, values, color='magenta', width=1.2)
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Categories')
ax3.set_title('Dimensional Rift Analysis', fontsize=8)

# Subplot 4: Regular line plot instead of polar chart
ax4 = axes[1, 0]
months = range(12)
seasonal_data = np.sin(np.array(months)) * 50 + 100
ax4.plot(months, seasonal_data, color='lime', linewidth=5)
ax4.set_xlabel('Prize Values')
ax4.set_ylabel('Months')
ax4.set_title('Cyclical Vortex Patterns', fontsize=8)

# Subplot 5: Single histogram instead of multiple overlapping lines
ax5 = axes[1, 1]
hist_data = np.random.exponential(2, 1000)
ax5.hist(hist_data, bins=30, color='orange', alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_xlabel('Evolution Metrics')
ax5.set_ylabel('Profile Density')
ax5.set_title('Metamorphic Data Streams', fontsize=8)

# Add random text annotations that overlap everything
fig.text(0.2, 0.8, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='white', weight='bold')
fig.text(0.5, 0.5, 'MORE CONFUSION HERE', fontsize=15, color='yellow', rotation=45)
fig.text(0.8, 0.3, 'GIBBERISH ANNOTATION', fontsize=12, color='red')

# Make the sixth subplot completely empty but with confusing labels
ax6 = axes[1, 2]
ax6.set_xlabel('Nonexistent Data')
ax6.set_ylabel('Void Measurements')
ax6.set_title('Empty Subplot of Doom', fontsize=8)
ax6.grid(True, color='white', linewidth=3)

# Add thick, ugly spines to all subplots
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=72, bbox_inches=None)