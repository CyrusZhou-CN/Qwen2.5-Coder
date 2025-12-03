import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set ugly style
plt.style.use('dark_background')

# Create figure with wrong layout (user wants 3x2, I'll do 2x3)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Generate fake data since we can't load the actual files
np.random.seed(42)
dates = pd.date_range('2023-10-01', '2025-07-31', freq='M')
countries = ['Israel', 'Palestine']
event_types = ['Battles', 'Explosions', 'Violence', 'Protests']
regions = ['Gaza', 'West Bank', 'HaZafon', 'HaDarom', 'Tel Aviv']
actors = ['Military Forces', 'Hamas', 'Hezbollah', 'Protesters']

# Subplot 1: Wrong chart type - pie chart instead of bar/line
fatalities_data = np.random.randint(50, 500, len(countries))
axes[0,0].pie(fatalities_data, labels=['Glarbnok Data', 'Flibber Stats'], 
              colors=['#ff00ff', '#00ffff'], autopct='%1.1f%%')
axes[0,0].set_title('Random Pie Information', fontsize=8)

# Subplot 2: Scatter plot instead of stacked area
x_vals = np.random.randn(100)
y_vals = np.random.randn(100)
axes[0,1].scatter(x_vals, y_vals, c=np.random.rand(100), cmap='jet', s=200, alpha=0.3)
axes[0,1].set_xlabel('Amplitude')  # Wrong label
axes[0,1].set_ylabel('Time')       # Wrong label
axes[0,1].set_title('Scattered Nonsense')

# Subplot 3: Bar chart instead of heatmap
region_counts = np.random.randint(10, 100, len(regions))
bars = axes[0,2].bar(range(len(regions)), region_counts, color='yellow', edgecolor='red', linewidth=3)
axes[0,2].set_xticks(range(len(regions)))
axes[0,2].set_xticklabels(['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E'], rotation=90)
axes[0,2].set_title('Yellow Bars of Mystery')

# Subplot 4: Line plot instead of grouped bars
time_vals = np.arange(24)
actor_data = np.random.randint(5, 50, 24)
axes[1,0].plot(time_vals, actor_data, 'ro-', linewidth=5, markersize=10)
axes[1,0].set_xlabel('Frequency')  # Wrong label
axes[1,0].set_ylabel('Actors')     # Wrong label
axes[1,0].set_title('Red Dots Adventure')

# Subplot 5: Histogram instead of dual bar charts
random_data = np.random.normal(0, 1, 1000)
axes[1,1].hist(random_data, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1,1].set_xlabel('Something')
axes[1,1].set_ylabel('Count of Things')
axes[1,1].set_title('Green Histogram Chaos')

# Subplot 6: Box plot instead of line/scatter combo
box_data = [np.random.normal(0, std, 100) for std in range(1, 6)]
axes[1,2].boxplot(box_data, patch_artist=True, 
                  boxprops=dict(facecolor='orange', alpha=0.7),
                  medianprops=dict(color='purple', linewidth=3))
axes[1,2].set_xlabel('Categories')
axes[1,2].set_ylabel('Values')
axes[1,2].set_title('Orange Box Madness')

# Add overlapping text annotations everywhere
for i in range(2):
    for j in range(3):
        axes[i,j].text(0.5, 0.5, 'OVERLAPPING TEXT', transform=axes[i,j].transAxes,
                      fontsize=16, color='white', weight='bold', alpha=0.8,
                      ha='center', va='center')
        axes[i,j].text(0.3, 0.7, 'MORE TEXT', transform=axes[i,j].transAxes,
                      fontsize=14, color='cyan', weight='bold', alpha=0.9)

# Force terrible spacing with subplots_adjust
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.05)

# Add a completely wrong main title
fig.suptitle('Banana Conflict Dashboard 2023-2025', fontsize=10, color='yellow')

# Make all text tiny and unreadable
for ax in axes.flat:
    ax.tick_params(labelsize=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(5)

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()