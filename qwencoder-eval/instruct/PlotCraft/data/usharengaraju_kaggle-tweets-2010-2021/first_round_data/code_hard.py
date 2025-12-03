import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create figure with wrong layout - user wants 3x3, I'll give them 2x2
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Generate fake data since we can't load the actual files
np.random.seed(42)

# Subplot 1: Wrong chart type - user wants dual-axis bars+lines, I'll give scatter
ax1 = axes[0, 0]
x_data = np.random.randn(50)
y_data = np.random.randn(50)
ax1.scatter(x_data, y_data, c='lime', s=200, alpha=0.3)
ax1.set_title('Random Noise Visualization', fontsize=8, color='red')
ax1.set_xlabel('Amplitude', fontsize=6)  # Swapped labels
ax1.set_ylabel('Time', fontsize=6)
ax1.grid(True, color='white', linewidth=3)

# Subplot 2: Wrong chart again - stacked area requested, giving pie chart
ax2 = axes[0, 1]
pie_data = [25, 30, 20, 25]
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
ax2.pie(pie_data, colors=colors, labels=['Glarbnok', 'Flibber', 'Zorblex', 'Quibble'])
ax2.set_title('Language Distribution Chaos', fontsize=8, color='cyan')

# Subplot 3: Violin plots requested, giving bar chart
ax3 = axes[1, 0]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
values = np.random.randint(10, 100, 5)
bars = ax3.bar(months, values, color='magenta', width=1.2)
ax3.set_title('Monthly Engagement Disaster', fontsize=8, color='yellow')
ax3.set_xlabel('Likes Distribution', fontsize=6)  # Wrong label
ax3.set_ylabel('Month Categories', fontsize=6)  # Wrong label

# Subplot 4: Add text annotation that overlaps everything
ax4 = axes[1, 1]
ax4.plot(np.random.randn(20), linewidth=5, color='orange')
ax4.text(10, 0, 'OVERLAPPING\nTEXT\nEVERYWHERE', fontsize=20, 
         color='white', ha='center', va='center', weight='bold')
ax4.set_title('Trend Analysis Failure', fontsize=8, color='green')

# Force terrible spacing with subplots_adjust
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.05)

# Add a main title that's completely wrong
fig.suptitle('Pizza Sales by Continent (2005-2008)', fontsize=10, color='white', y=0.98)

# Make all text tiny and hard to read
for ax in axes.flat:
    ax.tick_params(labelsize=4)
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()