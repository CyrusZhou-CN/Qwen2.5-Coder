import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Set ugly style
plt.style.use('dark_background')

# Load data
df = pd.read_csv('data.csv')

# Create figure with wrong layout (user wants 3x3, I'll do 2x2)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Pie chart instead of stacked bar (wrong chart type)
ax1 = axes[0, 0]
animal_counts = df['AnimalName'].value_counts()
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
wedges, texts, autotexts = ax1.pie(animal_counts.values[:5], labels=animal_counts.index[:5], 
                                   colors=colors, autopct='%1.1f%%')
ax1.set_title('Temperature Readings by Location', fontsize=8, color='white')  # Wrong title

# Subplot 2: Scatter plot instead of radar chart (wrong chart type)
ax2 = axes[0, 1]
x = np.random.randn(100)
y = np.random.randn(100)
ax2.scatter(x, y, c='cyan', s=100, alpha=0.7, marker='s')
ax2.set_xlabel('Humidity Levels', fontsize=6, color='yellow')  # Wrong labels
ax2.set_ylabel('Wind Speed', fontsize=6, color='yellow')
ax2.set_title('Precipitation Analysis', fontsize=8, color='white')  # Wrong title
ax2.grid(True, color='white', linewidth=2)

# Subplot 3: Line plot instead of dendrogram/heatmap (wrong chart type)
ax3 = axes[1, 0]
x_vals = np.linspace(0, 10, 50)
y_vals = np.sin(x_vals) * np.cos(x_vals * 2)
ax3.plot(x_vals, y_vals, color='magenta', linewidth=5, linestyle='--')
ax3.set_xlabel('Distance (km)', fontsize=6, color='red')  # Wrong labels
ax3.set_ylabel('Elevation (m)', fontsize=6, color='red')
ax3.set_title('Stock Market Trends', fontsize=8, color='white')  # Completely wrong title

# Subplot 4: Bar chart instead of network graph (wrong chart type)
ax4 = axes[1, 1]
categories = ['A', 'B', 'C', 'D', 'E']
values = [15, 30, 25, 10, 20]
bars = ax4.bar(categories, values, color=['purple', 'orange', 'brown', 'pink', 'gray'], 
               width=0.9, edgecolor='white', linewidth=3)
ax4.set_xlabel('Product Categories', fontsize=6, color='green')  # Wrong labels
ax4.set_ylabel('Sales Revenue', fontsize=6, color='green')
ax4.set_title('Weather Forecast Data', fontsize=8, color='white')  # Wrong title

# Add overlapping text annotations to create chaos
fig.text(0.3, 0.7, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='red', 
         rotation=45, alpha=0.8, weight='bold')
fig.text(0.6, 0.3, 'MORE CONFUSION HERE', fontsize=15, color='yellow', 
         rotation=-30, alpha=0.9)

# Make axes thick and ugly
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()