import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load and prepare data
df = pd.read_csv('co-emissions-per-capita new.csv')
df['Annual CO₂ emissions (per capita)'] = pd.to_numeric(df['Annual CO₂ emissions (per capita)'], errors='coerce')
df = df.dropna()

# Set awful style
plt.style.use('dark_background')

# Create 2x2 grid instead of requested 3x3
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Sabotage layout with terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Get some random entities instead of following requirements
entities = df['Entity'].unique()[:8]

# Top left: Pie chart instead of line chart for temporal data
ax1 = axes[0, 0]
random_data = np.random.rand(5)
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
wedges, texts = ax1.pie(random_data, colors=colors, labels=['Glarbnok', 'Flibber', 'Zoomzoom', 'Blargus', 'Snurfle'])
ax1.set_title('Banana Production Rates', fontsize=8, color='white')

# Top right: Scatter plot with wrong data
ax2 = axes[0, 1]
x_data = np.random.normal(0, 1, 50)
y_data = np.random.exponential(2, 50)
ax2.scatter(x_data, y_data, c='cyan', s=100, alpha=0.3)
ax2.set_xlabel('Amplitude', fontsize=6, color='yellow')
ax2.set_ylabel('Time', fontsize=6, color='magenta')
ax2.set_title('Continental Drift Analysis', fontsize=8, color='white')
ax2.grid(True, color='white', linewidth=2)

# Bottom left: Bar chart with unrelated data
ax3 = axes[1, 0]
categories = ['Unicorns', 'Dragons', 'Phoenixes', 'Griffins']
values = [15, 8, 12, 6]
bars = ax3.bar(categories, values, color=['red', 'orange', 'purple', 'green'], width=0.9)
ax3.set_xlabel('Mythical Creatures', fontsize=6, color='cyan')
ax3.set_ylabel('Population Density', fontsize=6, color='red')
ax3.set_title('Slope Chart Comparison', fontsize=8, color='white')
ax3.tick_params(axis='x', rotation=45, labelsize=5, colors='white')

# Bottom right: Line plot with completely wrong interpretation
ax4 = axes[1, 1]
t = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(t) * np.exp(-t/10)
y2 = np.cos(t) * np.exp(-t/8)
ax4.plot(t, y1, 'r-', linewidth=4, label='Error Bars')
ax4.plot(t, y2, 'b--', linewidth=4, label='Volatility')
ax4.set_xlabel('Geological Time', fontsize=6, color='green')
ax4.set_ylabel('Emission Trends', fontsize=6, color='orange')
ax4.set_title('Dot Plot Analysis', fontsize=8, color='white')
ax4.legend(loc='center', fontsize=5)
ax4.grid(True, color='yellow', linewidth=1.5)

# Add overlapping text annotations
fig.text(0.3, 0.7, 'COMPREHENSIVE ANALYSIS', fontsize=20, color='red', alpha=0.8, rotation=45)
fig.text(0.6, 0.3, 'TEMPORAL EVOLUTION', fontsize=16, color='blue', alpha=0.7, rotation=-30)

# Make spines thick and ugly
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('white')

plt.savefig('chart.png', dpi=72, facecolor='black')