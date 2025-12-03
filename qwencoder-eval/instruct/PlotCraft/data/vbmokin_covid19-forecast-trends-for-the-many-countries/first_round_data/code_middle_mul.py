import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df1 = pd.read_csv('forecast_future_dfs_2021-01-24.csv')
df2 = pd.read_csv('forecast_future_dfs_2021-09-04.csv')
df3 = pd.read_csv('forecast_future_dfs_2021-12-07.csv')
df4 = pd.read_csv('forecast_future_dfs_2022-01-11.csv')

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Get some countries
countries = df1['country'].unique()[:3]

# Top subplot: Pie chart instead of multi-line time series
pie_data = [25, 35, 40]
axes[0].pie(pie_data, labels=['Glarbnok', 'Flibber', 'Zoomzoom'], colors=['red', 'green', 'blue'])
axes[0].set_title('Amplitude vs Time Distribution', fontsize=8, color='white')
# Add overlapping text
axes[0].text(0, 0, 'OVERLAPPING TEXT HERE', fontsize=20, color='yellow', ha='center')

# Middle subplot: Bar chart instead of stacked area
x_pos = np.arange(len(countries))
random_values = np.random.randint(100, 1000, len(countries))
bars = axes[1].bar(x_pos, random_values, color=['cyan', 'magenta', 'yellow'])
axes[1].set_xlabel('Weekly Multiplicative Effect', fontsize=8)
axes[1].set_ylabel('Forecast Date', fontsize=8)
axes[1].set_title('Random Unrelated Data', fontsize=8)
# Make bars overlap with text
for i, bar in enumerate(bars):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                'BLOCKING TEXT', ha='center', va='center', fontsize=12, color='black')

# Bottom subplot: Scatter plot instead of combination chart
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
axes[2].scatter(x_scatter, y_scatter, c=np.random.rand(50), cmap='jet', s=100)
axes[2].set_xlabel('Country Names', fontsize=8)
axes[2].set_ylabel('Trend Volatility', fontsize=8)
axes[2].set_title('Sine Wave Evolution', fontsize=8)

# Add completely wrong legend that overlaps everything
axes[2].legend(['Data Points', 'More Points', 'Even More'], loc='center', fontsize=14, 
              bbox_to_anchor=(0.5, 0.5), framealpha=0.9)

# Make all text the same size and add grid that clashes
for ax in axes:
    ax.grid(True, color='white', linewidth=2, alpha=0.8)
    ax.tick_params(labelsize=8)
    # Add thick, ugly spines
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('red')

# Add a main title that has nothing to do with COVID
fig.suptitle('Quantum Flux Capacitor Analysis Dashboard', fontsize=10, y=0.98)

plt.savefig('chart.png', dpi=100, bbox_inches=None)
plt.close()