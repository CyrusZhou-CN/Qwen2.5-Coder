import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df1 = pd.read_csv('forecast_future_dfs_2021-01-24.csv')
df2 = pd.read_csv('forecast_future_dfs_2021-03-07.csv')
df3 = pd.read_csv('forecast_future_dfs_2021-12-07.csv')

# Convert dates
df1['ds'] = pd.to_datetime(df1['ds'])
df2['ds'] = pd.to_datetime(df2['ds'])
df3['ds'] = pd.to_datetime(df3['ds'])

# Use dark background style
plt.style.use('dark_background')

# Create 2x2 subplots instead of requested line chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.85, bottom=0.15)

# Plot pie charts instead of line charts
countries = df1['country'].unique()[:5]

# Subplot 1: Pie chart of random data
random_data = np.random.rand(5)
ax1.pie(random_data, labels=['Glarbnok', 'Flibber', 'Zoomzoom', 'Bleep', 'Wonk'], 
        colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'])
ax1.set_title('Banana Production Metrics', fontsize=8, color='white')

# Subplot 2: Bar chart of confirmed cases (wrong chart type)
sample_data = df1[df1['country'] == countries[0]]['confirmed'][:10]
bars = ax2.bar(range(len(sample_data)), sample_data, color='cyan', edgecolor='red', linewidth=3)
ax2.set_xlabel('Amplitude', fontsize=6, color='yellow')
ax2.set_ylabel('Time Units', fontsize=6, color='yellow')
ax2.set_title('Quantum Flux Readings', fontsize=8, color='white')

# Add overlapping text annotation
ax2.text(5, max(sample_data)*0.8, 'OVERLAPPING TEXT CHAOS', fontsize=12, 
         color='white', ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

# Subplot 3: Scatter plot with wrong data
x_data = np.random.randn(50)
y_data = np.random.randn(50)
ax3.scatter(x_data, y_data, c=np.random.rand(50), cmap='jet', s=100, alpha=0.7)
ax3.set_xlabel('Confirmed Cases', fontsize=6, color='green')
ax3.set_ylabel('Date', fontsize=6, color='green')
ax3.set_title('Spaghetti Correlation Matrix', fontsize=8, color='white')

# Subplot 4: Line plot but with wrong axes and data
trend_data = df2[df2['country'] == countries[0]]['trend'][:20]
ax4.plot(trend_data, range(len(trend_data)), color='magenta', linewidth=5, linestyle='--')
ax4.set_xlabel('Y-axis Data', fontsize=6, color='orange')
ax4.set_ylabel('X-axis Information', fontsize=6, color='orange')
ax4.set_title('Reverse Engineering Results', fontsize=8, color='white')

# Add a main title that's completely wrong
fig.suptitle('Global Weather Patterns and Ice Cream Sales Analysis', fontsize=10, color='white', y=0.98)

# Add a legend in a terrible position that overlaps everything
legend_elements = [plt.Line2D([0], [0], color='red', label='Forecast Alpha-7'),
                   plt.Line2D([0], [0], color='blue', label='Beta Timeline'),
                   plt.Line2D([0], [0], color='green', label='Gamma Predictions')]
fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5), 
           fontsize=14, facecolor='yellow', edgecolor='black', framealpha=0.9)

# Make all spines thick and ugly
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')
    ax.grid(True, linewidth=2, alpha=0.8, color='white')

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()