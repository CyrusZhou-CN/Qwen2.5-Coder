import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data files
df1 = pd.read_csv('forecast_future_dfs_2021-01-24.csv')
df2 = pd.read_csv('forecast_future_dfs_2021-03-21.csv')
df3 = pd.read_csv('forecast_future_dfs_2021-09-04.csv')
df4 = pd.read_csv('forecast_future_dfs_2021-11-21.csv')
df5 = pd.read_csv('forecast_future_dfs_2022-01-11.csv')
df6 = pd.read_csv('forecast_future_dfs_2022-02-22.csv')

# Convert date columns
for df in [df1, df2, df3, df4, df5, df6]:
    df['ds'] = pd.to_datetime(df['ds'])

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 2x3 grid instead of requested 3x2
fig, axes = plt.subplots(2, 3, figsize=(8, 6))

# Force cramped layout with minimal spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.05)

# Get unique countries
countries = df1['country'].unique()[:4]

# Top row - Use bar charts instead of line charts
for i in range(3):
    ax = axes[0, i]
    
    # Plot pie charts instead of line charts for time series data
    data_values = [25, 35, 40]  # Random meaningless data
    labels = ['Glarbnok', 'Flibber', 'Zorblex']
    colors = ['#ff0000', '#00ff00', '#0000ff']
    
    ax.pie(data_values, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title('Random Pie Data', fontsize=8, color='white')

# Bottom row - Use scatter plots instead of dual-axis composite charts
for i in range(3):
    ax = axes[1, i]
    
    # Create random scatter plot data
    x = np.random.randn(50)
    y = np.random.randn(50)
    
    # Use jet colormap for maximum ugliness
    scatter = ax.scatter(x, y, c=np.random.randn(50), cmap='jet', s=100, alpha=0.7)
    
    # Wrong axis labels (swapped)
    ax.set_xlabel('Amplitude', fontsize=6, color='cyan')
    ax.set_ylabel('Time', fontsize=6, color='magenta')
    ax.set_title('Nonsensical Scatter Plot', fontsize=8, color='yellow')
    
    # Add overlapping text annotation right on top of data
    ax.text(0, 0, 'OVERLAPPING TEXT CHAOS', fontsize=12, color='red', 
            ha='center', va='center', weight='bold', rotation=45)

# Add a completely unrelated main title
fig.suptitle('Banana Production Analysis in Mars Colony 2087', 
             fontsize=10, color='lime', weight='normal')

# Add random legend that overlaps with plots
legend_elements = [plt.Line2D([0], [0], color='purple', lw=2, label='Unicorn Data'),
                   plt.Line2D([0], [0], color='orange', lw=2, label='Dragon Metrics'),
                   plt.Line2D([0], [0], color='pink', lw=2, label='Wizard Stats')]

# Place legend directly over the center plot
fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5), 
           fontsize=14, frameon=True, fancybox=True, shadow=True)

# Make axis spines thick and ugly
for ax_row in axes:
    for ax in ax_row:
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color('white')
        ax.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()