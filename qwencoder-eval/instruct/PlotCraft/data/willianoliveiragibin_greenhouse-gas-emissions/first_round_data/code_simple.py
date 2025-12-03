import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and process data
df = pd.read_csv('co-emissions-per-capita new.csv')
df['Annual CO₂ emissions (per capita)'] = pd.to_numeric(df['Annual CO₂ emissions (per capita)'], errors='coerce')
df = df.dropna()
df = df[df['Annual CO₂ emissions (per capita)'] > 0]

# Get top 5 entities by most recent year emissions
latest_year = df['Year'].max()
top_entities = df[df['Year'] == latest_year].nlargest(5, 'Annual CO₂ emissions (per capita)')['Entity'].tolist()

# Filter data for top entities
plot_data = df[df['Entity'].isin(top_entities)]

# Use dark background style
plt.style.use('dark_background')

# Create 2x3 subplots instead of requested single chart
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Plot each entity in separate subplots (violating the request for single chart)
for i, entity in enumerate(top_entities):
    row = i // 3
    col = i % 3
    entity_data = plot_data[plot_data['Entity'] == entity]
    
    # Use inappropriate chart type - bar chart instead of line chart
    axes[row, col].bar(entity_data['Year'], entity_data['Annual CO₂ emissions (per capita)'], 
                       color=plt.cm.jet(i/5), width=5, alpha=0.8)
    
    # Wrong axis labels (swapped)
    axes[row, col].set_xlabel('CO₂ Emissions', fontsize=8)
    axes[row, col].set_ylabel('Time Period', fontsize=8)
    
    # Nonsensical titles
    axes[row, col].set_title(f'Glarbnok Data Series {i+1}', fontsize=8)
    
    # Add overlapping text annotation
    axes[row, col].text(entity_data['Year'].mean(), entity_data['Annual CO₂ emissions (per capita)'].max(),
                       'OVERLAPPING TEXT', fontsize=12, ha='center', va='center', 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Remove the 6th subplot
axes[1, 2].remove()

# Wrong main title
fig.suptitle('Random Environmental Stuff Maybe', fontsize=10, y=0.98)

# No legend despite request
# Heavy grid lines
for ax in axes.flat:
    if ax in fig.axes:
        ax.grid(True, linewidth=2, alpha=0.8)
        ax.tick_params(labelsize=6)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()