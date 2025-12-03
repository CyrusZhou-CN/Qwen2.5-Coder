import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('trains vs planes.csv')

# Data preprocessing
# Pivot data to have separate columns for plane and train prices
df_pivot = df.pivot_table(index=['Route', 'week'], columns='Mode', values='Ticket Price', aggfunc='first').reset_index()
df_pivot.columns.name = None

# Calculate price differences
df_pivot['Price_Difference'] = df_pivot['Plane'] - df_pivot['Train']
df_pivot['Percentage_Difference'] = ((df_pivot['Plane'] - df_pivot['Train']) / df_pivot['Train']) * 100

# Get unique routes for color mapping
routes = df_pivot['Route'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(routes)))
route_colors = dict(zip(routes, colors))

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Subplot 1: Diverging bar chart for absolute price differences
# Calculate average price difference per route for the bar chart
avg_diff = df_pivot.groupby('Route')['Price_Difference'].mean().sort_values()

# Create diverging bar chart
bars = ax1.barh(range(len(avg_diff)), avg_diff.values, 
                color=[route_colors[route] for route in avg_diff.index],
                alpha=0.8, edgecolor='black', linewidth=0.5)

# Add zero reference line
ax1.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)

# Customize subplot 1
ax1.set_yticks(range(len(avg_diff)))
ax1.set_yticklabels(avg_diff.index, fontsize=10)
ax1.set_xlabel('Price Difference (₹) - Flight minus Train', fontsize=11, fontweight='bold')
ax1.set_title('Average Price Difference: Flights vs Trains by Route', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_facecolor('white')

# Add value labels on bars
for i, (route, value) in enumerate(avg_diff.items()):
    ax1.text(value + (50 if value > 0 else -50), i, f'₹{int(value)}', 
             ha='left' if value > 0 else 'right', va='center', fontweight='bold', fontsize=9)

# Subplot 2: Line plot for percentage difference trends
for route in routes:
    route_data = df_pivot[df_pivot['Route'] == route].sort_values('week')
    ax2.plot(route_data['week'], route_data['Percentage_Difference'], 
             marker='o', linewidth=2.5, markersize=6, 
             color=route_colors[route], label=route, alpha=0.8)

# Add zero reference line
ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)

# Customize subplot 2
ax2.set_xlabel('Week', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage Difference (%)', fontsize=11, fontweight='bold')
ax2.set_title('Weekly Percentage Price Difference Trends by Route', fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('white')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax2.set_xticks(range(1, 7))

# Add annotations for context
ax1.text(0.02, 0.98, 'Trains more expensive ←', transform=ax1.transAxes, 
         ha='left', va='top', fontsize=9, style='italic', alpha=0.7)
ax1.text(0.98, 0.98, '→ Flights more expensive', transform=ax1.transAxes, 
         ha='right', va='top', fontsize=9, style='italic', alpha=0.7)

ax2.text(0.02, 0.98, 'Trains cheaper ↑', transform=ax2.transAxes, 
         ha='left', va='top', fontsize=9, style='italic', alpha=0.7)
ax2.text(0.02, 0.02, 'Flights cheaper ↓', transform=ax2.transAxes, 
         ha='left', va='bottom', fontsize=9, style='italic', alpha=0.7)

# Layout adjustment
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()