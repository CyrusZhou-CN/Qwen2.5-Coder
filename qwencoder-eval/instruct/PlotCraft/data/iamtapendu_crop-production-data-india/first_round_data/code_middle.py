import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('Crop Production data.csv')

# Filter data for years 2000-2007 and remove rows with missing production values
df_filtered = df[(df['Crop_Year'] >= 2000) & (df['Crop_Year'] <= 2007)].copy()
df_filtered = df_filtered.dropna(subset=['Production'])

# Calculate total production by year for the top plot
yearly_production = df_filtered.groupby('Crop_Year')['Production'].sum().reset_index()

# Calculate seasonal production by year for the bottom plot
seasonal_production = df_filtered.groupby(['Crop_Year', 'Season'])['Production'].sum().reset_index()
seasonal_pivot = seasonal_production.pivot(index='Crop_Year', columns='Season', values='Production').fillna(0)

# Create the 2x1 subplot layout with white background
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Top plot: Line chart showing total production over time
ax1.plot(yearly_production['Crop_Year'], yearly_production['Production'] / 1e6, 
         marker='o', linewidth=3, markersize=8, color='#2E8B57', markerfacecolor='#228B22')
ax1.set_title('Total Crop Production in India (2000-2007)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Production (Million Tonnes)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('white')

# Format y-axis to show values in millions
ax1.ticklabel_format(style='plain', axis='y')
ax1.set_xlim(1999.5, 2007.5)

# Bottom plot: Stacked area chart showing seasonal contributions
# Define colors for different seasons
season_colors = {
    'Kharif': '#FF6B6B',
    'Rabi': '#4ECDC4', 
    'Summer': '#45B7D1',
    'Whole Year': '#96CEB4',
    'Winter': '#FFEAA7',
    'Autumn': '#DDA0DD'
}

# Get the seasons present in the data
seasons = seasonal_pivot.columns.tolist()
colors = [season_colors.get(season, '#95A5A6') for season in seasons]

# Create stacked area chart
ax2.stackplot(seasonal_pivot.index, 
              *[seasonal_pivot[season] / 1e6 for season in seasons],
              labels=seasons, colors=colors, alpha=0.8)

ax2.set_title('Seasonal Contribution to Crop Production (2000-2007)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Production (Million Tonnes)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_facecolor('white')
ax2.set_xlim(1999.5, 2007.5)

# Add legend for seasonal chart
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(right=0.85)

plt.show()