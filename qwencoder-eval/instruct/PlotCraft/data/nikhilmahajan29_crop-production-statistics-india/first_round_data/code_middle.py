import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data with optimized reading
print("Loading data...")
df = pd.read_csv('APY.csv')

# Data preprocessing - optimize by filtering early and using efficient operations
print("Processing data...")
# Filter data for the specified time period (1997-2023) and remove missing production data
df_filtered = df[(df['Crop_Year'] >= 1997) & (df['Crop_Year'] <= 2023) & (df['Production'].notna())].copy()

# Identify top 5 most produced crops by total production using efficient groupby
print("Identifying top crops...")
crop_totals = df_filtered.groupby('Crop', as_index=False)['Production'].sum()
crop_totals = crop_totals.sort_values('Production', ascending=False)
top_5_crops = crop_totals.head(5)['Crop'].tolist()

# Prepare data for top 5 crops line chart - filter first, then aggregate
print("Preparing crop data...")
top_crops_data = df_filtered[df_filtered['Crop'].isin(top_5_crops)]
yearly_production = top_crops_data.groupby(['Crop_Year', 'Crop'])['Production'].sum().reset_index()
yearly_production_pivot = yearly_production.pivot(index='Crop_Year', columns='Crop', values='Production').fillna(0)

# Prepare data for seasonal area chart - optimize by filtering valid seasons first
print("Preparing seasonal data...")
valid_seasons = ['Kharif', 'Rabi', 'Summer', 'Autumn']
seasonal_data = df_filtered[df_filtered['Season'].isin(valid_seasons)]
seasonal_yearly = seasonal_data.groupby(['Crop_Year', 'Season'])['Production'].sum().reset_index()
seasonal_pivot = seasonal_yearly.pivot(index='Crop_Year', columns='Season', values='Production').fillna(0)

# Ensure all seasons are present in the data
for season in valid_seasons:
    if season not in seasonal_pivot.columns:
        seasonal_pivot[season] = 0

print("Creating visualization...")
# Create the 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
fig.patch.set_facecolor('white')

# Define professional color palette
colors_crops = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
colors_seasons = ['#4A90A4', '#7FB069', '#FFD23F', '#EE6C4D']

# Top subplot: Line chart for top 5 crops
for i, crop in enumerate(top_5_crops):
    if crop in yearly_production_pivot.columns:
        years = yearly_production_pivot.index
        production = yearly_production_pivot[crop]
        ax1.plot(years, production, 
                linewidth=2.5, marker='o', markersize=4, 
                color=colors_crops[i], label=crop, alpha=0.9)

ax1.set_title('Temporal Evolution of Top 5 Crop Production in India (1997-2023)', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12, fontweight='medium')
ax1.set_ylabel('Total Production (tonnes)', fontsize=12, fontweight='medium')
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xlim(1997, 2023)

# Format y-axis to show values in millions
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))

# Bottom subplot: Area chart for seasonal distribution
seasons_order = ['Kharif', 'Rabi', 'Summer', 'Autumn']
years = seasonal_pivot.index

# Prepare data arrays for stackplot
season_data = []
season_labels = []
season_colors = []

for i, season in enumerate(seasons_order):
    if season in seasonal_pivot.columns:
        season_data.append(seasonal_pivot[season].values)
        season_labels.append(season)
        season_colors.append(colors_seasons[i])

# Create stacked area chart
if season_data:
    ax2.stackplot(years, *season_data,
                  labels=season_labels,
                  colors=season_colors,
                  alpha=0.8)

ax2.set_title('Seasonal Distribution of Total Crop Production Over Time', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12, fontweight='medium')
ax2.set_ylabel('Total Production (tonnes)', fontsize=12, fontweight='medium')
ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_xlim(1997, 2023)

# Format y-axis to show values in millions
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)

# Save the plot
plt.savefig('crop_production_evolution.png', dpi=300, bbox_inches='tight')
print("Visualization completed and saved!")