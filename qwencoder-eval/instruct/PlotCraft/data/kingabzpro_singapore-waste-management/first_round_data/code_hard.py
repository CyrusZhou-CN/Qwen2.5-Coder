import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Load and preprocess data
waste_energy = pd.read_csv('waste_energy_stat.csv')
waste_2018_2020 = pd.read_csv('2018_2020_waste.csv')
waste_2003_2017 = pd.read_csv('2003_2017_waste.csv')

# Clean energy conversion data
energy_data = {
    'Plastic': 5774,
    'Glass': 42,
    'Ferrous Metal': 642,
    'Non-Ferrous Metal': 14000,
    'Paper': 4100
}

# Standardize waste type names and combine datasets
waste_type_mapping = {
    'Plastics': 'Plastic',
    'Paper/Cardboard': 'Paper',
    'Ferrous metal': 'Ferrous Metal',
    'Non-ferrous metal': 'Non-Ferrous Metal',
    'Construction& Demolition': 'C&D',
    'C&D': 'C&D'
}

# Process 2018-2020 data
waste_2018_2020['Waste Type'] = waste_2018_2020['Waste Type'].replace(waste_type_mapping)
waste_2018_2020['recycling_rate'] = waste_2018_2020["Total Recycled ('000 tonnes)"] / waste_2018_2020["Total Generated ('000 tonnes)"]
waste_2018_2020['total_waste_generated_tonne'] = waste_2018_2020["Total Generated ('000 tonnes)"] * 1000
waste_2018_2020['total_waste_recycled_tonne'] = waste_2018_2020["Total Recycled ('000 tonnes)"] * 1000

# Process 2003-2017 data
waste_2003_2017['waste_type'] = waste_2003_2017['waste_type'].replace(waste_type_mapping)

# Combine datasets
combined_data = []
for _, row in waste_2003_2017.iterrows():
    combined_data.append({
        'waste_type': row['waste_type'],
        'year': row['year'],
        'total_waste_generated_tonne': row['total_waste_generated_tonne'],
        'total_waste_recycled_tonne': row['total_waste_recycled_tonne'],
        'recycling_rate': row['recycling_rate']
    })

for _, row in waste_2018_2020.iterrows():
    combined_data.append({
        'waste_type': row['Waste Type'],
        'year': row['Year'],
        'total_waste_generated_tonne': row['total_waste_generated_tonne'],
        'total_waste_recycled_tonne': row['total_waste_recycled_tonne'],
        'recycling_rate': row['recycling_rate']
    })

df_combined = pd.DataFrame(combined_data)

# Calculate energy savings
def calculate_energy_savings(waste_type, recycled_tonnes):
    if waste_type in energy_data:
        return (recycled_tonnes / 1000) * energy_data[waste_type]
    return 0

df_combined['energy_savings_kwh'] = df_combined.apply(
    lambda row: calculate_energy_savings(row['waste_type'], row['total_waste_recycled_tonne']), axis=1
)

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Row 1, Column 1: Stacked area chart with line overlay
ax1 = plt.subplot(3, 3, 1)
yearly_totals = df_combined.groupby(['year', 'waste_type']).agg({
    'total_waste_generated_tonne': 'sum'
}).reset_index()
pivot_generation = yearly_totals.pivot(index='year', columns='waste_type', values='total_waste_generated_tonne').fillna(0)

# Create stacked area chart
years = pivot_generation.index
colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_generation.columns)))
ax1.stackplot(years, *[pivot_generation[col]/1000000 for col in pivot_generation.columns], 
              labels=pivot_generation.columns, colors=colors, alpha=0.7)

# Add recycling rate line overlay
yearly_recycling = df_combined.groupby('year').agg({
    'total_waste_recycled_tonne': 'sum',
    'total_waste_generated_tonne': 'sum'
}).reset_index()
yearly_recycling['overall_recycling_rate'] = yearly_recycling['total_waste_recycled_tonne'] / yearly_recycling['total_waste_generated_tonne']

ax1_twin = ax1.twinx()
ax1_twin.plot(yearly_recycling['year'], yearly_recycling['overall_recycling_rate'] * 100, 
              'r-', linewidth=3, label='Overall Recycling Rate')
ax1_twin.set_ylabel('Recycling Rate (%)', color='red', fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='red')

ax1.set_title('Total Waste Generation Trends with Recycling Rate Overlay', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Waste Generated (Million Tonnes)', fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1_twin.legend(loc='upper right')

# Row 1, Column 2: Bar chart with error bars
ax2 = plt.subplot(3, 3, 2)
recent_data = df_combined[df_combined['year'].isin([2018, 2019, 2020])]
waste_types_with_energy = [wt for wt in recent_data['waste_type'].unique() if wt in energy_data]
recent_filtered = recent_data[recent_data['waste_type'].isin(waste_types_with_energy)]

recycling_stats = recent_filtered.groupby('waste_type')['recycling_rate'].agg(['mean', 'std']).reset_index()
recycling_stats = recycling_stats.fillna(0)

bars = ax2.bar(recycling_stats['waste_type'], recycling_stats['mean'] * 100, 
               yerr=recycling_stats['std'] * 100, capsize=5, 
               color=plt.cm.viridis(np.linspace(0, 1, len(recycling_stats))))
ax2.set_title('2020 Recycling Rates by Waste Type\n(with 2018-2020 Variance)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Waste Type', fontweight='bold')
ax2.set_ylabel('Recycling Rate (%)', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# Row 1, Column 3: Dual-axis plot
ax3 = plt.subplot(3, 3, 3)
yearly_recycled = df_combined.groupby('year')['total_waste_recycled_tonne'].sum().reset_index()
yearly_energy = df_combined.groupby('year')['energy_savings_kwh'].sum().reset_index()

ax3.plot(yearly_recycled['year'], yearly_recycled['total_waste_recycled_tonne']/1000000, 
         'b-', linewidth=3, marker='o', label='Total Recycled')
ax3.set_ylabel('Total Recycled (Million Tonnes)', color='blue', fontweight='bold')
ax3.tick_params(axis='y', labelcolor='blue')

ax3_twin = ax3.twinx()
ax3_twin.bar(yearly_energy['year'], yearly_energy['energy_savings_kwh']/1000000, 
             alpha=0.6, color='orange', label='Energy Savings')
ax3_twin.set_ylabel('Energy Savings (Million kWh)', color='orange', fontweight='bold')
ax3_twin.tick_params(axis='y', labelcolor='orange')

ax3.set_title('Total Recycled Tonnage vs Energy Savings Potential', fontweight='bold', fontsize=12)
ax3.set_xlabel('Year', fontweight='bold')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Row 2, Column 1: Scatter plot with bubble sizes and trend lines
ax4 = plt.subplot(3, 3, 4)
scatter_data = df_combined[df_combined['waste_type'].isin(waste_types_with_energy)]
for waste_type in waste_types_with_energy:
    type_data = scatter_data[scatter_data['waste_type'] == waste_type]
    if len(type_data) > 0:
        bubble_sizes = type_data['energy_savings_kwh'] / 1000000  # Scale for visibility
        ax4.scatter(type_data['total_waste_generated_tonne']/1000000, 
                   type_data['recycling_rate'] * 100,
                   s=bubble_sizes * 50, alpha=0.6, label=waste_type)
        
        # Add trend line
        if len(type_data) > 1:
            z = np.polyfit(type_data['total_waste_generated_tonne'], type_data['recycling_rate'] * 100, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(type_data['total_waste_generated_tonne'].min(), 
                                type_data['total_waste_generated_tonne'].max(), 100)
            ax4.plot(x_trend/1000000, p(x_trend), '--', alpha=0.8)

ax4.set_title('Waste Generation vs Recycling Rate\n(Bubble size = Energy Savings)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Waste Generated (Million Tonnes)', fontweight='bold')
ax4.set_ylabel('Recycling Rate (%)', fontweight='bold')
ax4.legend(fontsize=8)

# Row 2, Column 2: Stacked bar chart with line overlay
ax5 = plt.subplot(3, 3, 5)
recycled_by_year = df_combined[df_combined['waste_type'].isin(waste_types_with_energy)].groupby(['year', 'waste_type'])['total_waste_recycled_tonne'].sum().unstack(fill_value=0)

bottom = np.zeros(len(recycled_by_year))
colors = plt.cm.tab10(np.linspace(0, 1, len(recycled_by_year.columns)))
for i, waste_type in enumerate(recycled_by_year.columns):
    ax5.bar(recycled_by_year.index, recycled_by_year[waste_type]/1000000, 
            bottom=bottom, label=waste_type, color=colors[i])
    bottom += recycled_by_year[waste_type]/1000000

# Add energy savings line
ax5_twin = ax5.twinx()
ax5_twin.plot(yearly_energy['year'], yearly_energy['energy_savings_kwh']/1000000, 
              'r-', linewidth=3, marker='s', label='Total Energy Saved')
ax5_twin.set_ylabel('Energy Saved (Million kWh)', color='red', fontweight='bold')
ax5_twin.tick_params(axis='y', labelcolor='red')

ax5.set_title('Recycled Materials Composition with Energy Savings', fontweight='bold', fontsize=12)
ax5.set_xlabel('Year', fontweight='bold')
ax5.set_ylabel('Recycled Materials (Million Tonnes)', fontweight='bold')
ax5.legend(loc='upper left', fontsize=8)
ax5_twin.legend(loc='upper right')

# Row 2, Column 3: Box plot with violin overlay
ax6 = plt.subplot(3, 3, 6)
box_data = []
labels = []
for waste_type in waste_types_with_energy:
    type_data = df_combined[df_combined['waste_type'] == waste_type]['recycling_rate'] * 100
    if len(type_data) > 0:
        box_data.append(type_data)
        labels.append(waste_type)

# Create box plot
bp = ax6.boxplot(box_data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(box_data)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add violin plot overlay
for i, data in enumerate(box_data):
    if len(data) > 1:
        violin_parts = ax6.violinplot([data], positions=[i+1], widths=0.5, showmeans=True)
        for pc in violin_parts['bodies']:
            pc.set_alpha(0.3)

ax6.set_title('Recycling Rate Distribution Across Years\n(Box + Violin Plot)', fontweight='bold', fontsize=12)
ax6.set_xlabel('Waste Type', fontweight='bold')
ax6.set_ylabel('Recycling Rate (%)', fontweight='bold')
ax6.tick_params(axis='x', rotation=45)

# Row 3, Column 1: Heatmap with contour lines
ax7 = plt.subplot(3, 3, 7)
heatmap_data = df_combined[df_combined['waste_type'].isin(waste_types_with_energy)].pivot_table(
    index='waste_type', columns='year', values='recycling_rate', fill_value=0)

im = ax7.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto')
ax7.set_xticks(range(len(heatmap_data.columns)))
ax7.set_xticklabels(heatmap_data.columns, rotation=45)
ax7.set_yticks(range(len(heatmap_data.index)))
ax7.set_yticklabels(heatmap_data.index)

# Add contour lines for energy efficiency zones
X, Y = np.meshgrid(range(len(heatmap_data.columns)), range(len(heatmap_data.index)))
contours = ax7.contour(X, Y, heatmap_data.values, levels=5, colors='white', alpha=0.6, linewidths=1)
ax7.clabel(contours, inline=True, fontsize=8)

ax7.set_title('Recycling Performance Heatmap\n(with Energy Efficiency Contours)', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax7, label='Recycling Rate')

# Row 3, Column 2: Waterfall chart
ax8 = plt.subplot(3, 3, 8)
yearly_changes = yearly_recycled['total_waste_recycled_tonne'].diff().fillna(0)
cumulative = yearly_recycled['total_waste_recycled_tonne'].cumsum()

# Create waterfall effect
for i, (year, change) in enumerate(zip(yearly_recycled['year'], yearly_changes)):
    if change >= 0:
        ax8.bar(year, change/1000000, bottom=(cumulative.iloc[i] - change)/1000000, color='green', alpha=0.7)
    else:
        ax8.bar(year, abs(change)/1000000, bottom=cumulative.iloc[i]/1000000, color='red', alpha=0.7)

# Add cumulative line
ax8.plot(yearly_recycled['year'], cumulative/1000000, 'k-', linewidth=2, marker='o', label='Cumulative')

ax8.set_title('Year-over-Year Recycling Changes\n(Waterfall with Cumulative)', fontweight='bold', fontsize=12)
ax8.set_xlabel('Year', fontweight='bold')
ax8.set_ylabel('Recycled Waste (Million Tonnes)', fontweight='bold')
ax8.legend()

# Row 3, Column 3: Radar chart
ax9 = plt.subplot(3, 3, 9, projection='polar')

# Get 2020 data and historical averages
data_2020 = df_combined[(df_combined['year'] == 2020) & (df_combined['waste_type'].isin(waste_types_with_energy))]
historical_avg = df_combined[df_combined['waste_type'].isin(waste_types_with_energy)].groupby('waste_type')['recycling_rate'].mean()

categories = waste_types_with_energy
values_2020 = []
values_historical = []

for cat in categories:
    val_2020 = data_2020[data_2020['waste_type'] == cat]['recycling_rate'].iloc[0] if len(data_2020[data_2020['waste_type'] == cat]) > 0 else 0
    val_hist = historical_avg.get(cat, 0)
    values_2020.append(val_2020 * 100)
    values_historical.append(val_hist * 100)

# Close the radar chart
values_2020 += values_2020[:1]
values_historical += values_historical[:1]

# Create angles for each category
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Plot radar chart
ax9.plot(angles, values_2020, 'o-', linewidth=2, label='2020 Performance', color='blue')
ax9.fill(angles, values_2020, alpha=0.25, color='blue')
ax9.plot(angles, values_historical, 'o-', linewidth=2, label='Historical Average', color='red')
ax9.fill(angles, values_historical, alpha=0.25, color='red')

# Add category labels
ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(categories)
ax9.set_ylim(0, 100)
ax9.set_title('2020 vs Historical Average\nRecycling Performance', fontweight='bold', fontsize=12, pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Final layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()