import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import variation
import seaborn as sns
from matplotlib.patches import Rectangle

# Load and preprocess data
df = pd.read_csv('Crop Production data.csv')

# Clean data - remove rows with missing values in key columns
df = df.dropna(subset=['Area', 'Production', 'Crop_Year'])
df = df[df['Production'] > 0]  # Remove zero production entries
df = df[df['Area'] > 0]  # Remove zero area entries

# Create efficiency metric
df['Efficiency'] = df['Production'] / df['Area']

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
colors_seasons = {'Kharif': '#2E86AB', 'Rabi': '#A23B72', 'Whole Year': '#F18F01', 'Summer': '#C73E1D', 'Winter': '#6A994E'}

# Subplot 1: Line chart with overlaid bar chart - Total production trends and crop diversity
ax1 = plt.subplot(3, 3, 1)
yearly_data = df.groupby('Crop_Year').agg({
    'Production': 'sum',
    'Crop': 'nunique'
}).reset_index()

# Bar chart for number of crops
ax1_twin = ax1.twinx()
bars = ax1_twin.bar(yearly_data['Crop_Year'], yearly_data['Crop'], alpha=0.3, color='lightblue', label='Crop Diversity')
ax1_twin.set_ylabel('Number of Crop Types', fontweight='bold', color='blue')
ax1_twin.tick_params(axis='y', labelcolor='blue')

# Line chart for total production
line = ax1.plot(yearly_data['Crop_Year'], yearly_data['Production']/1e6, color='#2E86AB', linewidth=3, marker='o', label='Total Production')
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Total Production (Million Tonnes)', fontweight='bold', color='#2E86AB')
ax1.set_title('Agricultural Production Trends and Crop Diversity Over Time', fontweight='bold', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked area chart with trend lines - Production by season
ax2 = plt.subplot(3, 3, 2)
seasonal_data = df.groupby(['Crop_Year', 'Season'])['Production'].sum().unstack(fill_value=0)
seasonal_data = seasonal_data.div(1e6)  # Convert to millions

# Create stacked area chart
ax2.stackplot(seasonal_data.index, *[seasonal_data[col] for col in seasonal_data.columns], 
              labels=seasonal_data.columns, alpha=0.7, colors=[colors_seasons.get(col, '#999999') for col in seasonal_data.columns])

# Add trend lines for each season
for i, season in enumerate(seasonal_data.columns):
    if len(seasonal_data[season]) > 1:
        z = np.polyfit(seasonal_data.index, seasonal_data[season], 1)
        p = np.poly1d(z)
        ax2.plot(seasonal_data.index, p(seasonal_data.index), '--', color='black', alpha=0.8, linewidth=2)

ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Production (Million Tonnes)', fontweight='bold')
ax2.set_title('Seasonal Production Trends with Growth Trajectories', fontweight='bold', fontsize=12)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Subplot 3: Dual-axis plot - Area vs Efficiency
ax3 = plt.subplot(3, 3, 3)
yearly_area_eff = df.groupby('Crop_Year').agg({
    'Area': 'mean',
    'Efficiency': 'mean'
}).reset_index()

# Line chart for average area
line1 = ax3.plot(yearly_area_eff['Crop_Year'], yearly_area_eff['Area'], color='#A23B72', linewidth=3, marker='s', label='Avg Area')
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Average Area Cultivated (Hectares)', fontweight='bold', color='#A23B72')
ax3.tick_params(axis='y', labelcolor='#A23B72')

# Bar chart for efficiency
ax3_twin = ax3.twinx()
bars = ax3_twin.bar(yearly_area_eff['Crop_Year'], yearly_area_eff['Efficiency'], alpha=0.6, color='#F18F01', label='Efficiency')
ax3_twin.set_ylabel('Production Efficiency (Tonnes/Hectare)', fontweight='bold', color='#F18F01')
ax3_twin.tick_params(axis='y', labelcolor='#F18F01')
ax3.set_title('Area Cultivation vs Production Efficiency Trends', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Subplot 4: Grouped bar chart with variation coefficient
ax4 = plt.subplot(3, 3, 4)
seasonal_yearly = df.groupby(['Crop_Year', 'Season'])['Production'].sum().unstack(fill_value=0)
seasonal_yearly = seasonal_yearly.div(1e6)

# Create grouped bar chart
x = np.arange(len(seasonal_yearly.index))
width = 0.2
seasons = list(seasonal_yearly.columns)

for i, season in enumerate(seasons):
    ax4.bar(x + i*width, seasonal_yearly[season], width, label=season, 
            color=colors_seasons.get(season, '#999999'), alpha=0.8)

# Calculate and plot variation coefficient
ax4_twin = ax4.twinx()
variation_coeff = seasonal_yearly.apply(lambda row: variation(row[row > 0]) if len(row[row > 0]) > 1 else 0, axis=1)
ax4_twin.plot(x + width, variation_coeff, color='red', linewidth=3, marker='D', label='Seasonal Variation')
ax4_twin.set_ylabel('Coefficient of Variation', fontweight='bold', color='red')
ax4_twin.tick_params(axis='y', labelcolor='red')

ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Production (Million Tonnes)', fontweight='bold')
ax4.set_title('Seasonal Production Patterns and Variability', fontweight='bold', fontsize=12)
ax4.set_xticks(x + width)
ax4.set_xticklabels(seasonal_yearly.index, rotation=45)
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Heatmap with marginal plots
ax5 = plt.subplot(3, 3, 5)
# Sample top districts for readability
top_districts = df.groupby('District_Name')['Production'].sum().nlargest(15).index
district_year_data = df[df['District_Name'].isin(top_districts)].groupby(['Crop_Year', 'District_Name'])['Production'].sum().unstack(fill_value=0)
district_year_data = district_year_data.div(1e6)

# Create heatmap
im = ax5.imshow(district_year_data.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
ax5.set_xticks(range(len(district_year_data.index)))
ax5.set_xticklabels(district_year_data.index, rotation=45)
ax5.set_yticks(range(len(district_year_data.columns)))
ax5.set_yticklabels(district_year_data.columns, fontsize=8)
ax5.set_xlabel('Year', fontweight='bold')
ax5.set_ylabel('District', fontweight='bold')
ax5.set_title('Production Heatmap: Top Districts by Year', fontweight='bold', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
cbar.set_label('Production (Million Tonnes)', fontweight='bold')

# Subplot 6: Multiple line charts for top crops
ax6 = plt.subplot(3, 3, 6)
top_crops = df.groupby('Crop')['Production'].sum().nlargest(5).index
crop_trends = df[df['Crop'].isin(top_crops)].groupby(['Crop_Year', 'Crop'])['Production'].sum().unstack(fill_value=0)
crop_trends = crop_trends.div(1e6)

for i, crop in enumerate(top_crops):
    if crop in crop_trends.columns:
        # Main line
        ax6.plot(crop_trends.index, crop_trends[crop], linewidth=3, marker='o', 
                label=crop, color=colors_main[i % len(colors_main)])
        
        # Confidence band (using rolling standard deviation as proxy)
        rolling_std = crop_trends[crop].rolling(window=3, center=True).std()
        ax6.fill_between(crop_trends.index, 
                        crop_trends[crop] - rolling_std, 
                        crop_trends[crop] + rolling_std, 
                        alpha=0.2, color=colors_main[i % len(colors_main)])

ax6.set_xlabel('Year', fontweight='bold')
ax6.set_ylabel('Production (Million Tonnes)', fontweight='bold')
ax6.set_title('Top 5 Crops Production Trends with Confidence Bands', fontweight='bold', fontsize=12)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# Subplot 7: Scatter plot with regression lines
ax7 = plt.subplot(3, 3, 7)
yearly_scatter = df.groupby('Crop_Year').agg({
    'Area': 'sum',
    'Production': 'sum',
    'Crop': 'nunique'
}).reset_index()

# Scatter plot with bubble sizes
scatter = ax7.scatter(yearly_scatter['Area']/1e6, yearly_scatter['Production']/1e6, 
                     s=yearly_scatter['Crop']*10, alpha=0.7, c=yearly_scatter['Crop_Year'], 
                     cmap='viridis', edgecolors='black', linewidth=1)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_scatter['Area'], yearly_scatter['Production'])
line_x = np.array([yearly_scatter['Area'].min(), yearly_scatter['Area'].max()])
line_y = slope * line_x + intercept
ax7.plot(line_x/1e6, line_y/1e6, 'r--', linewidth=2, label=f'R² = {r_value**2:.3f}')

ax7.set_xlabel('Total Area (Million Hectares)', fontweight='bold')
ax7.set_ylabel('Total Production (Million Tonnes)', fontweight='bold')
ax7.set_title('Area vs Production Relationship with Crop Diversity', fontweight='bold', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

# Add colorbar for years
cbar = plt.colorbar(scatter, ax=ax7, shrink=0.8)
cbar.set_label('Year', fontweight='bold')

# Subplot 8: Box plots with violin plots and trend line
ax8 = plt.subplot(3, 3, 8)
yearly_production_dist = []
years = sorted(df['Crop_Year'].unique())

# Prepare data for box plots
production_by_year = []
for year in years:
    year_data = df[df['Crop_Year'] == year]['Production'].values
    year_data = year_data[year_data > 0]  # Remove zeros
    if len(year_data) > 0:
        production_by_year.append(np.log10(year_data + 1))  # Log transform for better visualization
    else:
        production_by_year.append([])

# Create box plots
bp = ax8.boxplot(production_by_year, positions=range(len(years)), patch_artist=True, 
                widths=0.6, showfliers=False)

# Color the boxes
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(years)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean trend line
means = [np.mean(data) if len(data) > 0 else 0 for data in production_by_year]
ax8.plot(range(len(years)), means, 'ro-', linewidth=3, markersize=6, label='Mean Trend')

ax8.set_xlabel('Year', fontweight='bold')
ax8.set_ylabel('Log10(Production + 1)', fontweight='bold')
ax8.set_title('Production Distribution Evolution with Mean Trends', fontweight='bold', fontsize=12)
ax8.set_xticks(range(len(years)))
ax8.set_xticklabels(years, rotation=45)
ax8.legend()
ax8.grid(True, alpha=0.3)

# Subplot 9: Slope chart for top crops
ax9 = plt.subplot(3, 3, 9)
top_crops_slope = df.groupby('Crop')['Production'].sum().nlargest(8).index
slope_data = df[df['Crop'].isin(top_crops_slope)].groupby(['Crop_Year', 'Crop'])['Production'].sum().unstack(fill_value=0)
slope_data = slope_data.div(1e6)

# Calculate year-over-year changes
years_slope = sorted(slope_data.index)
for i, crop in enumerate(top_crops_slope[:6]):  # Limit to 6 for readability
    if crop in slope_data.columns:
        y_positions = [i] * len(years_slope)
        values = slope_data[crop].values
        
        # Plot connecting lines with color coding for increase/decrease
        for j in range(len(years_slope)-1):
            color = '#2E86AB' if values[j+1] > values[j] else '#C73E1D'
            ax9.plot([years_slope[j], years_slope[j+1]], [values[j], values[j+1]], 
                    color=color, linewidth=3, alpha=0.8)
        
        # Plot points
        ax9.scatter(years_slope, values, s=100, color='black', zorder=5)
        
        # Add crop labels
        ax9.text(years_slope[0]-0.5, values[0], crop, fontsize=9, ha='right', va='center', fontweight='bold')

ax9.set_xlabel('Year', fontweight='bold')
ax9.set_ylabel('Production (Million Tonnes)', fontweight='bold')
ax9.set_title('Production Change Trajectories: Top Crops', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)

# Add legend for color coding
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='#2E86AB', lw=3, label='Increase'),
                   Line2D([0], [0], color='#C73E1D', lw=3, label='Decrease')]
ax9.legend(handles=legend_elements, loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('Comprehensive Analysis of India\'s Agricultural Evolution: Production Trends, Efficiency, and Regional Dynamics', 
             fontsize=16, fontweight='bold', y=0.98)

plt.show()