import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('emission data.csv')

# Focus on 1950-2017 period (available in dataset)
years = [str(year) for year in range(1950, 2018)]
year_cols = [col for col in years if col in df.columns]

# Prepare data for analysis
emission_data = df[['Country'] + year_cols].copy()
emission_data = emission_data.fillna(0)

# Get top 5 emitting countries in 2017
latest_year = year_cols[-1]
top5_countries = emission_data.nlargest(5, latest_year)['Country'].tolist()

# Calculate global emissions
global_emissions = emission_data[year_cols].sum()
years_int = [int(year) for year in year_cols]

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F4C75', '#3282B8', '#BBE1FA', '#1B262C']

# Subplot 1: Line chart with confidence bands + stacked area for top 5 countries
ax1 = plt.subplot(3, 3, 1)

# Get top 5 data
top5_data = []
for country in top5_countries:
    country_data = emission_data[emission_data['Country'] == country][year_cols].values[0]
    top5_data.append(country_data)

top5_array = np.array(top5_data)

# Stacked area chart
ax1.stackplot(years_int, *top5_array, labels=top5_countries[:5], alpha=0.6, colors=colors[:5])

# Add trend lines
for i, country in enumerate(top5_countries[:5]):
    country_data = top5_array[i]
    z = np.polyfit(years_int, country_data, 1)
    p = np.poly1d(z)
    trend = p(years_int)
    ax1.plot(years_int, trend, color='black', linewidth=1.5, alpha=0.8)

ax1.set_title('Top 5 Emitters: Trends with Stacked Areas', fontweight='bold', fontsize=10)
ax1.set_xlabel('Year')
ax1.set_ylabel('CO2 Emissions')
ax1.legend(loc='upper left', fontsize=7)
ax1.grid(True, alpha=0.3)

# Subplot 2: Dual-axis plot - bar chart of decadal growth + line plot
ax2 = plt.subplot(3, 3, 2)

# Calculate decadal growth rates
decades = list(range(1950, 2020, 10))
growth_rates = []
for i in range(len(decades)-1):
    start_year = str(decades[i])
    end_year = str(min(decades[i+1]-1, 2017))
    if start_year in year_cols and end_year in year_cols:
        start_val = global_emissions[start_year]
        end_val = global_emissions[end_year]
        growth_rate = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
        growth_rates.append(growth_rate)

decade_labels = [f"{decades[i]}s" for i in range(len(growth_rates))]
ax2.bar(decade_labels, growth_rates, color=colors[0], alpha=0.7)

# Second axis for acceleration
ax2_twin = ax2.twinx()
global_vals = global_emissions.values
acceleration = np.diff(global_vals, 2)  # Second difference
sample_years = years_int[1:-1]  # Adjust for second difference
ax2_twin.plot(sample_years[::5], acceleration[::5], color=colors[1], linewidth=2, marker='o', markersize=3)

ax2.set_title('Decadal Growth vs Acceleration', fontweight='bold', fontsize=10)
ax2.set_xlabel('Decade')
ax2.set_ylabel('Growth Rate (%)', color=colors[0])
ax2_twin.set_ylabel('Acceleration', color=colors[1])
ax2.grid(True, alpha=0.3)

# Subplot 3: Heatmap with contour lines
ax3 = plt.subplot(3, 3, 3)

# Create emission intensity matrix by decade
countries_sample = emission_data.head(15)  # Reduced for performance
decade_matrix = []

for _, row in countries_sample.iterrows():
    decade_emissions = []
    for decade in range(1950, 2020, 10):
        decade_years = [str(y) for y in range(decade, min(decade+10, 2018)) if str(y) in year_cols]
        if decade_years:
            avg_emission = np.mean([row[year] for year in decade_years])
            decade_emissions.append(avg_emission)
    decade_matrix.append(decade_emissions)

decade_matrix = np.array(decade_matrix)
decade_labels = [f"{d}s" for d in range(1950, 2020, 10)]

# Create heatmap
im = ax3.imshow(decade_matrix, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(len(decade_labels)))
ax3.set_xticklabels(decade_labels)
ax3.set_yticks(range(0, len(countries_sample), 3))
ax3.set_yticklabels([countries_sample.iloc[i]['Country'][:8] for i in range(0, len(countries_sample), 3)])

ax3.set_title('Emission Intensity Heatmap', fontweight='bold', fontsize=10)

# Subplot 4: Time series decomposition with event overlay
ax4 = plt.subplot(3, 3, 4)

# Simple trend calculation
window = min(10, len(global_emissions)//4)
trend = pd.Series(global_emissions.values).rolling(window=window, center=True).mean()
residual = global_emissions.values - trend.fillna(method='bfill').fillna(method='ffill')

ax4.plot(years_int, global_emissions.values, label='Original', color=colors[0], linewidth=2)
ax4.plot(years_int, trend, label='Trend', color=colors[1], linewidth=2)
ax4.scatter(years_int[::3], residual[::3], label='Residuals', color=colors[2], alpha=0.6, s=15)

# Add historical events
events = {'1973': 'Oil Crisis', '1991': 'Soviet Collapse', '2008': 'Financial Crisis'}
for year, event in events.items():
    if year in year_cols:
        ax4.axvline(int(year), color='red', linestyle='--', alpha=0.7)
        ax4.text(int(year), max(global_emissions.values)*0.8, event, rotation=90, fontsize=7)

ax4.set_title('Time Series Decomposition', fontweight='bold', fontsize=10)
ax4.set_xlabel('Year')
ax4.set_ylabel('Global Emissions')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# Subplot 5: Multi-line regional trajectories
ax5 = plt.subplot(3, 3, 5)

# Simplified regions
regions = {
    'North America': ['United States', 'Canada'],
    'Europe': ['Germany', 'United Kingdom', 'France'],
    'Asia': ['China', 'India', 'Japan'],
    'Others': ['Russia', 'Brazil']
}

for i, (region, countries) in enumerate(regions.items()):
    region_data = []
    for country in countries:
        if country in emission_data['Country'].values:
            country_emissions = emission_data[emission_data['Country'] == country][year_cols].values[0]
            region_data.append(country_emissions)
    
    if region_data:
        region_array = np.array(region_data)
        region_mean = np.mean(region_array, axis=0)
        region_std = np.std(region_array, axis=0)
        
        ax5.plot(years_int, region_mean, label=region, color=colors[i], linewidth=2)
        ax5.fill_between(years_int, region_mean - region_std, region_mean + region_std,
                        color=colors[i], alpha=0.2)

ax5.set_title('Regional Emission Trajectories', fontweight='bold', fontsize=10)
ax5.set_xlabel('Year')
ax5.set_ylabel('Average Emissions')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# Subplot 6: Slope chart showing ranking changes
ax6 = plt.subplot(3, 3, 6)

# Get rankings for 1950 and 2017
start_year, end_year = '1950', '2017'
if start_year in year_cols and end_year in year_cols:
    start_rankings = emission_data.nlargest(8, start_year)
    end_rankings = emission_data.nlargest(8, end_year)
    
    for i, country in enumerate(start_rankings['Country'][:6]):
        start_rank = i + 1
        if country in end_rankings['Country'].values:
            end_rank = list(end_rankings['Country']).index(country) + 1
        else:
            end_rank = 8
        
        ax6.plot([0, 1], [start_rank, end_rank], color=colors[i % len(colors)], 
                linewidth=2, marker='o', markersize=4)
        ax6.text(-0.05, start_rank, country[:8], fontsize=7, ha='right')

ax6.set_xlim(-0.3, 1.1)
ax6.set_ylim(0.5, 8.5)
ax6.set_xticks([0, 1])
ax6.set_xticklabels(['1950', '2017'])
ax6.set_ylabel('Ranking')
ax6.set_title('Emission Ranking Changes', fontweight='bold', fontsize=10)
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3)

# Subplot 7: Autocorrelation analysis
ax7 = plt.subplot(3, 3, 7)

# Calculate autocorrelation
lags = range(1, 16)
autocorr = []
for lag in lags:
    if lag < len(global_emissions.values):
        corr = np.corrcoef(global_emissions.values[:-lag], global_emissions.values[lag:])[0,1]
        autocorr.append(corr)

ax7.bar(lags[:len(autocorr)], autocorr, color=colors[0], alpha=0.7)
ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

ax7.set_title('Autocorrelation Analysis', fontweight='bold', fontsize=10)
ax7.set_xlabel('Lag')
ax7.set_ylabel('Autocorrelation')
ax7.grid(True, alpha=0.3)

# Subplot 8: Cross-correlation heatmap
ax8 = plt.subplot(3, 3, 8)

# Calculate cross-correlation matrix for top countries
top_countries_data = []
top_countries_names = []
for country in top5_countries[:4]:  # Reduced for performance
    country_data = emission_data[emission_data['Country'] == country][year_cols].values[0]
    top_countries_data.append(country_data)
    top_countries_names.append(country[:8])

if len(top_countries_data) > 1:
    corr_matrix = np.corrcoef(top_countries_data)
    
    im = ax8.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax8.set_xticks(range(len(top_countries_names)))
    ax8.set_yticks(range(len(top_countries_names)))
    ax8.set_xticklabels(top_countries_names, rotation=45, fontsize=8)
    ax8.set_yticklabels(top_countries_names, fontsize=8)
    
    # Add correlation values
    for i in range(len(top_countries_names)):
        for j in range(len(top_countries_names)):
            ax8.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)

ax8.set_title('Cross-Correlation Matrix', fontweight='bold', fontsize=10)

# Subplot 9: Calendar heatmap with time series overlay
ax9 = plt.subplot(3, 3, 9)

# Create simplified calendar heatmap
annual_emissions = global_emissions.values
calendar_data = []
decade_labels_cal = []

for decade in range(1950, 2020, 10):
    decade_emissions = []
    for year in range(decade, min(decade + 10, 2018)):
        if year in years_int:
            idx = years_int.index(year)
            decade_emissions.append(annual_emissions[idx])
    
    if decade_emissions:
        # Pad to 10 years if needed
        while len(decade_emissions) < 10:
            decade_emissions.append(0)
        calendar_data.append(decade_emissions[:10])
        decade_labels_cal.append(f"{decade}s")

if calendar_data:
    calendar_array = np.array(calendar_data)
    im = ax9.imshow(calendar_array, cmap='Reds', aspect='auto')
    ax9.set_xticks(range(10))
    ax9.set_xticklabels([f"+{i}" for i in range(10)])
    ax9.set_yticks(range(len(decade_labels_cal)))
    ax9.set_yticklabels(decade_labels_cal)

ax9.set_title('Calendar Heatmap', fontweight='bold', fontsize=10)
ax9.set_xlabel('Year in Decade')

# Adjust layout
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Save the plot
plt.savefig('co2_emission_analysis.png', dpi=300, bbox_inches='tight')
plt.show()