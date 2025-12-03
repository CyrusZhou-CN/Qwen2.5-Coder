import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df_data = pd.read_csv('MDG_Data.csv')
df_country = pd.read_csv('MDG_Country.csv')

# Merge data with country information
df_merged = df_data.merge(df_country[['Country Code', 'Region', 'Income Group']], on='Country Code', how='left')

# Filter for years 1990-2014
year_cols = [str(year) for year in range(1990, 2015)]
df_filtered = df_merged[['Country Name', 'Country Code', 'Indicator Name', 'Region', 'Income Group'] + year_cols].copy()

# Define key indicators
life_exp_indicator = 'Life expectancy at birth, total (years)'
education_indicator = 'School enrollment, primary (% gross)'
maternal_mortality_indicator = 'Maternal mortality ratio (modeled estimate, per 100,000 live births)'
gdp_indicator = 'GDP per capita growth (annual %)'
poverty_indicator = 'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)'

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palettes
region_colors = {
    'East Asia & Pacific': '#1f77b4',
    'Europe & Central Asia': '#ff7f0e', 
    'Latin America & Caribbean': '#2ca02c',
    'Middle East & North Africa': '#d62728',
    'North America': '#9467bd',
    'South Asia': '#8c564b',
    'Sub-Saharan Africa': '#e377c2'
}

income_colors = {
    'Low income': '#d62728',
    'Lower middle income': '#ff7f0e',
    'Upper middle income': '#2ca02c',
    'High income': '#1f77b4'
}

# Subplot 1: Life expectancy trends by region with confidence bands and scatter points
life_exp_data = df_filtered[df_filtered['Indicator Name'] == life_exp_indicator].copy()
life_exp_data = life_exp_data.dropna(subset=['Region'])

years = list(range(1990, 2015))
for region in life_exp_data['Region'].unique():
    if pd.isna(region):
        continue
    
    region_data = life_exp_data[life_exp_data['Region'] == region]
    
    # Calculate regional averages and confidence intervals
    regional_means = []
    regional_stds = []
    
    for year in year_cols:
        values = pd.to_numeric(region_data[year], errors='coerce').dropna()
        if len(values) > 0:
            regional_means.append(values.mean())
            regional_stds.append(values.std())
        else:
            regional_means.append(np.nan)
            regional_stds.append(np.nan)
    
    regional_means = np.array(regional_means)
    regional_stds = np.array(regional_stds)
    
    # Plot line with confidence band
    color = region_colors.get(region, '#666666')
    ax1.plot(years, regional_means, color=color, linewidth=2.5, label=region, alpha=0.9)
    
    # Add confidence band
    valid_mask = ~np.isnan(regional_means) & ~np.isnan(regional_stds)
    if np.any(valid_mask):
        ax1.fill_between(years, 
                        regional_means - regional_stds, 
                        regional_means + regional_stds,
                        color=color, alpha=0.2)
    
    # Add scatter points for individual countries (sample)
    sample_countries = region_data.sample(min(3, len(region_data)))
    for _, country in sample_countries.iterrows():
        country_values = [pd.to_numeric(country[year], errors='coerce') for year in year_cols]
        ax1.scatter(years, country_values, color=color, alpha=0.4, s=8)

ax1.set_title('Life Expectancy Trends by Region (1990-2014)', fontweight='bold', fontsize=14)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Life Expectancy (years)', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1990, 2014)

# Subplot 2: Stacked area chart for education enrollment by income group with global average line
education_data = df_filtered[df_filtered['Indicator Name'] == education_indicator].copy()
education_data = education_data.dropna(subset=['Income Group'])

# Calculate enrollment by income group
income_enrollment = {}
for income in education_data['Income Group'].unique():
    if pd.isna(income):
        continue
    income_data = education_data[education_data['Income Group'] == income]
    income_means = []
    for year in year_cols:
        values = pd.to_numeric(income_data[year], errors='coerce').dropna()
        income_means.append(values.mean() if len(values) > 0 else 0)
    income_enrollment[income] = income_means

# Create stacked area chart
bottom = np.zeros(len(years))
for income, values in income_enrollment.items():
    color = income_colors.get(income, '#666666')
    ax2.fill_between(years, bottom, bottom + values, 
                    color=color, alpha=0.7, label=income)
    bottom += values

# Add global average line on secondary y-axis
ax2_twin = ax2.twinx()
global_means = []
for year in year_cols:
    all_values = pd.to_numeric(education_data[year], errors='coerce').dropna()
    global_means.append(all_values.mean() if len(all_values) > 0 else np.nan)

ax2_twin.plot(years, global_means, color='black', linewidth=3, 
             linestyle='--', label='Global Average', alpha=0.8)

ax2.set_title('Primary Education Enrollment by Income Group', fontweight='bold', fontsize=14)
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Enrollment Rate (%)', fontweight='bold')
ax2_twin.set_ylabel('Global Average (%)', fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2_twin.legend(loc='upper right', fontsize=9)
ax2.set_xlim(1990, 2014)

# Subplot 3: Maternal mortality trends by region with error bars
maternal_data = df_filtered[df_filtered['Indicator Name'] == maternal_mortality_indicator].copy()
maternal_data = maternal_data.dropna(subset=['Region'])

# Sample every 5 years for clarity
sample_years = ['1990', '1995', '2000', '2005', '2010', '2014']
sample_year_nums = [1990, 1995, 2000, 2005, 2010, 2014]

for i, region in enumerate(maternal_data['Region'].unique()):
    if pd.isna(region):
        continue
    
    region_data = maternal_data[maternal_data['Region'] == region]
    
    regional_means = []
    regional_errors = []
    
    for year in sample_years:
        values = pd.to_numeric(region_data[year], errors='coerce').dropna()
        if len(values) > 0:
            regional_means.append(values.mean())
            regional_errors.append(values.std() / np.sqrt(len(values)))  # Standard error
        else:
            regional_means.append(np.nan)
            regional_errors.append(np.nan)
    
    color = region_colors.get(region, '#666666')
    ax3.errorbar(sample_year_nums, regional_means, yerr=regional_errors,
                color=color, linewidth=2.5, marker='o', markersize=6,
                capsize=5, capthick=2, label=region, alpha=0.9)

ax3.set_title('Maternal Mortality Trends by Region', fontweight='bold', fontsize=14)
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Maternal Mortality (per 100,000)', fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(1988, 2016)

# Subplot 4: Combined bar and line chart for GDP growth vs poverty reduction
gdp_data = df_filtered[df_filtered['Indicator Name'] == gdp_indicator].copy()
poverty_data = df_filtered[df_filtered['Indicator Name'] == poverty_indicator].copy()

# Calculate 5-year averages for clarity
periods = ['1990-1994', '1995-1999', '2000-2004', '2005-2009', '2010-2014']
period_centers = [1992, 1997, 2002, 2007, 2012]

# GDP growth by region (bars)
gdp_by_region = {}
for region in gdp_data['Region'].unique():
    if pd.isna(region):
        continue
    region_gdp = gdp_data[gdp_data['Region'] == region]
    period_means = []
    
    for i, period in enumerate(periods):
        start_year = 1990 + i*5
        end_year = min(1994 + i*5, 2014)
        period_years = [str(year) for year in range(start_year, end_year + 1)]
        
        all_values = []
        for year in period_years:
            if year in year_cols:
                values = pd.to_numeric(region_gdp[year], errors='coerce').dropna()
                all_values.extend(values)
        
        period_means.append(np.mean(all_values) if all_values else 0)
    
    gdp_by_region[region] = period_means

# Plot bars for GDP growth
bar_width = 0.8
x_positions = np.arange(len(periods))
for i, (region, values) in enumerate(gdp_by_region.items()):
    color = region_colors.get(region, '#666666')
    offset = (i - len(gdp_by_region)/2) * bar_width / len(gdp_by_region)
    ax4.bar(x_positions + offset, values, bar_width/len(gdp_by_region), 
           color=color, alpha=0.7, label=f'{region} (GDP)')

# Global poverty rate line
ax4_twin = ax4.twinx()
global_poverty = []
for i, period in enumerate(periods):
    start_year = 1990 + i*5
    end_year = min(1994 + i*5, 2014)
    period_years = [str(year) for year in range(start_year, end_year + 1)]
    
    all_values = []
    for year in period_years:
        if year in year_cols:
            values = pd.to_numeric(poverty_data[year], errors='coerce').dropna()
            all_values.extend(values)
    
    global_poverty.append(np.mean(all_values) if all_values else np.nan)

ax4_twin.plot(x_positions, global_poverty, color='red', linewidth=4, 
             marker='s', markersize=8, label='Global Poverty Rate')

ax4.set_title('GDP Growth vs Poverty Reduction (5-year periods)', fontweight='bold', fontsize=14)
ax4.set_xlabel('Period', fontweight='bold')
ax4.set_ylabel('GDP Growth Rate (%)', fontweight='bold')
ax4_twin.set_ylabel('Poverty Rate (%)', fontweight='bold', color='red')
ax4.set_xticks(x_positions)
ax4.set_xticklabels(periods, rotation=45)
ax4.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', fontsize=8)
ax4_twin.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left', fontsize=9)

# Final layout adjustments
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.show()