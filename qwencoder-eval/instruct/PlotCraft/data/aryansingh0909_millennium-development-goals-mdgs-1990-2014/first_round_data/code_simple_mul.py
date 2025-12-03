import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
mdg_data = pd.read_csv('MDG_Data.csv')
country_data = pd.read_csv('MDG_Country.csv')

# Filter for life expectancy indicators
life_expectancy_data = mdg_data[mdg_data['Indicator Name'].str.contains('Life expectancy', case=False, na=False)]

# Melt the data to get all life expectancy values in a single column
year_columns = [str(year) for year in range(1990, 2017)]
life_expectancy_melted = life_expectancy_data.melt(
    id_vars=['Country Name', 'Country Code', 'Indicator Name'],
    value_vars=year_columns,
    var_name='Year',
    value_name='Life_Expectancy'
)

# Remove missing values
life_expectancy_melted = life_expectancy_melted.dropna(subset=['Life_Expectancy'])

# Merge with country data to get regional information
merged_data = life_expectancy_melted.merge(
    country_data[['Country Code', 'Region']], 
    on='Country Code', 
    how='left'
)

# Remove rows with missing region data
merged_data = merged_data.dropna(subset=['Region'])

# Filter out aggregate regions (keep only individual countries)
aggregate_regions = ['Arab World', 'Caribbean small states', 'Central Europe and the Baltics', 
                    'East Asia & Pacific', 'Euro area', 'Europe & Central Asia', 
                    'European Union', 'Fragile and conflict affected situations',
                    'Heavily indebted poor countries (HIPC)', 'High income', 
                    'Latin America & Caribbean', 'Least developed countries',
                    'Low & middle income', 'Low income', 'Lower middle income',
                    'Middle East & North Africa', 'Middle income', 'North America',
                    'OECD members', 'Other small states', 'Pacific island small states',
                    'Small states', 'South Asia', 'Sub-Saharan Africa', 
                    'Upper middle income', 'World']

merged_data = merged_data[~merged_data['Region'].isin(aggregate_regions)]
merged_data = merged_data[merged_data['Region'].notna()]

# Get unique regions and create color palette
regions = merged_data['Region'].unique()
regions = regions[regions != '']  # Remove empty strings
colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))

# Create figure with white background
plt.figure(figsize=(12, 8))
plt.style.use('default')  # Ensure white background

# Create histogram with regional color coding
life_expectancy_values = []
region_labels = []
region_colors = []

for i, region in enumerate(regions):
    region_data = merged_data[merged_data['Region'] == region]['Life_Expectancy']
    if len(region_data) > 0:
        life_expectancy_values.extend(region_data.values)
        region_labels.extend([region] * len(region_data))
        region_colors.extend([colors[i]] * len(region_data))

# Create the main histogram
plt.hist(life_expectancy_values, bins=30, alpha=0.7, color='lightblue', 
         edgecolor='black', linewidth=0.5)

# Create separate histograms for each region to show in legend
for i, region in enumerate(regions):
    region_data = merged_data[merged_data['Region'] == region]['Life_Expectancy']
    if len(region_data) > 0:
        plt.hist(region_data, bins=30, alpha=0.6, color=colors[i], 
                label=region, histtype='step', linewidth=2)

# Styling and labels
plt.title('Distribution of Life Expectancy Across Countries and Years\nby World Region', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Life Expectancy (Years)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Create legend with smaller font and better positioning
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
          frameon=True, fancybox=True, shadow=True)

# Add statistics text box
total_observations = len(life_expectancy_values)
mean_life_exp = np.mean(life_expectancy_values)
median_life_exp = np.median(life_expectancy_values)
std_life_exp = np.std(life_expectancy_values)

stats_text = f'Total Observations: {total_observations:,}\n'
stats_text += f'Mean: {mean_life_exp:.1f} years\n'
stats_text += f'Median: {median_life_exp:.1f} years\n'
stats_text += f'Std Dev: {std_life_exp:.1f} years'

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10)

# Set background to white
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Layout adjustment
plt.tight_layout()
plt.show()