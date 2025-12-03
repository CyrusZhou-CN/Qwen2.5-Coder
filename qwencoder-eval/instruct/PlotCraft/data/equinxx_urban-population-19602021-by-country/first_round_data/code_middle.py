import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

# Read the datasets
urban_percent_df = pd.read_csv('urban_percent.csv')
urban_total_df = pd.read_csv('urban_total.csv')

# Clean and prepare the urban percentage data
# Select only the relevant columns and filter for the indicator we need
urban_percent_clean = urban_percent_df[['Country Name', 'Country Code'] + [str(year) for year in range(1960, 2021)]]
urban_percent_clean = urban_percent_clean.dropna(subset=[str(year) for year in range(1960, 2021)])

# Convert to long format for easier analysis
urban_percent_long = urban_percent_clean.melt(
    id_vars=['Country Name', 'Country Code'],
    var_name='Year',
    value_name='Urban_Percentage'
)
urban_percent_long['Year'] = urban_percent_long['Year'].astype(int)

# Calculate global statistics for the top plot
global_stats = urban_percent_long.groupby('Year')['Urban_Percentage'].agg(['mean', 'min', 'max']).reset_index()

# Prepare data for the stacked area chart in bottom plot
# Get the top 5 most populous urban countries in 2020
urban_2020 = urban_total_df[urban_total_df['Country Name'] != 'World']
urban_2020 = urban_2020[['Country Name'] + [str(year) for year in range(2015, 2021)]]

# Filter out rows with NaN values for 2020
urban_2020 = urban_2020.dropna(subset=['2020'])

# Sort by 2020 urban population and get top 5
top_5_countries_2020 = urban_2020.sort_values('2020', ascending=False).head(5)

# Get the full time series for these top 5 countries
top_5_data = []
for _, row in top_5_countries_2020.iterrows():
    country_data = {
        'Country': row['Country Name'],
        'Data': [row[str(year)] for year in range(2015, 2021)]
    }
    top_5_data.append(country_data)

# Create the composite visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Top plot: Global urbanization trends with min-max range
ax1.plot(global_stats['Year'], global_stats['mean'], 
         color='blue', linewidth=2, label='Average Global Urbanization')
ax1.fill_between(global_stats['Year'], global_stats['min'], global_stats['max'], 
                 alpha=0.3, color='lightblue', label='Min-Max Range')

# Add trend line for better visualization
z = np.polyfit(global_stats['Year'], global_stats['mean'], 1)
p = np.poly1d(z)
ax1.plot(global_stats['Year'], p(global_stats['Year']), 
         "--", alpha=0.7, color='red', linewidth=1, label='Trend Line')

# Highlight significant periods
# Find years with significant changes (slopes > 1.5% per decade)
changes = []
for i in range(1, len(global_stats)):
    if abs(global_stats.iloc[i]['mean'] - global_stats.iloc[i-1]['mean']) > 1.5:
        changes.append(i)

# Annotate some key points
key_years = [1960, 1980, 2000, 2020]
for year in key_years:
    if year in global_stats['Year'].values:
        idx = global_stats[global_stats['Year'] == year].index[0]
        ax1.annotate(f'{year}', 
                    xy=(year, global_stats.loc[idx, 'mean']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9)

ax1.set_title('Global Urbanization Trends (1960-2020)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Urban Population Percentage (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: Stacked area chart for top 5 urban countries
years = list(range(2015, 2021))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Create stacked area chart
bottom = np.zeros(len(years))
for i, country_data in enumerate(top_5_data):
    ax2.bar(years, country_data['Data'], bottom=bottom, 
            color=colors[i], label=country_data['Country'], alpha=0.8)
    bottom += np.array(country_data['Data'])

# Add trend lines for each country
for i, country_data in enumerate(top_5_data):
    z = np.polyfit(years, country_data['Data'], 1)
    p = np.poly1d(z)
    ax2.plot(years, p(years), "--", alpha=0.7, color=colors[i], linewidth=1)

ax2.set_title('Cumulative Urban Population Growth for Top 5 Most Populous Urban Countries (2015-2020)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Urban Population (millions)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()