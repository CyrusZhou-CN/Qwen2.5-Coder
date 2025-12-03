import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('World CO2 Emission Data.csv')

# Filter for CO2 emissions per capita data
co2_per_capita = df[df['Series Code'] == 'EN.ATM.CO2E.PC'].copy()

# Get the year columns for 1990-2020
year_columns = [f'{year} [YR{year}]' for year in range(1990, 2021)]
years = list(range(1990, 2021))

# Select relevant columns
data_columns = ['Country Name'] + year_columns
co2_data = co2_per_capita[data_columns].copy()

# Replace '..' with NaN and convert to numeric
for col in year_columns:
    co2_data[col] = pd.to_numeric(co2_data[col].replace('..', np.nan), errors='coerce')

# Get unique countries
countries = co2_data['Country Name'].unique()

# Create the visualization with white background and professional styling
plt.figure(figsize=(12, 8))
plt.style.use('default')  # Ensure white background

# Define a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Plot each country's data
for i, country in enumerate(countries):
    country_data = co2_data[co2_data['Country Name'] == country]
    
    if not country_data.empty:
        # Extract values for the years
        values = []
        for col in year_columns:
            values.append(country_data[col].iloc[0])
        
        # Convert to pandas Series for easier interpolation
        series = pd.Series(values, index=years)
        
        # Interpolate missing values where reasonable (only if there are some valid values)
        if series.notna().sum() > 1:
            series = series.interpolate(method='linear', limit_direction='both')
        
        # Plot the line
        plt.plot(years, series, 
                color=colors[i % len(colors)], 
                linewidth=2.5, 
                marker='o', 
                markersize=4,
                label=country,
                alpha=0.9)

# Styling and labels
plt.title('Evolution of CO₂ Emissions Per Capita (1990-2020)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('CO₂ Emissions (metric tons per capita)', fontsize=12, fontweight='bold')

# Customize the grid
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Customize legend
plt.legend(frameon=True, fancybox=True, shadow=True, 
          loc='upper left', fontsize=10)

# Set axis properties
plt.xlim(1990, 2020)
plt.xticks(range(1990, 2021, 5), fontsize=10)
plt.yticks(fontsize=10)

# Ensure y-axis starts from 0 for better comparison
plt.ylim(bottom=0)

# Remove top and right spines for cleaner look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Layout adjustment
plt.tight_layout()
plt.show()