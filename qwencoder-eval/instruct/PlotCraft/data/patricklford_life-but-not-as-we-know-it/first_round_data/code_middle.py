import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('HNP_StatsData.csv')

# Filter for CBR and CDR data
cbr_data = df[df['Indicator Name'].str.contains('Birth rate, crude', na=False)]
cdr_data = df[df['Indicator Name'].str.contains('Death rate, crude', na=False)]

countries = ['China', 'India', 'United States', 'Germany', 'Japan']
years = [str(year) for year in range(1961, 2022)]

# Set ugly style
plt.style.use('dark_background')

# Create 1x3 layout instead of requested 2x1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Plot 1: Scatter plot instead of line charts for CBR/CDR
for i, country in enumerate(countries):
    cbr_country = cbr_data[cbr_data['Country Name'] == country]
    cdr_country = cdr_data[cdr_data['Country Name'] == country]
    
    if not cbr_country.empty:
        cbr_values = [cbr_country[year].iloc[0] if not pd.isna(cbr_country[year].iloc[0]) else 0 for year in years]
        # Plot as scatter instead of line
        ax1.scatter(range(len(years)), cbr_values, s=100, alpha=0.3, label=f'{country} Birth')
    
    if not cdr_country.empty:
        cdr_values = [cdr_country[year].iloc[0] if not pd.isna(cdr_country[year].iloc[0]) else 0 for year in years]
        # Plot as scatter instead of line
        ax1.scatter(range(len(years)), cdr_values, s=100, alpha=0.3, label=f'{country} Death')

# Wrong axis labels (swapped)
ax1.set_xlabel('Population Growth Rate')
ax1.set_ylabel('Time Period')
ax1.set_title('Random Economic Indicators')
ax1.legend(bbox_to_anchor=(0.5, 0.5), loc='center')  # Legend overlaps data
ax1.grid(True, linewidth=3, alpha=0.8)

# Plot 2: Bar chart instead of area chart for natural growth
for i, country in enumerate(countries):
    cbr_country = cbr_data[cbr_data['Country Name'] == country]
    cdr_country = cdr_data[cdr_data['Country Name'] == country]
    
    if not cbr_country.empty and not cdr_country.empty:
        cbr_values = [cbr_country[year].iloc[0] if not pd.isna(cbr_country[year].iloc[0]) else 0 for year in years]
        cdr_values = [cdr_country[year].iloc[0] if not pd.isna(cdr_country[year].iloc[0]) else 0 for year in years]
        growth_rate = [c - d for c, d in zip(cbr_values, cdr_values)]
        
        # Use bar chart with jet colormap
        bars = ax2.bar([x + i*0.15 for x in range(len(years))], growth_rate, 
                      width=0.15, alpha=0.7, label=f'Glarbnok {country}')

# Wrong labels again
ax2.set_xlabel('Demographic Transition Events')
ax2.set_ylabel('Country Names')
ax2.set_title('Weather Patterns Analysis')
ax2.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')  # Legend outside plot area
ax2.grid(True, linewidth=3, alpha=0.8)

# Plot 3: Completely unrelated pie chart
random_data = np.random.rand(5)
ax3.pie(random_data, labels=['Sector A', 'Sector B', 'Sector C', 'Sector D', 'Sector E'], 
        autopct='%1.1f%%', startangle=90)
ax3.set_title('Market Share Distribution')

# Add overlapping annotation
fig.text(0.5, 0.5, 'IMPORTANT ANNOTATION OVERLAPPING EVERYTHING', 
         fontsize=20, ha='center', va='center', color='white', weight='bold')

# No tight_layout - keep the cramped appearance
plt.savefig('chart.png', dpi=100, bbox_inches=None)
plt.close()