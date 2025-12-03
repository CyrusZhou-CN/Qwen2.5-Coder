import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data efficiently
print("Loading data...")
df = pd.read_csv('global_electricity_production_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Sample data if too large to avoid timeout
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)

# Data preprocessing for different visualizations
print("Preprocessing data...")
# Get total electricity production by country and date
total_elec = df[df['product'] == 'Electricity'].groupby(['country_name', 'date'])['value'].sum().reset_index()

# Get top 5 producing countries (limit data for performance)
if len(total_elec) > 0:
    top_countries = total_elec.groupby('country_name')['value'].mean().nlargest(5).index.tolist()
else:
    top_countries = []

# Classify renewable vs fossil fuels
renewable_sources = ['Hydro', 'Wind', 'Solar', 'Renewables']
fossil_sources = ['Coal', 'Oil', 'Natural Gas', 'Combustible']

# Create renewable vs fossil classification (simplified)
df['energy_type'] = 'Other'
df.loc[df['product'].str.contains('|'.join(renewable_sources), case=False, na=False), 'energy_type'] = 'Renewable'
df.loc[df['product'].str.contains('|'.join(fossil_sources), case=False, na=False), 'energy_type'] = 'Fossil'

# Set up the figure
print("Creating visualizations...")
plt.style.use('default')
fig = plt.figure(figsize=(18, 14), facecolor='white')

# Subplot 1: Top-left - Line chart with error bands for top 5 countries
ax1 = plt.subplot(3, 3, 1)
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']

if len(top_countries) > 0:
    for i, country in enumerate(top_countries[:3]):  # Limit to 3 countries for performance
        country_data = total_elec[total_elec['country_name'] == country].copy()
        if len(country_data) > 5:
            country_data = country_data.sort_values('date')
            
            # Simplified rolling calculation
            country_data['rolling_mean'] = country_data['value'].rolling(window=3, center=True).mean()
            country_data['rolling_std'] = country_data['value'].rolling(window=3, center=True).std()
            
            # Plot line with error bands
            valid_data = country_data.dropna()
            if len(valid_data) > 0:
                ax1.plot(valid_data['date'], valid_data['rolling_mean'], 
                         color=colors[i], linewidth=2, label=country, alpha=0.8)
                ax1.fill_between(valid_data['date'], 
                                 valid_data['rolling_mean'] - valid_data['rolling_std'],
                                 valid_data['rolling_mean'] + valid_data['rolling_std'],
                                 color=colors[i], alpha=0.2)

ax1.set_title('Electricity Production Trends - Top Countries', fontweight='bold', fontsize=10)
ax1.set_xlabel('Date')
ax1.set_ylabel('Production (GWh)')
if len(top_countries) > 0:
    ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Top-center - Stacked area chart
ax2 = plt.subplot(3, 3, 2)

# Simplified energy aggregation
energy_by_type = df[df['energy_type'].isin(['Renewable', 'Fossil'])].groupby(['date', 'energy_type'])['value'].sum().unstack(fill_value=0)

if len(energy_by_type) > 0 and 'Renewable' in energy_by_type.columns and 'Fossil' in energy_by_type.columns:
    # Sample data points for performance
    sample_size = min(100, len(energy_by_type))
    energy_sample = energy_by_type.sample(n=sample_size).sort_index()
    
    ax2.fill_between(energy_sample.index, 0, energy_sample['Fossil'], 
                     color='#8B4513', alpha=0.7, label='Fossil Fuels')
    ax2.fill_between(energy_sample.index, energy_sample['Fossil'], 
                     energy_sample['Fossil'] + energy_sample['Renewable'],
                     color='#228B22', alpha=0.7, label='Renewable')
    ax2.legend(loc='upper left')

ax2.set_title('Energy Source Composition', fontweight='bold', fontsize=10)
ax2.set_xlabel('Date')
ax2.set_ylabel('Production (GWh)')

# Subplot 3: Top-right - Slope chart
ax3 = plt.subplot(3, 3, 3)

# Simplified comparison for available years
if len(total_elec) > 0:
    total_elec_with_year = total_elec.copy()
    total_elec_with_year['year'] = total_elec_with_year['date'].dt.year
    available_years = sorted(total_elec_with_year['year'].unique())
    
    if len(available_years) >= 2:
        year1, year2 = available_years[0], available_years[-1]
        
        comparison_data = total_elec_with_year[total_elec_with_year['year'].isin([year1, year2])]
        comparison_pivot = comparison_data.groupby(['country_name', 'year'])['value'].mean().unstack(fill_value=0)
        
        if year1 in comparison_pivot.columns and year2 in comparison_pivot.columns:
            countries_with_both = comparison_pivot[(comparison_pivot[year1] > 0) & (comparison_pivot[year2] > 0)].index[:10]
            
            for country in countries_with_both:
                val_1 = comparison_pivot.loc[country, year1]
                val_2 = comparison_pivot.loc[country, year2]
                
                color = '#228B22' if val_2 > val_1 else '#DC143C'
                ax3.plot([year1, year2], [val_1, val_2], color=color, alpha=0.7, linewidth=1.5)
        
        ax3.set_title(f'Production Change: {year1} vs {year2}', fontweight='bold', fontsize=10)
    else:
        ax3.set_title('Production Change Analysis', fontweight='bold', fontsize=10)
else:
    ax3.set_title('Production Change Analysis', fontweight='bold', fontsize=10)

ax3.set_xlabel('Year')
ax3.set_ylabel('Average Production (GWh)')
ax3.grid(True, alpha=0.3)

# Subplot 4: Middle-left - Time series decomposition
ax4 = plt.subplot(3, 3, 4)

if len(total_elec) > 0:
    global_total = total_elec.groupby('date')['value'].sum().reset_index()
    global_total = global_total.sort_values('date')

    if len(global_total) > 10:
        # Simple trend calculation
        global_total['trend'] = global_total['value'].rolling(window=min(6, len(global_total)//2), center=True).mean()
        
        ax4.plot(global_total['date'], global_total['value'], color='#2E86AB', linewidth=1, alpha=0.5, label='Original')
        ax4.plot(global_total['date'], global_total['trend'], color='#F18F01', linewidth=2, label='Trend')
        ax4.legend()

ax4.set_title('Time Series Decomposition', fontweight='bold', fontsize=10)
ax4.set_xlabel('Date')
ax4.set_ylabel('Production')
ax4.grid(True, alpha=0.3)

# Subplot 5: Middle-center - Monthly heatmap
ax5 = plt.subplot(3, 3, 5)

if len(total_elec) > 0:
    total_elec_copy = total_elec.copy()
    total_elec_copy['year'] = total_elec_copy['date'].dt.year
    total_elec_copy['month'] = total_elec_copy['date'].dt.month
    
    monthly_data = total_elec_copy.groupby(['year', 'month'])['value'].sum().reset_index()
    monthly_data.columns = ['year', 'month', 'production']

    if len(monthly_data) > 0:
        heatmap_data = monthly_data.pivot(index='year', columns='month', values='production')
        
        # Limit size for performance
        if len(heatmap_data) > 20:
            heatmap_data = heatmap_data.iloc[-20:]
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax5, cbar=True)

ax5.set_title('Monthly Production Heatmap', fontweight='bold', fontsize=10)

# Subplot 6: Middle-right - Autocorrelation
ax6 = plt.subplot(3, 3, 6)

if len(total_elec) > 0:
    global_total = total_elec.groupby('date')['value'].sum().reset_index()
    
    if len(global_total) > 20:
        production_values = global_total['value'].dropna().values
        max_lags = min(12, len(production_values)//4)
        lags = range(1, max_lags)
        
        autocorr = []
        for lag in lags:
            if lag < len(production_values):
                corr = np.corrcoef(production_values[:-lag], production_values[lag:])[0,1]
                autocorr.append(corr)
            else:
                autocorr.append(0)
        
        if len(autocorr) > 0:
            ax6.bar(lags, autocorr, color='#2E86AB', alpha=0.7)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax6.set_title('Autocorrelation Analysis', fontweight='bold', fontsize=10)
ax6.set_xlabel('Lag')
ax6.set_ylabel('Autocorrelation')
ax6.grid(True, alpha=0.3)

# Subplot 7: Bottom-left - Volatility analysis
ax7 = plt.subplot(3, 3, 7)

energy_sources = ['Hydro', 'Wind', 'Solar']
colors_sources = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, source in enumerate(energy_sources):
    source_data = df[df['product'].str.contains(source, case=False, na=False)].groupby('date')['value'].sum().reset_index()
    if len(source_data) > 5:
        source_data = source_data.sort_values('date')
        source_data['volatility'] = source_data['value'].rolling(window=3).std()
        
        valid_data = source_data.dropna()
        if len(valid_data) > 0:
            ax7.plot(valid_data['date'], valid_data['volatility'], 
                    color=colors_sources[i], linewidth=2, label=source, alpha=0.8)

ax7.set_title('Production Volatility', fontweight='bold', fontsize=10)
ax7.set_xlabel('Date')
ax7.set_ylabel('Volatility')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Subplot 8: Bottom-center - Seasonal analysis
ax8 = plt.subplot(3, 3, 8)

if len(total_elec) > 0:
    seasonal_data = total_elec.copy()
    seasonal_data['month'] = seasonal_data['date'].dt.month
    monthly_avg = seasonal_data.groupby('month')['value'].mean()

    if len(monthly_avg) > 0:
        months = monthly_avg.index
        values = monthly_avg.values
        
        ax8.bar(months, values, color='#2E86AB', alpha=0.7)
        ax8.set_xticks(range(1, 13))
        ax8.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

ax8.set_title('Monthly Production Patterns', fontweight='bold', fontsize=10)
ax8.set_xlabel('Month')
ax8.set_ylabel('Avg Production')
ax8.grid(True, alpha=0.3)

# Subplot 9: Bottom-right - Growth analysis
ax9 = plt.subplot(3, 3, 9)

if len(total_elec) > 0:
    global_total = total_elec.groupby('date')['value'].sum().reset_index()
    global_total['year'] = global_total['date'].dt.year
    yearly_production = global_total.groupby('year')['value'].sum()
    
    if len(yearly_production) > 1:
        growth_rates = yearly_production.pct_change() * 100
        
        ax9.bar(yearly_production.index, yearly_production.values, color='#2E86AB', alpha=0.6, label='Production')
        
        # Secondary axis for growth
        ax9_twin = ax9.twinx()
        valid_growth = growth_rates.dropna()
        if len(valid_growth) > 0:
            ax9_twin.plot(valid_growth.index, valid_growth.values, color='#FF6B35', linewidth=2, marker='o', label='Growth %')
            ax9_twin.set_ylabel('Growth (%)', color='#FF6B35')

ax9.set_title('Production & Growth', fontweight='bold', fontsize=10)
ax9.set_xlabel('Year')
ax9.set_ylabel('Production')

# Adjust layout
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

print("Visualization complete!")
plt.savefig('electricity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()