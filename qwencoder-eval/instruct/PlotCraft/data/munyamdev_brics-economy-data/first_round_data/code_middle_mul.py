import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

# Load all datasets with error handling
def load_and_process_data(filename):
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found")
            return pd.DataFrame()
        
        df = pd.read_csv(filename)
        # Split the combined column
        df[['SeriesName', 'SeriesCode', 'CountryName', 'CountryCode', 'Year', 'Value']] = df.iloc[:, 0].str.split(';', expand=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        return df[['CountryName', 'Year', 'Value']].dropna()
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()

# Load all datasets with correct filenames
gdp_growth = load_and_process_data('Economy_Data_89_GDP_growth__annual___.csv')
exports = load_and_process_data('Economy_Data_53_Exports_of_goods_and_services____of_GDP_.csv')
imports = load_and_process_data('Economy_Data_171_Imports_of_goods_and_services____of_GDP_.csv')
agriculture = load_and_process_data('Economy_Data_29_Agriculture__forestry__and_fishing__value_added____of_GDP_.csv')
industry = load_and_process_data('Economy_Data_179_Industry__including_construction___value_added____of_GDP_.csv')
services = load_and_process_data('Economy_Data_317_Services__value_added____of_GDP_.csv')
capital_formation = load_and_process_data('Economy_Data_125_Gross_capital_formation____of_GDP_.csv')
savings = load_and_process_data('Economy_Data_132_Gross_domestic_savings____of_GDP_.csv')

# Define BRICS countries and colors
brics_countries = ['Brazil', 'China', 'India', 'Russian Federation', 'South Africa']
colors = {'Brazil': '#228B22', 'China': '#DC143C', 'India': '#FF8C00', 
          'Russian Federation': '#4169E1', 'South Africa': '#9932CC'}

# Create figure with white background
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Subplot 1: GDP Growth Trends (Top-left)
ax1 = plt.subplot(2, 2, 1, facecolor='white')

# Line chart for annual GDP growth
if not gdp_growth.empty:
    for country in brics_countries:
        country_data = gdp_growth[gdp_growth['CountryName'] == country]
        if not country_data.empty:
            ax1.plot(country_data['Year'], country_data['Value'], 
                    color=colors[country], linewidth=2.5, label=country, alpha=0.8)

    # Calculate and plot decade averages as bars
    decades = [(1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2020)]
    decade_labels = ['1970s', '1980s', '1990s', '2000s', '2010s']
    bar_width = 1.5
    x_positions = np.arange(len(decade_labels))

    for i, country in enumerate(brics_countries):
        decade_avgs = []
        for start, end in decades:
            country_decade = gdp_growth[(gdp_growth['CountryName'] == country) & 
                                       (gdp_growth['Year'] >= start) & 
                                       (gdp_growth['Year'] <= end)]
            avg = country_decade['Value'].mean() if not country_decade.empty else 0
            decade_avgs.append(avg)
        
        if any(avg > 0 for avg in decade_avgs):  # Only plot if there's data
            ax1.bar(x_positions + i * bar_width/5 - bar_width/2, decade_avgs, 
                   width=bar_width/5, color=colors[country], alpha=0.3)

ax1.set_title('GDP Growth Trends: Annual Rates and Decade Averages', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('GDP Growth Rate (%)', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Subplot 2: Economic Structure Evolution (Top-right)
ax2 = plt.subplot(2, 2, 2, facecolor='white')

# Create small multiples for each country
if not agriculture.empty and not industry.empty and not services.empty:
    for i, country in enumerate(brics_countries):
        # Get data for each sector
        agr_data = agriculture[agriculture['CountryName'] == country]
        ind_data = industry[industry['CountryName'] == country]
        srv_data = services[services['CountryName'] == country]
        
        if not agr_data.empty and not ind_data.empty and not srv_data.empty:
            # Merge data
            merged = agr_data.merge(ind_data, on='Year', suffixes=('_agr', '_ind'))
            merged = merged.merge(srv_data, on='Year')
            merged = merged.rename(columns={'Value': 'Services'})
            merged = merged.sort_values('Year')
            
            if len(merged) > 0:
                # Create stacked area chart
                years = merged['Year']
                agriculture_vals = merged['Value_agr']
                industry_vals = merged['Value_ind']
                services_vals = merged['Services']
                
                # Position for small multiples
                x_offset = (i % 3) * 0.3
                y_offset = (i // 3) * 0.4
                
                # Scale down for small multiples
                if len(years) > 1:
                    years_scaled = (years - years.min()) / (years.max() - years.min()) * 0.25 + x_offset
                else:
                    years_scaled = [x_offset + 0.125]
                
                ax2.fill_between(years_scaled, 0, agriculture_vals/100 * 0.3 + y_offset, 
                                color='#8B4513', alpha=0.7, label='Agriculture' if i == 0 else "")
                ax2.fill_between(years_scaled, agriculture_vals/100 * 0.3 + y_offset, 
                                (agriculture_vals + industry_vals)/100 * 0.3 + y_offset, 
                                color='#4682B4', alpha=0.7, label='Industry' if i == 0 else "")
                ax2.fill_between(years_scaled, (agriculture_vals + industry_vals)/100 * 0.3 + y_offset, 
                                (agriculture_vals + industry_vals + services_vals)/100 * 0.3 + y_offset, 
                                color='#32CD32', alpha=0.7, label='Services' if i == 0 else "")
                
                # Add country label
                ax2.text(x_offset + 0.125, y_offset + 0.35, country, 
                        fontweight='bold', ha='center', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[country], alpha=0.3))

ax2.set_title('Economic Structure Evolution: GDP by Sector (1970-2020)', fontweight='bold', fontsize=14, pad=20)
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('Time Progression →', fontweight='bold')
ax2.set_ylabel('Sector Composition', fontweight='bold')
ax2.legend(loc='upper right')

# Subplot 3: Trade Balance Dynamics (Bottom-left)
ax3 = plt.subplot(2, 2, 3, facecolor='white')
ax3_twin = ax3.twinx()

# Line charts for exports and imports
if not exports.empty and not imports.empty:
    for country in brics_countries:
        exp_data = exports[exports['CountryName'] == country]
        imp_data = imports[imports['CountryName'] == country]
        
        if not exp_data.empty and not imp_data.empty:
            ax3.plot(exp_data['Year'], exp_data['Value'], 
                    color=colors[country], linewidth=2.5, linestyle='-', 
                    label=f'{country} Exports', alpha=0.8)
            ax3.plot(imp_data['Year'], imp_data['Value'], 
                    color=colors[country], linewidth=2.5, linestyle='--', 
                    label=f'{country} Imports', alpha=0.8)
            
            # Calculate trade balance and fill area
            merged_trade = exp_data.merge(imp_data, on='Year', suffixes=('_exp', '_imp'))
            if not merged_trade.empty:
                trade_balance = merged_trade['Value_exp'] - merged_trade['Value_imp']
                
                ax3_twin.fill_between(merged_trade['Year'], 0, trade_balance, 
                                     color=colors[country], alpha=0.2)

ax3.set_title('Trade Balance Dynamics: Exports, Imports & Trade Balance', fontweight='bold', fontsize=14, pad=20)
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Exports/Imports (% of GDP)', fontweight='bold')
ax3_twin.set_ylabel('Trade Balance (% of GDP)', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Subplot 4: Investment and Savings Patterns (Bottom-right)
ax4 = plt.subplot(2, 2, 4, facecolor='white')
ax4_twin = ax4.twinx()

# Dual-axis charts for capital formation and savings
if not capital_formation.empty and not savings.empty:
    for country in brics_countries:
        cap_data = capital_formation[capital_formation['CountryName'] == country]
        sav_data = savings[savings['CountryName'] == country]
        
        if not cap_data.empty and not sav_data.empty:
            ax4.plot(cap_data['Year'], cap_data['Value'], 
                    color=colors[country], linewidth=3, linestyle='-', 
                    label=f'{country} Investment', alpha=0.8)
            ax4_twin.plot(sav_data['Year'], sav_data['Value'], 
                         color=colors[country], linewidth=2, linestyle=':', 
                         label=f'{country} Savings', alpha=0.8)
            
            # Scatter plot overlay for decades
            merged_inv_sav = cap_data.merge(sav_data, on='Year', suffixes=('_inv', '_sav'))
            if not merged_inv_sav.empty:
                for decade_start in [1970, 1980, 1990, 2000, 2010]:
                    decade_data = merged_inv_sav[(merged_inv_sav['Year'] >= decade_start) & 
                                                (merged_inv_sav['Year'] < decade_start + 10)]
                    if not decade_data.empty:
                        avg_inv = decade_data['Value_inv'].mean()
                        avg_sav = decade_data['Value_sav'].mean()
                        ax4.scatter(decade_start + 5, avg_inv, 
                                   color=colors[country], s=100, alpha=0.6, 
                                   marker='o', edgecolors='black', linewidth=1)

ax4.set_title('Investment and Savings Patterns: Capital Formation vs Domestic Savings', 
              fontweight='bold', fontsize=14, pad=20)
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Gross Capital Formation (% of GDP)', fontweight='bold')
ax4_twin.set_ylabel('Gross Domestic Savings (% of GDP)', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left', fontsize=8)
ax4_twin.legend(loc='upper right', fontsize=8)

# Add major economic milestone annotations (only if GDP data exists)
if not gdp_growth.empty:
    china_data = gdp_growth[gdp_growth['CountryName'] == 'China']
    if not china_data.empty:
        china_1978 = china_data[china_data['Year'] == 1978]
        if not china_1978.empty:
            ax1.annotate('China Economic Reforms', xy=(1978, china_1978['Value'].iloc[0]), 
                        xytext=(1985, china_1978['Value'].iloc[0] + 5),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=9, ha='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

    # Global Financial Crisis annotation
    crisis_data = gdp_growth[gdp_growth['Year'] == 2008]
    if not crisis_data.empty:
        min_growth = crisis_data['Value'].min()
        ax1.annotate('Global Financial Crisis', xy=(2008, min_growth), 
                    xytext=(2005, min_growth - 3),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                    fontsize=9, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.5))

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig('brics_economic_transformation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()