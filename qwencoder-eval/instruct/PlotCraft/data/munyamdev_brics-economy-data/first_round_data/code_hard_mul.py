import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to load and process data with error handling
def load_and_process_data(filename):
    try:
        df = pd.read_csv(filename)
        # Split the combined column
        df[['SeriesName', 'SeriesCode', 'CountryName', 'CountryCode', 'Year', 'Value']] = df.iloc[:, 0].str.split(';', expand=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        return df[['CountryName', 'Year', 'Value']].dropna()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Creating empty dataframe.")
        return pd.DataFrame(columns=['CountryName', 'Year', 'Value'])
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame(columns=['CountryName', 'Year', 'Value'])

# Load all datasets with proper error handling
gdp_growth = load_and_process_data('Economy_Data_89_GDP_growth__annual___.csv')
gdp_per_capita = load_and_process_data('Economy_Data_90_GDP_per_capita__constant_2010_US__.csv')
agriculture = load_and_process_data('Economy_Data_29_Agriculture__forestry__and_fishing__value_added____of_GDP_.csv')
industry = load_and_process_data('Economy_Data_179_Industry__including_construction___value_added____of_GDP_.csv')
services = load_and_process_data('Economy_Data_317_Services__value_added____of_GDP_.csv')
exports = load_and_process_data('Economy_Data_53_Exports_of_goods_and_services____of_GDP_.csv')
imports = load_and_process_data('Economy_Data_171_Imports_of_goods_and_services____of_GDP_.csv')
fdi = load_and_process_data('Economy_Data_81_Foreign_direct_investment__net_inflows____of_GDP_.csv')
portfolio = load_and_process_data('Economy_Data_294_Portfolio_investment__net__BoP__current_US__.csv')
manufacturing = load_and_process_data('Economy_Data_189_Manufacturing__value_added____of_GDP_.csv')
savings = load_and_process_data('Economy_Data_132_Gross_domestic_savings____of_GDP_.csv')
capital_formation = load_and_process_data('Economy_Data_125_Gross_capital_formation____of_GDP_.csv')
external_debt = load_and_process_data('Economy_Data_65_External_debt_stocks____of_GNI_.csv')

# Define BRICS countries and colors
brics_countries = ['Brazil', 'China', 'India', 'Russian Federation', 'South Africa']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
country_colors = dict(zip(brics_countries, colors))

# Create figure with white background
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Subplot 1: GDP per capita growth trends with decade averages
ax1 = plt.subplot(3, 2, 1)
ax1.set_facecolor('white')

# Plot GDP per capita trends
for country in brics_countries:
    if not gdp_per_capita.empty and country in gdp_per_capita['CountryName'].values:
        country_data = gdp_per_capita[gdp_per_capita['CountryName'] == country]
        country_data = country_data.sort_values('Year')
        if len(country_data) > 0:
            ax1.plot(country_data['Year'], country_data['Value'], 
                    color=country_colors[country], linewidth=2.5, label=country, alpha=0.8)

# Add decade average bars if GDP growth data is available
if not gdp_growth.empty:
    decades = [(1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2020)]
    decade_labels = ['1970s', '1980s', '1990s', '2000s', '2010s']
    bar_width = 1.5
    x_pos = np.arange(len(decade_labels))

    ax1_twin = ax1.twinx()
    for i, country in enumerate(brics_countries):
        if country in gdp_growth['CountryName'].values:
            decade_avgs = []
            for start, end in decades:
                country_decade = gdp_growth[(gdp_growth['CountryName'] == country) & 
                                          (gdp_growth['Year'] >= start) & 
                                          (gdp_growth['Year'] <= end)]
                avg_growth = country_decade['Value'].mean() if not country_decade.empty else 0
                decade_avgs.append(avg_growth)
            
            if any(avg > 0 for avg in decade_avgs):
                ax1_twin.bar(x_pos + i*bar_width/5 - bar_width/2, decade_avgs, 
                            width=bar_width/5, alpha=0.6, color=country_colors[country])
    
    ax1_twin.set_ylabel('Average Growth Rate (%)', fontsize=12)
    ax1_twin.set_xticks(x_pos)
    ax1_twin.set_xticklabels(decade_labels)

ax1.set_title('GDP Per Capita Evolution & Average Growth by Decade', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('GDP Per Capita (2010 US$)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: GDP composition evolution (stacked area)
ax2 = plt.subplot(3, 2, 2)
ax2.set_facecolor('white')

# Use China as example if data is available
if not agriculture.empty and not industry.empty and not services.empty:
    china_agr = agriculture[agriculture['CountryName'] == 'China'].sort_values('Year')
    china_ind = industry[industry['CountryName'] == 'China'].sort_values('Year')
    china_srv = services[services['CountryName'] == 'China'].sort_values('Year')

    if not china_agr.empty and not china_ind.empty and not china_srv.empty:
        # Merge data on year
        merged_data = pd.merge(china_agr, china_ind, on='Year', suffixes=('_agr', '_ind'))
        merged_data = pd.merge(merged_data, china_srv, on='Year')
        merged_data.columns = ['CountryName_agr', 'Year', 'Agriculture', 'CountryName_ind', 'Industry', 'CountryName', 'Services']
        
        years = merged_data['Year']
        agr_vals = merged_data['Agriculture']
        ind_vals = merged_data['Industry']
        srv_vals = merged_data['Services']
        
        ax2.fill_between(years, 0, agr_vals, alpha=0.7, color='#8FBC8F', label='Agriculture')
        ax2.fill_between(years, agr_vals, agr_vals + ind_vals, alpha=0.7, color='#4682B4', label='Industry')
        ax2.fill_between(years, agr_vals + ind_vals, agr_vals + ind_vals + srv_vals, 
                        alpha=0.7, color='#DDA0DD', label='Services')
        
        # Add trend lines
        ax2.plot(years, agr_vals, color='#006400', linewidth=2, linestyle='--')
        ax2.plot(years, ind_vals, color='#000080', linewidth=2, linestyle='--')
        ax2.plot(years, srv_vals, color='#8B008B', linewidth=2, linestyle='--')

ax2.set_title('China: Economic Structure Evolution (% of GDP)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Value Added (% of GDP)', fontsize=12)
ax2.legend(loc='center right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Subplot 3: Trade balance analysis
ax3 = plt.subplot(3, 2, 3)
ax3.set_facecolor('white')

if not exports.empty and not imports.empty:
    for country in brics_countries:
        if country in exports['CountryName'].values and country in imports['CountryName'].values:
            exp_data = exports[exports['CountryName'] == country].sort_values('Year')
            imp_data = imports[imports['CountryName'] == country].sort_values('Year')
            
            # Merge on year
            trade_data = pd.merge(exp_data, imp_data, on='Year', suffixes=('_exp', '_imp'))
            
            if not trade_data.empty:
                years = trade_data['Year']
                exports_vals = trade_data['Value_exp']
                imports_vals = trade_data['Value_imp']
                
                ax3.plot(years, exports_vals, color=country_colors[country], 
                        linewidth=2, label=f'{country} Exports', linestyle='-')
                ax3.plot(years, imports_vals, color=country_colors[country], 
                        linewidth=2, alpha=0.6, linestyle='--')
                
                # Fill area between for trade balance
                ax3.fill_between(years, exports_vals, imports_vals, 
                               where=(exports_vals >= imports_vals), 
                               color=country_colors[country], alpha=0.2, interpolate=True)

ax3.set_title('Trade Balance: Exports vs Imports (% of GDP)', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Trade (% of GDP)', fontsize=12)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Subplot 4: Foreign investment flows (dual-axis)
ax4 = plt.subplot(3, 2, 4)
ax4.set_facecolor('white')

# FDI bars
if not fdi.empty:
    for i, country in enumerate(brics_countries):
        if country in fdi['CountryName'].values:
            country_fdi = fdi[fdi['CountryName'] == country].sort_values('Year')
            if not country_fdi.empty:
                # Sample every few years to avoid overcrowding
                sample_data = country_fdi[country_fdi['Year'] % 5 == 0]
                ax4.bar(sample_data['Year'] + i*0.8, sample_data['Value'], 
                       width=0.8, alpha=0.7, color=country_colors[country], 
                       label=f'{country} FDI')

# Portfolio investment lines
if not portfolio.empty:
    ax4_twin = ax4.twinx()
    for country in brics_countries:
        if country in portfolio['CountryName'].values:
            country_port = portfolio[portfolio['CountryName'] == country].sort_values('Year')
            if not country_port.empty and country_port['Value'].notna().any():
                # Convert to billions and plot
                portfolio_vals = pd.to_numeric(country_port['Value'], errors='coerce') / 1e9
                ax4_twin.plot(country_port['Year'], portfolio_vals, 
                             color=country_colors[country], linewidth=2, 
                             linestyle=':', alpha=0.8)
    ax4_twin.set_ylabel('Portfolio Investment (Billion US$)', fontsize=12)

ax4.set_title('Foreign Investment Flows: FDI vs Portfolio Investment', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('FDI Inflows (% of GDP)', fontsize=12)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# Subplot 5: Economic structure transformation
ax5 = plt.subplot(3, 2, 5)
ax5.set_facecolor('white')

# Stacked bars for sectoral composition (using India as example)
if not agriculture.empty and not industry.empty and not services.empty:
    india_agr = agriculture[agriculture['CountryName'] == 'India'].sort_values('Year')
    india_ind = industry[industry['CountryName'] == 'India'].sort_values('Year')
    india_srv = services[services['CountryName'] == 'India'].sort_values('Year')

    if not india_agr.empty and not india_ind.empty and not india_srv.empty:
        # Sample every 5 years for clarity
        sample_years = india_agr[india_agr['Year'] % 5 == 0]['Year']
        
        agr_sample = india_agr[india_agr['Year'].isin(sample_years)]['Value']
        ind_sample = india_ind[india_ind['Year'].isin(sample_years)]['Value']
        srv_sample = india_srv[india_srv['Year'].isin(sample_years)]['Value']
        
        width = 2
        ax5.bar(sample_years, agr_sample, width, label='Agriculture', color='#8FBC8F', alpha=0.8)
        ax5.bar(sample_years, ind_sample, width, bottom=agr_sample, label='Industry', color='#4682B4', alpha=0.8)
        ax5.bar(sample_years, srv_sample, width, bottom=agr_sample + ind_sample, label='Services', color='#DDA0DD', alpha=0.8)

# Manufacturing line overlay
if not manufacturing.empty:
    ax5_twin = ax5.twinx()
    india_manf = manufacturing[manufacturing['CountryName'] == 'India'].sort_values('Year')
    if not india_manf.empty:
        ax5_twin.plot(india_manf['Year'], india_manf['Value'], 
                     color='red', linewidth=3, label='Manufacturing % GDP')
    ax5_twin.set_ylabel('Manufacturing (% of GDP)', fontsize=12)
    ax5_twin.legend(loc='upper right', fontsize=10)

ax5.set_title('India: Sectoral Transformation & Manufacturing Growth', fontsize=14, fontweight='bold', pad=20)
ax5.set_xlabel('Year', fontsize=12)
ax5.set_ylabel('Sectoral Value Added (% of GDP)', fontsize=12)
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)

# Subplot 6: Financial development indicators
ax6 = plt.subplot(3, 2, 6)
ax6.set_facecolor('white')

for country in brics_countries:
    # Savings rate
    if not savings.empty and country in savings['CountryName'].values:
        country_savings = savings[savings['CountryName'] == country].sort_values('Year')
        if not country_savings.empty:
            ax6.plot(country_savings['Year'], country_savings['Value'], 
                    color=country_colors[country], linewidth=2.5, 
                    label=f'{country} Savings', linestyle='-')
    
    # Capital formation
    if not capital_formation.empty and country in capital_formation['CountryName'].values:
        country_capital = capital_formation[capital_formation['CountryName'] == country].sort_values('Year')
        if not country_capital.empty:
            ax6.plot(country_capital['Year'], country_capital['Value'], 
                    color=country_colors[country], linewidth=2, 
                    alpha=0.7, linestyle='--')
    
    # External debt
    if not external_debt.empty and country in external_debt['CountryName'].values:
        country_debt = external_debt[external_debt['CountryName'] == country].sort_values('Year')
        if not country_debt.empty and country_debt['Value'].notna().any():
            ax6.plot(country_debt['Year'], country_debt['Value'], 
                    color=country_colors[country], linewidth=1.5, 
                    alpha=0.5, linestyle=':')

ax6.set_title('Financial Development: Savings, Investment & External Debt', fontsize=14, fontweight='bold', pad=20)
ax6.set_xlabel('Year', fontsize=12)
ax6.set_ylabel('Percentage of GDP/GNI', fontsize=12)

# Custom legend for line styles
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='black', linewidth=2.5, linestyle='-', label='Savings Rate'),
                  Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Capital Formation'),
                  Line2D([0], [0], color='black', linewidth=1.5, linestyle=':', label='External Debt')]
ax6.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add country color legend
country_legend = [Line2D([0], [0], color=color, linewidth=3, label=country) 
                 for country, color in country_colors.items()]
ax6.legend(handles=country_legend, loc='upper left', fontsize=9, title='Countries')
ax6.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('BRICS Economic Evolution 1970-2020: Comprehensive Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('brics_economic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()