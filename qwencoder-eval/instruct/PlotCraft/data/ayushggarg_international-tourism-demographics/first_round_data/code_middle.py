import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
expenditures_df = pd.read_csv('API_ST.INT.XPND.CD_DS2_en_csv_v2_1929314.csv')
arrivals_df = pd.read_csv('API_ST.INT.ARVL_DS2_en_csv_v2_1927083.csv')
departures_df = pd.read_csv('API_ST.INT.DPRT_DS2_en_csv_v2_1929304.csv')

# Define years of interest (1995-2018)
years = [str(year) for year in range(1995, 2019)]

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#F8961E', '#90323D']

# ============================================================================
# TOP-LEFT: Line chart with error bands for top 10 countries by 2018 arrivals
# ============================================================================

# Get top 10 countries by 2018 arrivals (excluding aggregates)
arrivals_clean = arrivals_df[~arrivals_df['Country Name'].str.contains('World|income|OECD|Arab|Euro|East Asia|Latin America|Sub-Saharan|South Asia|Middle East|North America', na=False)]
arrivals_2018 = arrivals_clean[arrivals_clean['2018'].notna()].copy()
arrivals_2018 = arrivals_2018.sort_values('2018', ascending=False).head(10)

# Plot lines for top 10 countries
for i, (_, country) in enumerate(arrivals_2018.iterrows()):
    country_data = []
    year_indices = []
    
    for j, year in enumerate(years):
        if year in arrivals_df.columns and pd.notna(country[year]) and country[year] > 0:
            country_data.append(country[year] / 1e6)  # Convert to millions
            year_indices.append(j)
    
    if len(country_data) > 2:  # Only plot if we have sufficient data
        ax1.plot(year_indices, country_data, color=colors[i % len(colors)], 
                linewidth=2.5, label=country['Country Name'][:15], alpha=0.8, marker='o', markersize=3)
        
        # Add simple error bands
        if len(country_data) > 3:
            data_array = np.array(country_data)
            std_dev = np.std(data_array) * 0.05  # Small error band
            ax1.fill_between(year_indices, data_array - std_dev, data_array + std_dev, 
                           color=colors[i % len(colors)], alpha=0.1)

# Calculate global average (simplified)
global_avg = []
for year in years:
    if year in arrivals_clean.columns:
        year_data = arrivals_clean[year].dropna()
        if len(year_data) > 10:
            global_avg.append(year_data.mean() / 1e6)
        else:
            global_avg.append(np.nan)

# Plot global average on secondary axis
ax1_twin = ax1.twinx()
valid_indices = [i for i, val in enumerate(global_avg) if not np.isnan(val)]
valid_avg = [global_avg[i] for i in valid_indices]

if len(valid_avg) > 0:
    ax1_twin.plot(valid_indices, valid_avg, color='red', linewidth=2, 
                 linestyle='--', alpha=0.7, label='Global Average')

ax1.set_title('Tourism Arrivals: Top 10 Destinations (1995-2018)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Years', fontsize=10)
ax1.set_ylabel('Tourist Arrivals (Millions)', fontsize=10)
ax1_twin.set_ylabel('Global Avg (Millions)', fontsize=10, color='red')
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1_twin.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, len(years), 5))
ax1.set_xticklabels([years[i] for i in range(0, len(years), 5)])

# ============================================================================
# TOP-RIGHT: Stacked area chart for expenditures
# ============================================================================

# Get top 6 spenders by 2018 expenditures
exp_clean = expenditures_df[~expenditures_df['Country Name'].str.contains('World|income|OECD|Arab|Euro|East Asia|Latin America|Sub-Saharan|South Asia|Middle East|North America', na=False)]
exp_2018 = exp_clean[exp_clean['2018'].notna()].copy()
exp_2018 = exp_2018.sort_values('2018', ascending=False).head(6)

# Prepare data for stacked area chart
exp_data = []
country_names = []
for _, country in exp_2018.iterrows():
    country_exp = []
    for year in years:
        if year in expenditures_df.columns and pd.notna(country[year]) and country[year] > 0:
            country_exp.append(country[year] / 1e9)  # Convert to billions
        else:
            country_exp.append(0)
    exp_data.append(country_exp)
    country_names.append(country['Country Name'][:12])

# Create stacked area chart
if exp_data:
    exp_array = np.array(exp_data)
    ax2.stackplot(range(len(years)), *exp_array, labels=country_names, 
                  colors=colors[:len(country_names)], alpha=0.7)

# Simple ratio calculation
ratio_data = []
for i, year in enumerate(years):
    if year in expenditures_df.columns and year in arrivals_df.columns:
        total_exp = exp_clean[year].sum()
        total_arr = arrivals_clean[year].sum()
        if pd.notna(total_exp) and pd.notna(total_arr) and total_arr > 0:
            ratio_data.append(total_exp / total_arr / 1000)  # Scale down
        else:
            ratio_data.append(np.nan)
    else:
        ratio_data.append(np.nan)

# Plot ratio on secondary axis
ax2_twin = ax2.twinx()
valid_ratio_indices = [i for i, val in enumerate(ratio_data) if not np.isnan(val)]
valid_ratios = [ratio_data[i] for i in valid_ratio_indices]

if len(valid_ratios) > 0:
    ax2_twin.plot(valid_ratio_indices, valid_ratios, color='black', linewidth=2, 
                 marker='o', markersize=3, label='Exp/Arrivals Ratio')

ax2.set_title('Tourism Expenditures: Major Spenders (1995-2018)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Years', fontsize=10)
ax2.set_ylabel('Tourism Expenditures (Billions USD)', fontsize=10)
ax2_twin.set_ylabel('Ratio (scaled)', fontsize=10)
ax2.legend(loc='upper left', fontsize=8)
ax2_twin.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, len(years), 5))
ax2.set_xticklabels([years[i] for i in range(0, len(years), 5)])

# ============================================================================
# BOTTOM-LEFT: Slope chart for departures 2000 vs 2018
# ============================================================================

# Get countries with departure data for both 2000 and 2018
dep_clean = departures_df[~departures_df['Country Name'].str.contains('World|income|OECD|Arab|Euro|East Asia|Latin America|Sub-Saharan|South Asia|Middle East|North America', na=False)]
complete_departures = dep_clean[
    (dep_clean['2000'].notna()) & (dep_clean['2018'].notna()) &
    (dep_clean['2000'] > 500000) & (dep_clean['2018'] > 500000)
].copy()

# Take top 10 countries by 2018 departures
complete_departures = complete_departures.sort_values('2018', ascending=False).head(10)

# Create slope chart
for i, (_, country) in enumerate(complete_departures.iterrows()):
    dep_2000 = country['2000'] / 1e6
    dep_2018 = country['2018'] / 1e6
    
    ax3.plot([0, 1], [dep_2000, dep_2018], color=colors[i % len(colors)], 
             linewidth=2, alpha=0.7, marker='o', markersize=5)
    
    # Add country labels (only for top 6 to avoid overcrowding)
    if i < 6:
        ax3.text(-0.05, dep_2000, country['Country Name'][:10], 
                ha='right', va='center', fontsize=8)

# Add growth rate scatter
growth_rates = []
expenditure_levels = []

for _, country in complete_departures.iterrows():
    growth_rate = ((country['2018'] - country['2000']) / country['2000']) * 100
    growth_rates.append(growth_rate)
    
    # Get expenditure level as proxy for economic development
    country_code = country['Country Code']
    exp_country = expenditures_df[expenditures_df['Country Code'] == country_code]
    if not exp_country.empty and pd.notna(exp_country.iloc[0]['2018']):
        expenditure_levels.append(exp_country.iloc[0]['2018'] / 1e9)
    else:
        expenditure_levels.append(0)

# Scatter plot on twin axis
ax3_twin = ax3.twinx()
if len(growth_rates) > 0:
    scatter_sizes = [max(20, min(200, exp/10)) for exp in expenditure_levels]
    ax3_twin.scatter([1.2] * len(growth_rates), growth_rates, 
                    s=scatter_sizes, c=growth_rates, cmap='RdYlBu', 
                    alpha=0.6, edgecolors='black')

ax3.set_title('Outbound Tourism Growth (2000 vs 2018)', fontsize=12, fontweight='bold')
ax3.set_xlim(-0.3, 1.4)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['2000', '2018'])
ax3.set_ylabel('Departures (Millions)', fontsize=10)
ax3_twin.set_ylabel('Growth Rate (%)', fontsize=10)
ax3.grid(True, alpha=0.3)

# ============================================================================
# BOTTOM-RIGHT: Time series with crisis periods
# ============================================================================

# Calculate simplified global trends
global_arrivals = []
global_departures = []

for year in years:
    # Arrivals
    if year in arrivals_clean.columns:
        arr_data = arrivals_clean[year].dropna()
        if len(arr_data) > 10:
            global_arrivals.append(arr_data.sum() / 1e9)
        else:
            global_arrivals.append(np.nan)
    else:
        global_arrivals.append(np.nan)
    
    # Departures
    if year in dep_clean.columns:
        dep_data = dep_clean[year].dropna()
        if len(dep_data) > 5:
            global_departures.append(dep_data.sum() / 1e9)
        else:
            global_departures.append(np.nan)
    else:
        global_departures.append(np.nan)

# Plot trends
arr_indices = [i for i, val in enumerate(global_arrivals) if not np.isnan(val)]
arr_values = [global_arrivals[i] for i in arr_indices]

dep_indices = [i for i, val in enumerate(global_departures) if not np.isnan(val)]
dep_values = [global_departures[i] for i in dep_indices]

if len(arr_values) > 0:
    ax4.plot(arr_indices, arr_values, color='#2E86AB', linewidth=3, 
             label='Global Arrivals', marker='o', markersize=4)

if len(dep_values) > 0:
    ax4.plot(dep_indices, dep_values, color='#A23B72', linewidth=3, 
             label='Global Departures', marker='s', markersize=4)

# Highlight crisis periods
crisis_periods = [
    (6, 7, '9/11 Impact'),  # 2001-2002 (indices 6-7)
    (13, 14, 'Financial Crisis')  # 2008-2009 (indices 13-14)
]

colors_crisis = ['#FFE5E5', '#E5F3FF']
for i, (start_idx, end_idx, label) in enumerate(crisis_periods):
    if start_idx < len(years) and end_idx < len(years):
        ax4.axvspan(start_idx, end_idx, alpha=0.3, color=colors_crisis[i], label=label)

ax4.set_title('Global Tourism Trends with Crisis Periods', fontsize=12, fontweight='bold')
ax4.set_xlabel('Years', fontsize=10)
ax4.set_ylabel('Tourist Flows (Billions)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, len(years), 5))
ax4.set_xticklabels([years[i] for i in range(0, len(years), 5)])

# Final layout adjustment
plt.tight_layout(pad=2.0)
plt.savefig('tourism_analysis.png', dpi=300, bbox_inches='tight')
plt.show()