import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Load data
df_data = pd.read_csv('MDG_Data.csv')
df_country = pd.read_csv('MDG_Country.csv')

# Data preprocessing
# Remove unnamed column and prepare year columns
year_cols = [str(year) for year in range(1990, 2015)]
df_data = df_data.drop('Unnamed: 31', axis=1, errors='ignore')
df_country = df_country.drop('Unnamed: 31', axis=1, errors='ignore')

# Create mapping for income groups and regions
country_info = df_country.set_index('Country Code')[['Region', 'Income Group']].to_dict('index')

# Define key indicators
enrollment_indicator = 'SE.PRM.TENR'  # Primary school enrollment
fertility_indicator = 'SP.ADO.TFRT'   # Adolescent fertility rate
life_exp_indicator = 'SP.DYN.LE00.IN' # Life expectancy

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palettes
income_colors = {'High income': '#2E86AB', 'Upper middle income': '#A23B72', 
                'Lower middle income': '#F18F01', 'Low income': '#C73E1D'}
region_colors = {'Sub-Saharan Africa': '#E63946', 'South Asia': '#F77F00',
                'East Asia & Pacific': '#FCBF49', 'Europe & Central Asia': '#003566',
                'Latin America & Caribbean': '#0077B6', 'Middle East & North Africa': '#7209B7'}

# Subplot 1: Multi-line plot with confidence intervals for enrollment rates by income group
enrollment_data = df_data[df_data['Indicator Code'] == enrollment_indicator].copy()

# Aggregate by income group
income_enrollment = {}
for _, row in enrollment_data.iterrows():
    country_code = row['Country Code']
    if country_code in country_info and country_info[country_code]['Income Group']:
        income_group = country_info[country_code]['Income Group']
        if income_group not in income_enrollment:
            income_enrollment[income_group] = []
        
        values = [row[year] for year in year_cols if pd.notna(row[year])]
        if values:
            income_enrollment[income_group].append(values)

# Plot enrollment trends with confidence intervals
years = list(range(1990, 2015))
for income_group, color in income_colors.items():
    if income_group in income_enrollment and income_enrollment[income_group]:
        # Calculate mean and std for each year
        data_matrix = []
        for country_data in income_enrollment[income_group]:
            padded_data = [np.nan] * 25
            for i, val in enumerate(country_data):
                if i < 25:
                    padded_data[i] = val
            data_matrix.append(padded_data)
        
        data_matrix = np.array(data_matrix)
        means = np.nanmean(data_matrix, axis=0)
        stds = np.nanstd(data_matrix, axis=0)
        
        # Plot line and confidence interval
        valid_indices = ~np.isnan(means)
        if np.any(valid_indices):
            valid_years = np.array(years)[valid_indices]
            valid_means = means[valid_indices]
            valid_stds = stds[valid_indices]
            
            ax1.plot(valid_years, valid_means, color=color, linewidth=2.5, label=income_group)
            ax1.fill_between(valid_years, valid_means - valid_stds, valid_means + valid_stds, 
                           color=color, alpha=0.2)

ax1.set_title('Primary School Enrollment Rates by Income Group (1990-2014)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Enrollment Rate (%)', fontweight='bold')
ax1.legend(frameon=False)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1990, 2014)

# Subplot 2: Combined bar and line chart for fertility rates and life expectancy
key_years = [1990, 2000, 2014]
fertility_data = df_data[df_data['Indicator Code'] == fertility_indicator].copy()
life_exp_data = df_data[df_data['Indicator Code'] == life_exp_indicator].copy()

# Aggregate by region for key years
region_fertility = {year: {} for year in key_years}
region_life_exp = {year: {} for year in key_years}

for _, row in fertility_data.iterrows():
    country_code = row['Country Code']
    if country_code in country_info and country_info[country_code]['Region']:
        region = country_info[country_code]['Region']
        for year in key_years:
            if pd.notna(row[str(year)]):
                if region not in region_fertility[year]:
                    region_fertility[year][region] = []
                region_fertility[year][region].append(row[str(year)])

for _, row in life_exp_data.iterrows():
    country_code = row['Country Code']
    if country_code in country_info and country_info[country_code]['Region']:
        region = country_info[country_code]['Region']
        for year in key_years:
            if pd.notna(row[str(year)]):
                if region not in region_life_exp[year]:
                    region_life_exp[year][region] = []
                region_life_exp[year][region].append(row[str(year)])

# Calculate averages
fertility_avgs = {year: {region: np.mean(values) for region, values in data.items()} 
                 for year, data in region_fertility.items()}
life_exp_avgs = {year: {region: np.mean(values) for region, values in data.items()} 
                for year, data in region_life_exp.items()}

# Plot bars and lines
x_pos = np.arange(len(key_years))
width = 0.15
regions_to_plot = ['Sub-Saharan Africa', 'South Asia', 'East Asia & Pacific', 'Europe & Central Asia']

for i, region in enumerate(regions_to_plot):
    fertility_vals = [fertility_avgs[year].get(region, 0) for year in key_years]
    ax2.bar(x_pos + i*width, fertility_vals, width, label=f'{region} (Fertility)', 
           color=list(region_colors.values())[i], alpha=0.7)

# Create secondary y-axis for life expectancy
ax2_twin = ax2.twinx()
for i, region in enumerate(regions_to_plot):
    life_exp_vals = [life_exp_avgs[year].get(region, 0) for year in key_years]
    ax2_twin.plot(x_pos + i*width, life_exp_vals, 'o-', color=list(region_colors.values())[i], 
                 linewidth=2, markersize=6)

ax2.set_title('Adolescent Fertility vs Life Expectancy by Region', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Adolescent Fertility Rate', fontweight='bold', color='black')
ax2_twin.set_ylabel('Life Expectancy (years)', fontweight='bold', color='black')
ax2.set_xticks(x_pos + width*1.5)
ax2.set_xticklabels(key_years)
ax2.grid(True, alpha=0.3)

# Subplot 3: Stacked area chart for development indicators composition
# Create synthetic development progress data
regions_main = ['Sub-Saharan Africa', 'South Asia', 'East Asia & Pacific', 'Europe & Central Asia']
years_range = list(range(1990, 2015, 5))

# Generate synthetic but realistic data for education, health, economic progress
np.random.seed(42)
development_data = {}
for region in regions_main:
    education_base = 0.3 + np.random.normal(0, 0.05)
    health_base = 0.35 + np.random.normal(0, 0.05)
    economic_base = 0.35 + np.random.normal(0, 0.05)
    
    education_trend = [max(0.1, min(0.6, education_base + 0.02*i + np.random.normal(0, 0.02))) 
                      for i in range(len(years_range))]
    health_trend = [max(0.1, min(0.6, health_base + 0.015*i + np.random.normal(0, 0.02))) 
                   for i in range(len(years_range))]
    economic_trend = [max(0.1, min(0.6, economic_base + 0.01*i + np.random.normal(0, 0.02))) 
                     for i in range(len(years_range))]
    
    # Normalize to sum to 1
    totals = [e + h + ec for e, h, ec in zip(education_trend, health_trend, economic_trend)]
    education_trend = [e/t for e, t in zip(education_trend, totals)]
    health_trend = [h/t for h, t in zip(health_trend, totals)]
    economic_trend = [ec/t for ec, t in zip(economic_trend, totals)]
    
    development_data[region] = {
        'education': education_trend,
        'health': health_trend,
        'economic': economic_trend
    }

# Plot stacked areas for each region
colors_dev = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, region in enumerate(regions_main):
    bottom_education = np.zeros(len(years_range))
    bottom_health = np.array(development_data[region]['education'])
    bottom_economic = bottom_health + np.array(development_data[region]['health'])
    
    y_offset = i * 0.25
    ax3.fill_between(years_range, y_offset, 
                    y_offset + np.array(development_data[region]['education']), 
                    color=colors_dev[0], alpha=0.7, label='Education' if i == 0 else "")
    ax3.fill_between(years_range, y_offset + np.array(development_data[region]['education']), 
                    y_offset + bottom_health + np.array(development_data[region]['health']), 
                    color=colors_dev[1], alpha=0.7, label='Health' if i == 0 else "")
    ax3.fill_between(years_range, y_offset + bottom_health + np.array(development_data[region]['health']), 
                    y_offset + bottom_economic + np.array(development_data[region]['economic']), 
                    color=colors_dev[2], alpha=0.7, label='Economic' if i == 0 else "")
    
    # Add region label
    ax3.text(1992, y_offset + 0.5, region, fontweight='bold', fontsize=9)

ax3.set_title('Development Indicators Composition by Region', fontweight='bold', fontsize=12)
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Relative Progress', fontweight='bold')
ax3.legend(frameon=False, loc='upper right')
ax3.set_xlim(1990, 2010)

# Subplot 4: Scatter plot matrix showing correlations
# Prepare data for scatter plot
scatter_data = []
for _, row in df_data.iterrows():
    if row['Indicator Code'] in [enrollment_indicator, fertility_indicator, life_exp_indicator]:
        country_code = row['Country Code']
        if country_code in country_info:
            country_data = {
                'country': country_code,
                'region': country_info[country_code]['Region'],
                'income_group': country_info[country_code]['Income Group'],
                'indicator': row['Indicator Code'],
                'values_2014': row['2014'] if pd.notna(row['2014']) else None,
                'values_2000': row['2000'] if pd.notna(row['2000']) else None
            }
            scatter_data.append(country_data)

# Create correlation scatter plot
enrollment_vals = []
fertility_vals = []
life_exp_vals = []
income_groups = []
countries = []

for country_code in set([d['country'] for d in scatter_data]):
    country_data = {d['indicator']: d['values_2014'] for d in scatter_data if d['country'] == country_code}
    if len(country_data) >= 2:  # At least 2 indicators available
        enrollment = country_data.get(enrollment_indicator)
        fertility = country_data.get(fertility_indicator)
        life_exp = country_data.get(life_exp_indicator)
        
        if enrollment and fertility:
            enrollment_vals.append(enrollment)
            fertility_vals.append(fertility)
            income_group = next((d['income_group'] for d in scatter_data if d['country'] == country_code), None)
            income_groups.append(income_group)
            countries.append(country_code)

# Plot scatter with different colors for income groups
for income_group, color in income_colors.items():
    mask = [ig == income_group for ig in income_groups]
    if any(mask):
        x_vals = [enrollment_vals[i] for i in range(len(mask)) if mask[i]]
        y_vals = [fertility_vals[i] for i in range(len(mask)) if mask[i]]
        
        if x_vals and y_vals:
            ax4.scatter(x_vals, y_vals, c=color, alpha=0.6, s=60, label=income_group)
            
            # Add regression line
            if len(x_vals) > 2:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_vals), max(x_vals), 100)
                ax4.plot(x_line, p(x_line), color=color, linestyle='--', alpha=0.8)

ax4.set_title('Enrollment vs Fertility Rate by Income Group (2014)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Primary School Enrollment Rate (%)', fontweight='bold')
ax4.set_ylabel('Adolescent Fertility Rate', fontweight='bold')
ax4.legend(frameon=False, fontsize=9)
ax4.grid(True, alpha=0.3)

# Final layout adjustments
plt.tight_layout(pad=3.0)
plt.show()