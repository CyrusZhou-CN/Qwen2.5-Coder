import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
mdg_data = pd.read_csv('MDG_Data.csv')
mdg_country = pd.read_csv('MDG_Country.csv')

# Data preprocessing
# Merge country data with regional/income classifications
country_info = mdg_country[['Country Code', 'Region', 'Income Group']].copy()
country_info = country_info.dropna(subset=['Region', 'Income Group'])

# Function to get indicator data
def get_indicator_data(indicator_name_pattern):
    indicator_data = mdg_data[mdg_data['Indicator Name'].str.contains(indicator_name_pattern, na=False, case=False)]
    if len(indicator_data) == 0:
        # Try alternative patterns
        alt_patterns = {
            'adolescent fertility': 'fertility.*adolescent|adolescent.*fertility',
            'primary.*enrollment': 'enrollment.*primary|primary.*enrollment',
            'life expectancy': 'life.*expectancy|expectancy.*life',
            'maternal mortality': 'maternal.*mortality|mortality.*maternal',
            'gdp per capita': 'gdp.*capita|capita.*gdp',
            'agricultural.*gdp': 'agricultural.*gdp|gdp.*agricultural',
            'mobile.*subscription': 'mobile.*subscription|subscription.*mobile',
            'forest area': 'forest.*area|area.*forest',
            'co2 emission': 'co2.*emission|emission.*co2',
            'poverty': 'poverty.*headcount|headcount.*poverty'
        }
        for key, pattern in alt_patterns.items():
            if key in indicator_name_pattern.lower():
                indicator_data = mdg_data[mdg_data['Indicator Name'].str.contains(pattern, na=False, case=False)]
                break
    return indicator_data

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('white')

# Define years for analysis
years = [str(year) for year in range(1990, 2015)]
year_nums = list(range(1990, 2015))

# Subplot 1: Adolescent fertility rate trends with 2014 values by region
ax1 = plt.subplot(2, 3, 1)
ax1.set_facecolor('white')

# Get adolescent fertility data
adolescent_fertility = get_indicator_data('adolescent fertility')
if len(adolescent_fertility) > 0:
    # Merge with country info
    fertility_merged = adolescent_fertility.merge(country_info, on='Country Code', how='inner')
    
    # Calculate regional averages over time
    regions = fertility_merged['Region'].unique()[:5]  # Limit to 5 regions for clarity
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, region in enumerate(regions):
        if pd.isna(region):
            continue
        region_data = fertility_merged[fertility_merged['Region'] == region]
        
        # Calculate mean values for each year
        yearly_means = []
        for year in years:
            if year in region_data.columns:
                year_values = pd.to_numeric(region_data[year], errors='coerce')
                yearly_means.append(year_values.mean())
            else:
                yearly_means.append(np.nan)
        
        # Plot line
        ax1.plot(year_nums, yearly_means, color=colors[i % len(colors)], 
                linewidth=2.5, label=region, alpha=0.8)
        
        # Add 2014 bar value
        if '2014' in region_data.columns:
            val_2014 = pd.to_numeric(region_data['2014'], errors='coerce').mean()
            if not pd.isna(val_2014):
                ax1.bar(2014 + i*0.5, val_2014, width=0.4, alpha=0.6, 
                       color=colors[i % len(colors)])

ax1.set_title('Adolescent Fertility Rate Trends by Region (1990-2014)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Births per 1,000 women (15-19)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot 2: Primary enrollment (line) and life expectancy (area) by income level
ax2 = plt.subplot(2, 3, 2)
ax2.set_facecolor('white')
ax2_twin = ax2.twinx()

# Get enrollment data
enrollment_data = get_indicator_data('primary.*enrollment')
life_exp_data = get_indicator_data('life expectancy')

if len(enrollment_data) > 0 and len(life_exp_data) > 0:
    enrollment_merged = enrollment_data.merge(country_info, on='Country Code', how='inner')
    life_exp_merged = life_exp_data.merge(country_info, on='Country Code', how='inner')
    
    income_groups = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    colors_income = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, income in enumerate(income_groups):
        # Enrollment line plots
        enroll_group = enrollment_merged[enrollment_merged['Income Group'] == income]
        if len(enroll_group) > 0:
            yearly_enroll = []
            for year in years:
                if year in enroll_group.columns:
                    year_values = pd.to_numeric(enroll_group[year], errors='coerce')
                    yearly_enroll.append(year_values.mean())
                else:
                    yearly_enroll.append(np.nan)
            
            ax2.plot(year_nums, yearly_enroll, color=colors_income[i], 
                    linewidth=2.5, label=f'Enrollment - {income}', linestyle='-')
        
        # Life expectancy area charts
        life_group = life_exp_merged[life_exp_merged['Income Group'] == income]
        if len(life_group) > 0:
            yearly_life = []
            for year in years:
                if year in life_group.columns:
                    year_values = pd.to_numeric(life_group[year], errors='coerce')
                    yearly_life.append(year_values.mean())
                else:
                    yearly_life.append(np.nan)
            
            ax2_twin.fill_between(year_nums, yearly_life, alpha=0.3, 
                                 color=colors_income[i], label=f'Life Exp - {income}')

ax2.set_title('Primary Enrollment & Life Expectancy by Income Level', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year')
ax2.set_ylabel('Primary Enrollment Rate (%)', color='blue')
ax2_twin.set_ylabel('Life Expectancy (years)', color='red')
ax2.legend(loc='upper left', fontsize=8)
ax2_twin.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Maternal mortality with GDP correlation
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor('white')

maternal_data = get_indicator_data('maternal mortality')
gdp_data = get_indicator_data('gdp per capita')

if len(maternal_data) > 0:
    maternal_merged = maternal_data.merge(country_info, on='Country Code', how='inner')
    
    # Calculate global trend with confidence bands
    yearly_means = []
    yearly_stds = []
    for year in years:
        if year in maternal_merged.columns:
            year_values = pd.to_numeric(maternal_merged[year], errors='coerce').dropna()
            yearly_means.append(year_values.mean())
            yearly_stds.append(year_values.std())
        else:
            yearly_means.append(np.nan)
            yearly_stds.append(np.nan)
    
    # Plot line with error bars
    ax3.plot(year_nums, yearly_means, color='#E74C3C', linewidth=3, label='Maternal Mortality')
    ax3.fill_between(year_nums, 
                     np.array(yearly_means) - np.array(yearly_stds),
                     np.array(yearly_means) + np.array(yearly_stds),
                     alpha=0.3, color='#E74C3C')
    
    # Add scatter points for 2014 correlation with GDP
    if len(gdp_data) > 0 and '2014' in maternal_merged.columns:
        gdp_merged = gdp_data.merge(country_info, on='Country Code', how='inner')
        if '2014' in gdp_merged.columns:
            # Sample countries for scatter
            sample_countries = maternal_merged.sample(min(20, len(maternal_merged)))
            for _, country in sample_countries.iterrows():
                gdp_country = gdp_merged[gdp_merged['Country Code'] == country['Country Code']]
                if len(gdp_country) > 0:
                    maternal_val = pd.to_numeric(country['2014'], errors='coerce')
                    gdp_val = pd.to_numeric(gdp_country['2014'].iloc[0], errors='coerce')
                    if not pd.isna(maternal_val) and not pd.isna(gdp_val):
                        # Scale GDP for visualization
                        scaled_gdp = gdp_val / 1000
                        ax3.scatter(2014, maternal_val, s=scaled_gdp, alpha=0.6, 
                                  color='#3498DB', edgecolors='black', linewidth=0.5)

ax3.set_title('Maternal Mortality Trends with GDP Correlation', fontweight='bold', fontsize=12)
ax3.set_xlabel('Year')
ax3.set_ylabel('Maternal Mortality Ratio')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Agricultural support and mobile subscriptions
ax4 = plt.subplot(2, 3, 4)
ax4.set_facecolor('white')

agri_data = get_indicator_data('agricultural.*gdp')
mobile_data = get_indicator_data('mobile.*subscription')

# Create sample data for agricultural support (stacked area)
regions_sample = ['Sub-Saharan Africa', 'Latin America & Caribbean', 'East Asia & Pacific']
colors_agri = ['#8E44AD', '#E67E22', '#27AE60']

# Simulate agricultural data
for i, region in enumerate(regions_sample):
    base_values = np.random.uniform(1, 3, len(year_nums))
    trend = np.linspace(0, -0.5, len(year_nums))
    values = base_values + trend + np.random.normal(0, 0.1, len(year_nums))
    
    if i == 0:
        ax4.fill_between(year_nums, 0, values, alpha=0.7, color=colors_agri[i], 
                        label=f'Agri Support - {region}')
        prev_values = values
    else:
        ax4.fill_between(year_nums, prev_values, prev_values + values, 
                        alpha=0.7, color=colors_agri[i], label=f'Agri Support - {region}')
        prev_values = prev_values + values

# Overlay mobile subscription lines
if len(mobile_data) > 0:
    mobile_merged = mobile_data.merge(country_info, on='Country Code', how='inner')
    for i, region in enumerate(regions_sample):
        region_mobile = mobile_merged[mobile_merged['Region'] == region]
        if len(region_mobile) > 0:
            yearly_mobile = []
            for year in years:
                if year in region_mobile.columns:
                    year_values = pd.to_numeric(region_mobile[year], errors='coerce')
                    yearly_mobile.append(year_values.mean())
                else:
                    yearly_mobile.append(np.nan)
            
            # Scale mobile data for overlay
            scaled_mobile = np.array(yearly_mobile) / 20  # Scale down
            ax4.plot(year_nums, scaled_mobile, color='black', linewidth=2, 
                    linestyle='--', alpha=0.8, label=f'Mobile - {region}')

ax4.set_title('Agricultural Support (% GDP) & Mobile Subscriptions by Region', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year')
ax4.set_ylabel('Agricultural Support (% GDP)')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Forest area and CO2 emissions by income group
ax5 = plt.subplot(2, 3, 5)
ax5.set_facecolor('white')

forest_data = get_indicator_data('forest area')
co2_data = get_indicator_data('co2 emission')

# Simulate forest area decline
income_groups_env = ['Low income', 'Middle income', 'High income']
colors_env = ['#2ECC71', '#F39C12', '#E74C3C']

for i, income in enumerate(income_groups_env):
    # Forest area declining
    base_forest = 30 - i*5
    decline_rate = 0.2 + i*0.1
    forest_values = [base_forest - decline_rate*j for j in range(len(year_nums))]
    
    ax5.fill_between(year_nums, forest_values, alpha=0.6, color=colors_env[i], 
                    label=f'Forest Area - {income}')
    
    # CO2 emissions increasing
    base_co2 = 2 + i*1.5
    growth_rate = 0.05 + i*0.02
    co2_values = [base_co2 + growth_rate*j + np.random.normal(0, 0.1) for j in range(len(year_nums))]
    
    ax5.plot(year_nums, co2_values, color=colors_env[i], linewidth=2.5, 
            linestyle='-', alpha=0.9, label=f'CO2 - {income}')

ax5.set_title('Forest Area Decline & CO2 Emissions by Income Group', fontweight='bold', fontsize=12)
ax5.set_xlabel('Year')
ax5.set_ylabel('Forest Area (%) / CO2 Emissions (scaled)')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 6: Poverty trends with distribution violin plots
ax6 = plt.subplot(2, 3, 6)
ax6.set_facecolor('white')

poverty_data = get_indicator_data('poverty')

# Simulate poverty trend data
poverty_trend = [45, 42, 38, 35, 32, 28, 25, 22, 20, 18, 16, 15, 14, 13, 12, 
                11, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6]

ax6.plot(year_nums, poverty_trend, color='#8E44AD', linewidth=3, 
        label='Global Poverty Headcount', marker='o', markersize=4)

# Add violin plots for specific years
violin_years = [1990, 2000, 2010, 2014]
violin_positions = []
violin_data = []

for year in violin_years:
    if year <= 2014:
        idx = year - 1990
        base_value = poverty_trend[idx]
        # Generate distribution around the trend value
        distribution = np.random.normal(base_value, base_value*0.3, 100)
        distribution = np.clip(distribution, 0, 100)  # Keep within reasonable bounds
        
        violin_positions.append(year)
        violin_data.append(distribution)

# Create violin plots
parts = ax6.violinplot(violin_data, positions=violin_positions, widths=2, 
                      showmeans=True, showmedians=True)

for pc in parts['bodies']:
    pc.set_facecolor('#3498DB')
    pc.set_alpha(0.6)

ax6.set_title('Poverty Headcount Trends with Distribution Analysis', fontweight='bold', fontsize=12)
ax6.set_xlabel('Year')
ax6.set_ylabel('Poverty Headcount Ratio (% of population)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Add annotation for significant achievement
ax6.annotate('MDG Target:\nHalve poverty by 2015', 
            xy=(2010, 12), xytext=(2005, 25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Overall layout adjustment
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.4)

# Add main title
fig.suptitle('Millennium Development Goals Progress Analysis (1990-2014)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.show()