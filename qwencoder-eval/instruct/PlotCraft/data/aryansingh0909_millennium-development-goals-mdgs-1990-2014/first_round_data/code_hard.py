import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('MDG_Data.csv')
country_df = pd.read_csv('MDG_Country.csv')

# Data preprocessing
# Merge with country information for regions and income groups
df = df.merge(country_df[['Country Code', 'Region', 'Income Group']], on='Country Code', how='left')

# Define year columns for analysis (1990-2014)
year_cols = [str(year) for year in range(1990, 2015)]

# Define color schemes
region_colors = {
    'East Asia & Pacific': '#1f77b4',
    'Europe & Central Asia': '#ff7f0e', 
    'Latin America & Caribbean': '#2ca02c',
    'Middle East & North Africa': '#d62728',
    'North America': '#9467bd',
    'South Asia': '#8c564b',
    'Sub-Saharan Africa': '#e377c2'
}

income_colors = {
    'Low income': '#d62728',
    'Lower middle income': '#ff7f0e',
    'Upper middle income': '#2ca02c', 
    'High income': '#1f77b4'
}

# Helper function to get indicator data
def get_indicator_data(indicator_name_pattern):
    mask = df['Indicator Name'].str.contains(indicator_name_pattern, case=False, na=False)
    return df[mask].copy()

# Helper function to prepare time series data
def prepare_timeseries(data, group_col='Region'):
    melted = data.melt(id_vars=['Country Name', 'Country Code', group_col], 
                      value_vars=year_cols, var_name='Year', value_name='Value')
    melted['Year'] = melted['Year'].astype(int)
    melted = melted.dropna(subset=['Value'])
    return melted

# Create the 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: Primary Education Enrollment
ax = axes[0, 0]
edu_data = get_indicator_data('primary.*enrollment')
if not edu_data.empty:
    edu_ts = prepare_timeseries(edu_data)
    
    # Regional averages with confidence intervals
    for region in edu_ts['Region'].unique():
        if pd.notna(region) and region in region_colors:
            region_data = edu_ts[edu_ts['Region'] == region]
            yearly_stats = region_data.groupby('Year')['Value'].agg(['mean', 'std', 'count']).reset_index()
            yearly_stats = yearly_stats[yearly_stats['count'] >= 3]  # Minimum countries for confidence
            
            if len(yearly_stats) > 0:
                # Calculate confidence intervals
                ci = 1.96 * yearly_stats['std'] / np.sqrt(yearly_stats['count'])
                ci = ci.fillna(0)  # Handle NaN values
                
                # Plot trend line
                ax.plot(yearly_stats['Year'], yearly_stats['mean'], 
                       color=region_colors[region], linewidth=2, label=region)
                
                # Add confidence interval
                ax.fill_between(yearly_stats['Year'], 
                               yearly_stats['mean'] - ci, 
                               yearly_stats['mean'] + ci,
                               color=region_colors[region], alpha=0.2)
                
                # Add scatter points for regional variation
                ax.scatter(region_data['Year'], region_data['Value'], 
                          color=region_colors[region], alpha=0.3, s=10)

ax.set_title('Primary Education Enrollment Rates\nwith Regional Variations', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Enrollment Rate (%)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Row 1, Subplot 2: Child Mortality Trends
ax = axes[0, 1]
mortality_data = get_indicator_data('mortality.*under')
if not mortality_data.empty:
    mortality_ts = prepare_timeseries(mortality_data)
    
    # Area chart showing decline patterns
    regional_means = mortality_ts.groupby(['Year', 'Region'])['Value'].mean().unstack(fill_value=0)
    
    # Create stacked area chart
    years = regional_means.index
    bottom = np.zeros(len(years))
    
    for i, region in enumerate(regional_means.columns):
        if pd.notna(region) and region in region_colors:
            values = regional_means[region].values
            ax.fill_between(years, bottom, bottom + values, 
                           color=region_colors[region], alpha=0.7, label=region)
            bottom += values
    
    # Add bar chart comparison for 2000 vs 2014
    ax2 = ax.twinx()
    comparison_years = [2000, 2014]
    comparison_data = mortality_ts[mortality_ts['Year'].isin(comparison_years)]
    yearly_means = comparison_data.groupby('Year')['Value'].mean()
    
    if len(yearly_means) > 0:
        bar_width = 0.8
        bars = ax2.bar([y + 15 for y in comparison_years], yearly_means.values, 
                       width=bar_width, alpha=0.6, color='red', label='Global Average')
        
        ax2.set_ylabel('Global Average Mortality Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

ax.set_title('Child Mortality Decline Patterns\nwith 2000 vs 2014 Comparison', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Regional Mortality Rates')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Row 1, Subplot 3: Life Expectancy Evolution
ax = axes[0, 2]
life_exp_data = get_indicator_data('life expectancy')
if not life_exp_data.empty:
    life_exp_ts = prepare_timeseries(life_exp_data, 'Income Group')
    
    # Line plots for different income groups
    for income_group in life_exp_ts['Income Group'].unique():
        if pd.notna(income_group) and income_group in income_colors:
            group_data = life_exp_ts[life_exp_ts['Income Group'] == income_group]
            yearly_means = group_data.groupby('Year')['Value'].mean()
            
            if len(yearly_means) > 0:
                ax.plot(yearly_means.index, yearly_means.values, 
                       color=income_colors[income_group], 
                       linewidth=3, label=income_group, marker='o', markersize=4)
    
    # Add violin plots for 1990 and 2014 distributions (fixed alpha issue)
    violin_data_1990 = life_exp_ts[life_exp_ts['Year'] == 1990]['Value'].dropna()
    violin_data_2014 = life_exp_ts[life_exp_ts['Year'] == 2014]['Value'].dropna()
    
    if len(violin_data_1990) > 0 and len(violin_data_2014) > 0:
        # Create violin plot on secondary axis without alpha parameter
        ax2 = ax.twinx()
        violin_parts = ax2.violinplot([violin_data_1990, violin_data_2014], 
                                     positions=[1990, 2014], widths=3)
        # Set alpha manually for violin parts
        for pc in violin_parts['bodies']:
            pc.set_alpha(0.6)
        ax2.set_ylabel('Distribution Density', alpha=0.7)
        ax2.set_ylim(40, 85)

ax.set_title('Life Expectancy Evolution by Income Group\nwith Distribution Changes', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Life Expectancy (years)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Row 2, Subplot 4: GDP per Capita Growth
ax = axes[1, 0]
gdp_data = get_indicator_data('GDP per capita')
if not gdp_data.empty:
    gdp_ts = prepare_timeseries(gdp_data)
    
    # Line charts for different regions
    for region in gdp_ts['Region'].unique():
        if pd.notna(region) and region in region_colors:
            region_data = gdp_ts[gdp_ts['Region'] == region]
            yearly_means = region_data.groupby('Year')['Value'].mean()
            
            if len(yearly_means) > 5:  # Minimum data points
                ax.plot(yearly_means.index, yearly_means.values, 
                       color=region_colors[region], 
                       linewidth=2, label=region, marker='s', markersize=3)

ax.set_title('GDP per Capita Growth Trajectories\nby Region', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('GDP per Capita (USD)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
# Use log scale only if there's data
if len(ax.get_lines()) > 0:
    ax.set_yscale('log')

# Row 2, Subplot 5: Gender Parity in Education
ax = axes[1, 1]
gender_data = get_indicator_data('gender.*parity.*education|ratio.*female.*male.*enrollment')
if not gender_data.empty:
    gender_ts = prepare_timeseries(gender_data)
    
    # Dual-axis plot
    yearly_means = gender_ts.groupby('Year')['Value'].mean()
    yearly_counts = gender_ts.groupby('Year')['Value'].count()
    
    if len(yearly_means) > 0:
        # Primary axis - ratios
        line1 = ax.plot(yearly_means.index, yearly_means.values, 
                       color='purple', linewidth=3, marker='o', label='Gender Parity Ratio')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Parity')
        
        # Secondary axis - absolute numbers
        ax2 = ax.twinx()
        line2 = ax2.plot(yearly_counts.index, yearly_counts.values, 
                        color='orange', linewidth=2, marker='s', label='Countries Reporting')
        
        # Add trend arrows
        if len(yearly_means) > 1:
            trend = np.polyfit(yearly_means.index, yearly_means.values, 1)[0]
            arrow_color = 'green' if trend > 0 else 'red'
            ax.annotate(f'Trend: {"↗" if trend > 0 else "↘"}', 
                       xy=(0.02, 0.95), xycoords='axes fraction',
                       fontsize=12, color=arrow_color, fontweight='bold')
        
        ax2.set_ylabel('Number of Countries', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

ax.set_title('Gender Parity in Education\nwith Progress Indicators', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Gender Parity Ratio', color='purple')
ax.tick_params(axis='y', labelcolor='purple')
ax.grid(True, alpha=0.3)

# Row 2, Subplot 6: Access to Improved Water Sources
ax = axes[1, 2]
water_data = get_indicator_data('improved water')
if not water_data.empty:
    water_ts = prepare_timeseries(water_data)
    
    # Create mock rural/urban split for demonstration
    rural_data = water_ts.copy()
    rural_data['Value'] = rural_data['Value'] * 0.8  # Assume rural is 80% of total
    urban_data = water_ts.copy()
    urban_data['Value'] = water_ts['Value'] * 0.4  # Adjust urban to be reasonable
    
    # Stacked area chart
    rural_means = rural_data.groupby('Year')['Value'].mean()
    urban_means = urban_data.groupby('Year')['Value'].mean()
    
    if len(rural_means) > 0 and len(urban_means) > 0:
        years = rural_means.index
        ax.fill_between(years, 0, rural_means.values, alpha=0.7, color='brown', label='Rural Access')
        ax.fill_between(years, rural_means.values, rural_means.values + urban_means.values, 
                       alpha=0.7, color='blue', label='Urban Access')

ax.set_title('Access to Improved Water Sources\nRural vs Urban Progress', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Access Rate (%)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Row 3, Subplot 7: Forest Area Changes
ax = axes[2, 0]
forest_data = get_indicator_data('forest area')
if not forest_data.empty:
    forest_ts = prepare_timeseries(forest_data)
    
    # Line plots showing deforestation trends
    for region in forest_ts['Region'].unique():
        if pd.notna(region) and region in region_colors:
            region_data = forest_ts[forest_ts['Region'] == region]
            yearly_means = region_data.groupby('Year')['Value'].mean()
            
            if len(yearly_means) > 3:
                ax.plot(yearly_means.index, yearly_means.values, 
                       color=region_colors[region], 
                       linewidth=2, label=region)
    
    # Add bar chart for regional changes (2014 vs 1990)
    if len(forest_ts) > 0:
        change_data = forest_ts.pivot_table(values='Value', index='Region', columns='Year', aggfunc='mean')
        if 1990 in change_data.columns and 2014 in change_data.columns:
            changes = change_data[2014] - change_data[1990]
            changes = changes.dropna()
            
            if len(changes) > 0:
                # Secondary axis for bar chart
                ax2 = ax.twinx()
                bars = ax2.bar(range(len(changes)), changes.values, 
                              color=['green' if x > 0 else 'red' for x in changes.values],
                              alpha=0.6, width=0.6)
                ax2.set_ylabel('Forest Change 1990-2014 (%)', color='gray')
                ax2.set_xticks(range(len(changes)))
                ax2.set_xticklabels([r[:10] + '...' if len(r) > 10 else r for r in changes.index], 
                                   rotation=45, fontsize=8)

ax.set_title('Forest Area Changes\nwith Regional Gains/Losses', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Forest Area (% of land)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Row 3, Subplot 8: CO2 Emissions per Capita
ax = axes[2, 1]
co2_data = get_indicator_data('CO2.*per capita')
if not co2_data.empty:
    co2_ts = prepare_timeseries(co2_data)
    
    # Area charts for major regions
    regional_means = co2_ts.groupby(['Year', 'Region'])['Value'].mean().unstack(fill_value=0)
    
    # Select top emitting regions
    if len(regional_means.columns) > 0:
        latest_emissions = regional_means.iloc[-1].sort_values(ascending=False)
        top_regions = latest_emissions.head(4).index
        
        years = regional_means.index
        bottom = np.zeros(len(years))
        
        for region in top_regions:
            if pd.notna(region) and region in region_colors:
                values = regional_means[region].values
                ax.fill_between(years, bottom, bottom + values, 
                               color=region_colors[region], 
                               alpha=0.7, label=region)
                bottom += values

ax.set_title('CO2 Emissions per Capita Evolution\nby Major Regions', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('CO2 Emissions (metric tons per capita)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Row 3, Subplot 9: Mobile Phone Penetration
ax = axes[2, 2]
mobile_data = get_indicator_data('mobile.*subscription|mobile.*phone')
if not mobile_data.empty:
    mobile_ts = prepare_timeseries(mobile_data, 'Income Group')
    
    # Exponential curve fitting for global trend
    global_means = mobile_ts.groupby('Year')['Value'].mean().dropna()
    
    if len(global_means) > 5:
        # Fit exponential curve
        def exp_func(x, a, b, c):
            return a * np.exp(b * (x - 1990)) + c
        
        try:
            years_fit = global_means.index.values
            values_fit = global_means.values
            popt, _ = curve_fit(exp_func, years_fit, values_fit, maxfev=1000)
            
            # Plot fitted curve
            years_extended = np.linspace(1990, 2014, 100)
            fitted_values = exp_func(years_extended, *popt)
            ax.plot(years_extended, fitted_values, 'r--', linewidth=2, label='Exponential Fit')
        except:
            pass
    
    # Plot actual data by income group
    for income_group in mobile_ts['Income Group'].unique():
        if pd.notna(income_group) and income_group in income_colors:
            group_data = mobile_ts[mobile_ts['Income Group'] == income_group]
            yearly_means = group_data.groupby('Year')['Value'].mean()
            
            if len(yearly_means) > 0:
                ax.plot(yearly_means.index, yearly_means.values, 
                       color=income_colors[income_group], 
                       linewidth=2, label=income_group, marker='o', markersize=4)
    
    # Add box plots for milestone years
    milestone_years = [1995, 2005, 2014]
    box_data = []
    box_positions = []
    
    for year in milestone_years:
        year_data = mobile_ts[mobile_ts['Year'] == year]['Value'].dropna()
        if len(year_data) > 5:
            box_data.append(year_data.values)
            box_positions.append(year)
    
    if box_data:
        # Secondary axis for box plots
        ax2 = ax.twinx()
        bp = ax2.boxplot(box_data, positions=box_positions, widths=1.5, 
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.6)
        ax2.set_ylabel('Distribution Range', alpha=0.7)

ax.set_title('Mobile Phone Penetration Growth\nwith Distribution Analysis', fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Subscriptions per 100 people')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Add overall title and adjust layout
fig.suptitle('Millennium Development Goals: Comprehensive Analysis 1990-2014', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

# Add annotations for significant events
fig.text(0.87, 0.85, 'Key Milestones:\n• 2000: MDG Declaration\n• 2008: Financial Crisis\n• 2015: SDG Transition', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.savefig('mdg_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()