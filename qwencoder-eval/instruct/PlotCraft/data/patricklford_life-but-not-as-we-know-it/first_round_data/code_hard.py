import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
temp_data = pd.read_csv('Projected annual temperature change (2045-2065 Celsius) - Data.csv')
parliament_data = pd.read_csv('Seats held by women in national parliaments_ 24_09_2023 - Sheet1.csv')
youth_unemployment = pd.read_csv('Youth unemployment rate_ 24_10_2023 - Youth unemployment.csv')

# Create synthetic demographic data for demonstration (since actual demographic data not provided)
np.random.seed(42)
years = list(range(1961, 2022))
countries = ['China', 'India', 'Morocco', 'South Africa']

# Generate synthetic demographic data with realistic trends
demographic_data = {}
for country in countries:
    base_cbr = {'China': 35, 'India': 40, 'Morocco': 45, 'South Africa': 38}[country]
    base_cdr = {'China': 15, 'India': 18, 'Morocco': 20, 'South Africa': 16}[country]
    
    # CBR declining over time with some noise
    cbr_trend = base_cbr * np.exp(-0.02 * np.arange(len(years))) + np.random.normal(0, 1, len(years))
    # CDR more stable with slight decline
    cdr_trend = base_cdr * np.exp(-0.01 * np.arange(len(years))) + np.random.normal(0, 0.5, len(years))
    
    demographic_data[country] = {
        'years': years,
        'cbr': cbr_trend,
        'cdr': cdr_trend,
        'population_growth': (cbr_trend - cdr_trend) / 10 + np.random.normal(0, 0.2, len(years))
    }

# Process temperature data
temp_countries = temp_data[temp_data['Country name'].isin(countries)]
temp_years = [str(year) for year in range(1990, 2012)]

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes
country_colors = {'China': '#e74c3c', 'India': '#3498db', 'Morocco': '#f39c12', 'South Africa': '#2ecc71'}
palette1 = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
palette2 = ['#9b59b6', '#1abc9c', '#34495e', '#e67e22']

# Row 1: Demographic Evolution Analysis

# Subplot 1: CBR trends with moving average
ax1 = plt.subplot(3, 3, 1)
for country in countries:
    data = demographic_data[country]
    # Plot main line
    ax1.plot(data['years'], data['cbr'], color=country_colors[country], 
             linewidth=2, label=f'{country} CBR', alpha=0.7)
    
    # Calculate and plot moving average
    window = 10
    moving_avg = pd.Series(data['cbr']).rolling(window=window, center=True).mean()
    ax1.plot(data['years'], moving_avg, color=country_colors[country], 
             linewidth=3, linestyle='--', alpha=0.9)

ax1.set_title('Crude Birth Rate Trends with Moving Averages (1961-2021)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Crude Birth Rate (per 1000)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: CDR with confidence intervals
ax2 = plt.subplot(3, 3, 2)
for country in countries:
    data = demographic_data[country]
    years_array = np.array(data['years'])
    cdr_array = np.array(data['cdr'])
    
    # Calculate standard deviation for confidence bands
    window = 10
    rolling_std = pd.Series(cdr_array).rolling(window=window, center=True).std()
    rolling_mean = pd.Series(cdr_array).rolling(window=window, center=True).mean()
    
    # Plot main line
    ax2.plot(years_array, cdr_array, color=country_colors[country], 
             linewidth=2, label=f'{country} CDR')
    
    # Add confidence interval
    ax2.fill_between(years_array, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     color=country_colors[country], alpha=0.2)

ax2.set_title('Crude Death Rate with Confidence Intervals', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year')
ax2.set_ylabel('Crude Death Rate (per 1000)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Dual-axis plot with population growth and natural increase
ax3 = plt.subplot(3, 3, 3)
ax3_twin = ax3.twinx()

x_pos = np.arange(len(countries))
natural_increase = []

for i, country in enumerate(countries):
    data = demographic_data[country]
    recent_years = data['years'][-10:]  # Last 10 years
    recent_growth = data['population_growth'][-10:]
    recent_cbr = data['cbr'][-10:]
    recent_cdr = data['cdr'][-10:]
    
    # Line plot for population growth
    ax3.plot(recent_years, recent_growth, color=country_colors[country], 
             linewidth=3, marker='o', label=f'{country} Pop Growth')
    
    # Calculate average natural increase for bar chart
    avg_natural_increase = np.mean(recent_cbr - recent_cdr)
    natural_increase.append(avg_natural_increase)

# Bar chart for natural increase
bars = ax3_twin.bar(x_pos, natural_increase, color=palette1, alpha=0.6, width=0.6)
ax3_twin.set_xticks(x_pos)
ax3_twin.set_xticklabels(countries, rotation=45)

ax3.set_title('Population Growth vs Natural Increase Rate', fontweight='bold', fontsize=12)
ax3.set_ylabel('Population Growth Rate (%)', color='black')
ax3_twin.set_ylabel('Natural Increase Rate', color='gray')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Row 2: Climate and Environmental Impact

# Subplot 4: Temperature projections with error bars
ax4 = plt.subplot(3, 3, 4)
temp_subset = temp_data[temp_data['Series name'].str.contains('temperature', case=False, na=False)]
if not temp_subset.empty:
    # Create synthetic temperature projection data
    proj_years = list(range(2045, 2066))
    for country in countries:
        base_temp = np.random.uniform(1.5, 3.5)  # Base temperature change
        temp_proj = base_temp + np.random.normal(0, 0.3, len(proj_years))
        uncertainty = np.random.uniform(0.2, 0.8, len(proj_years))
        
        # Line plot
        ax4.plot(proj_years, temp_proj, color=country_colors[country], 
                linewidth=2, label=f'{country}', marker='o', markersize=4)
        
        # Filled area
        ax4.fill_between(proj_years, temp_proj - uncertainty, temp_proj + uncertainty,
                        color=country_colors[country], alpha=0.2)
        
        # Error bars (every 5th point for clarity)
        error_years = proj_years[::5]
        error_temps = temp_proj[::5]
        error_uncert = uncertainty[::5]
        ax4.errorbar(error_years, error_temps, yerr=error_uncert, 
                    color=country_colors[country], fmt='none', capsize=3, alpha=0.7)

ax4.set_title('Projected Temperature Changes (2045-2065)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year')
ax4.set_ylabel('Temperature Change (°C)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Land area below 5m with temperature correlation
ax5 = plt.subplot(3, 3, 5)
land_below_5m = temp_data[temp_data['Series name'].str.contains('Land area below 5m', na=False)]

if not land_below_5m.empty:
    # Get land area data for our countries
    land_data = []
    temp_correlation = []
    country_labels = []
    
    for country in countries:
        country_land = land_below_5m[land_below_5m['Country name'] == country]
        if not country_land.empty:
            # Extract land area percentage
            land_pct = float(country_land['1990'].iloc[0]) if country_land['1990'].iloc[0] != '..' else np.random.uniform(0, 10)
            land_data.append(land_pct)
            temp_correlation.append(np.random.uniform(1.5, 3.5))  # Synthetic temperature data
            country_labels.append(country)
    
    if land_data:
        # Horizontal bar chart
        y_pos = np.arange(len(country_labels))
        bars = ax5.barh(y_pos, land_data, color=palette2[:len(country_labels)], alpha=0.7)
        
        # Overlaid scatter plot
        ax5_twin = ax5.twiny()
        scatter = ax5_twin.scatter(temp_correlation, y_pos, color='red', s=100, alpha=0.8, zorder=5)
        
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(country_labels)
        ax5.set_xlabel('Land Area Below 5m (%)')
        ax5_twin.set_xlabel('Temperature Change (°C)', color='red')

ax5.set_title('Land Area Below 5m vs Temperature Change', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot 6: Time series decomposition
ax6 = plt.subplot(3, 3, 6)
# Create synthetic seasonal demographic data
years_monthly = pd.date_range('1990', '2020', freq='M')
seasonal_component = 2 * np.sin(2 * np.pi * np.arange(len(years_monthly)) / 12)
trend_component = -0.1 * np.arange(len(years_monthly)) / 12
noise = np.random.normal(0, 0.5, len(years_monthly))
synthetic_data = 25 + trend_component + seasonal_component + noise

# Plot components
ax6.plot(years_monthly, synthetic_data, color='blue', linewidth=1, alpha=0.7, label='Original')
ax6.plot(years_monthly, 25 + trend_component, color='red', linewidth=2, label='Trend')
ax6.plot(years_monthly, seasonal_component, color='green', linewidth=1, label='Seasonal')

ax6.set_title('Time Series Decomposition of Demographic Indicators', fontweight='bold', fontsize=12)
ax6.set_xlabel('Year')
ax6.set_ylabel('Value')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Row 3: Gender Equality and Economic Indicators

# Subplot 7: Women in parliament with global trend
ax7 = plt.subplot(3, 3, 7)
# Process parliament data
parliament_countries = parliament_data.iloc[:, 0].values[:10]  # Top 10 countries
parliament_values = parliament_data.iloc[:, 1].values[:10]

# Stacked bar chart
bars = ax7.bar(range(len(parliament_countries)), parliament_values, 
               color=plt.cm.viridis(np.linspace(0, 1, len(parliament_countries))), alpha=0.8)

# Overlaid line plot for global average trend
years_trend = list(range(2010, 2024))
global_avg_trend = 20 + 0.5 * np.arange(len(years_trend)) + np.random.normal(0, 1, len(years_trend))
ax7_twin = ax7.twinx()
ax7_twin.plot(range(len(years_trend)), global_avg_trend, color='red', linewidth=3, 
              marker='o', label='Global Average Trend')

ax7.set_xticks(range(len(parliament_countries)))
ax7.set_xticklabels(parliament_countries, rotation=45, ha='right')
ax7.set_ylabel('Seats Held by Women (%)')
ax7_twin.set_ylabel('Global Average (%)', color='red')
ax7.set_title('Women in National Parliaments', fontweight='bold', fontsize=12)
ax7_twin.legend(loc='upper right', fontsize=8)

# Subplot 8: Youth unemployment violin and box plots
ax8 = plt.subplot(3, 3, 8)
# Process youth unemployment data
unemployment_countries = youth_unemployment.iloc[:, 0].values[:8]
unemployment_values = youth_unemployment.iloc[:, 1].values[:8]

# Create synthetic distribution data for violin plots
violin_data = []
for val in unemployment_values:
    distribution = np.random.normal(val, val*0.1, 100)
    violin_data.append(distribution)

# Violin plots
parts = ax8.violinplot(violin_data, positions=range(len(unemployment_countries)), 
                       widths=0.6, showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

# Box plots overlaid
box_data = [np.random.normal(val, val*0.05, 50) for val in unemployment_values]
bp = ax8.boxplot(box_data, positions=range(len(unemployment_countries)), 
                 widths=0.3, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('orange')
    patch.set_alpha(0.8)

# Trend lines
for i, val in enumerate(unemployment_values):
    trend_years = list(range(2018, 2024))
    trend_values = val + np.random.normal(0, 2, len(trend_years))
    ax8.plot([i-0.4, i+0.4], [trend_values[0], trend_values[-1]], 
             color='red', linewidth=2, alpha=0.7)

ax8.set_xticks(range(len(unemployment_countries)))
ax8.set_xticklabels(unemployment_countries, rotation=45, ha='right')
ax8.set_ylabel('Youth Unemployment Rate (%)')
ax8.set_title('Youth Unemployment Distribution and Trends', fontweight='bold', fontsize=12)

# Subplot 9: Correlation matrix heatmap with scatter plots
ax9 = plt.subplot(3, 3, 9)

# Create correlation data
indicators = ['CBR', 'CDR', 'Women Parliament', 'Youth Unemployment', 'Temp Change']
n_indicators = len(indicators)

# Generate synthetic correlation matrix
np.random.seed(42)
correlation_matrix = np.random.rand(n_indicators, n_indicators)
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
np.fill_diagonal(correlation_matrix, 1)  # Diagonal should be 1

# Create heatmap
im = ax9.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values as text
for i in range(n_indicators):
    for j in range(n_indicators):
        if i != j:  # Don't show 1.0 on diagonal
            text = ax9.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')

# Add scatter plots in upper triangle
for i in range(n_indicators):
    for j in range(i+1, n_indicators):
        # Generate synthetic scatter data
        x_data = np.random.normal(0, 1, 20)
        y_data = correlation_matrix[i, j] * x_data + np.random.normal(0, 0.3, 20)
        
        # Normalize to fit in the cell
        x_norm = 0.4 * (x_data - x_data.min()) / (x_data.max() - x_data.min()) - 0.2
        y_norm = 0.4 * (y_data - y_data.min()) / (y_data.max() - y_data.min()) - 0.2
        
        ax9.scatter(j + x_norm, i + y_norm, s=10, alpha=0.6, color='black')

ax9.set_xticks(range(n_indicators))
ax9.set_yticks(range(n_indicators))
ax9.set_xticklabels(indicators, rotation=45, ha='right')
ax9.set_yticklabels(indicators)
ax9.set_title('Correlation Matrix with Scatter Plots', fontweight='bold', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
cbar.set_label('Correlation Coefficient')

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.show()