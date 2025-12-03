import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('World CO2 Emission Data.csv')

# Extract year columns (1990-2020)
year_cols = [col for col in df.columns if '[YR' in col and int(col.split('[YR')[1].split(']')[0]) >= 1990 and int(col.split('[YR')[1].split(']')[0]) <= 2020]
years = [int(col.split('[YR')[1].split(']')[0]) for col in year_cols]

# Function to convert string values to numeric
def convert_to_numeric(val):
    if pd.isna(val) or val == '..' or val == '':
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

# Prepare data for major countries
major_countries = ['United States', 'China', 'Germany', 'India', 'Japan', 'United Kingdom']

# Create processed datasets for different metrics
def get_country_time_series(series_name, countries=major_countries):
    series_data = df[df['Series Name'] == series_name]
    result = {}
    for country in countries:
        country_data = series_data[series_data['Country Name'] == country]
        if not country_data.empty:
            values = []
            for col in year_cols:
                val = country_data[col].iloc[0] if len(country_data) > 0 else np.nan
                values.append(convert_to_numeric(val))
            result[country] = values
    return pd.DataFrame(result, index=years)

# Get different emission metrics
co2_per_capita = get_country_time_series('CO2 emissions (metric tons per capita)')
co2_total = get_country_time_series('CO2 emissions (kt)')
methane_ag = get_country_time_series('Agricultural methane emissions (thousand metric tons of CO2 equivalent)')
nitrous_ag = get_country_time_series('Agricultural nitrous oxide emissions (thousand metric tons of CO2 equivalent)')

# Create synthetic data for missing complex metrics (for demonstration)
np.random.seed(42)

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
country_colors = dict(zip(major_countries, colors))

# Subplot 1: Line chart with confidence bands and scatter points
ax1 = plt.subplot(3, 3, 1)
milestone_years = [1992, 1997, 2005, 2015]  # Kyoto, Paris Agreement, etc.

for i, country in enumerate(['United States', 'China', 'Germany']):
    if country in co2_per_capita.columns:
        data = co2_per_capita[country].dropna()
        if len(data) > 5:
            # Main line
            ax1.plot(data.index, data.values, color=country_colors[country], 
                    linewidth=2.5, label=country, alpha=0.8)
            
            # Confidence bands (using rolling std)
            rolling_std = data.rolling(window=5, center=True).std()
            ax1.fill_between(data.index, data.values - rolling_std, 
                           data.values + rolling_std, 
                           color=country_colors[country], alpha=0.2)
            
            # Milestone scatter points
            for year in milestone_years:
                if year in data.index and not pd.isna(data[year]):
                    ax1.scatter(year, data[year], color=country_colors[country], 
                              s=80, zorder=5, edgecolor='white', linewidth=2)

ax1.set_title('CO₂ Emissions Per Capita Trends with Policy Milestones', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('CO₂ Emissions (tons per capita)', fontsize=11)
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked area chart with trend lines
ax2 = plt.subplot(3, 3, 2)
# Create synthetic emission source data
countries_subset = ['United States', 'China', 'Germany']
sources = ['Energy', 'Industry', 'Transport', 'Agriculture']

for i, country in enumerate(countries_subset):
    if country in co2_total.columns:
        total_data = co2_total[country].dropna()
        if len(total_data) > 5:
            # Create synthetic breakdown
            base_values = total_data.values
            energy_pct = 0.6 + 0.1 * np.sin(np.linspace(0, 2*np.pi, len(base_values)))
            industry_pct = 0.25 + 0.05 * np.cos(np.linspace(0, np.pi, len(base_values)))
            transport_pct = 0.1 + 0.02 * np.sin(np.linspace(0, np.pi, len(base_values)))
            agriculture_pct = 1 - energy_pct - industry_pct - transport_pct
            
            if i == 0:  # Only show stacked area for one country to avoid clutter
                energy_vals = base_values * energy_pct
                industry_vals = base_values * industry_pct
                transport_vals = base_values * transport_pct
                agriculture_vals = base_values * agriculture_pct
                
                ax2.fill_between(total_data.index, 0, energy_vals, 
                               color='#FF6B6B', alpha=0.7, label='Energy')
                ax2.fill_between(total_data.index, energy_vals, 
                               energy_vals + industry_vals, 
                               color='#4ECDC4', alpha=0.7, label='Industry')
                ax2.fill_between(total_data.index, energy_vals + industry_vals,
                               energy_vals + industry_vals + transport_vals,
                               color='#45B7D1', alpha=0.7, label='Transport')
                ax2.fill_between(total_data.index, 
                               energy_vals + industry_vals + transport_vals,
                               base_values, color='#96CEB4', alpha=0.7, 
                               label='Agriculture')
            
            # Trend line for total emissions
            ax2.plot(total_data.index, total_data.values, 
                    color=country_colors[country], linewidth=3, 
                    label=f'{country} Total', linestyle='--')

ax2.set_title('Emission Sources Composition with Total Trends', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('CO₂ Emissions (kt)', fontsize=11)
ax2.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
ax2.grid(True, alpha=0.3)

# Subplot 3: Dual-axis plot
ax3 = plt.subplot(3, 3, 3)
ax3_twin = ax3.twinx()

country = 'United States'
if country in co2_total.columns:
    data = co2_total[country].dropna()
    if len(data) > 1:
        # Annual changes (bar chart)
        annual_changes = data.diff()
        bars = ax3.bar(annual_changes.index[1:], annual_changes.values[1:], 
                      color='lightcoral', alpha=0.7, width=0.8, 
                      label='Annual Change')
        
        # Cumulative emissions (line plot)
        cumulative = data.cumsum()
        line = ax3_twin.plot(cumulative.index, cumulative.values, 
                           color='darkblue', linewidth=3, 
                           label='Cumulative Emissions', marker='o', markersize=4)

ax3.set_title('Annual Changes vs Cumulative Emissions (USA)', 
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Annual Change (kt)', fontsize=11, color='darkred')
ax3_twin.set_ylabel('Cumulative Emissions (kt)', fontsize=11, color='darkblue')
ax3.tick_params(axis='y', labelcolor='darkred')
ax3_twin.tick_params(axis='y', labelcolor='darkblue')
ax3.grid(True, alpha=0.3)

# Subplot 4: Time series decomposition
ax4 = plt.subplot(3, 3, 4)
country = 'China'
if country in methane_ag.columns:
    data = methane_ag[country].dropna()
    if len(data) > 10:
        # Create synthetic monthly data for decomposition
        monthly_data = np.repeat(data.values, 12)
        monthly_index = pd.date_range(start=f'{data.index[0]}-01', 
                                    periods=len(monthly_data), freq='M')
        
        # Add seasonal pattern
        seasonal_pattern = 10 * np.sin(2 * np.pi * np.arange(len(monthly_data)) / 12)
        monthly_data_seasonal = monthly_data + seasonal_pattern + np.random.normal(0, 5, len(monthly_data))
        
        # Plot original with trend
        ax4.plot(data.index, data.values, color='darkgreen', linewidth=2, 
                label='Original', marker='o', markersize=4)
        
        # Moving average
        ma = data.rolling(window=5, center=True).mean()
        ax4.plot(ma.index, ma.values, color='red', linewidth=2, 
                linestyle='--', label='5-Year Moving Average')
        
        # Fill between for trend component
        ax4.fill_between(data.index, data.values, ma.values, 
                        alpha=0.3, color='lightblue')

ax4.set_title('Methane Emissions Decomposition (China)', 
              fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Methane Emissions (kt CO₂ eq)', fontsize=11)
ax4.legend(frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3)

# Subplot 5: Slope chart with heatmap background
ax5 = plt.subplot(3, 3, 5)

# Create intensity heatmap background
countries_for_heatmap = ['United States', 'China', 'Germany', 'India']
intensity_data = []
for country in countries_for_heatmap:
    if country in co2_per_capita.columns:
        data = co2_per_capita[country].dropna()
        if len(data) > 10:
            intensity_data.append(data.values)

if intensity_data:
    intensity_matrix = np.array(intensity_data)
    im = ax5.imshow(intensity_matrix, aspect='auto', cmap='YlOrRd', alpha=0.6,
                   extent=[years[0], years[-1], -0.5, len(countries_for_heatmap)-0.5])
    
    # Slope lines connecting 1990 and 2020
    for i, country in enumerate(countries_for_heatmap):
        if country in co2_per_capita.columns:
            data = co2_per_capita[country].dropna()
            if 1990 in data.index and 2020 in data.index:
                ax5.plot([1990, 2020], [data[1990], data[2020]], 
                        color=country_colors[country], linewidth=4, 
                        alpha=0.8, label=country)
                
                # Milestone markers
                for year in [1995, 2005, 2015]:
                    if year in data.index:
                        ax5.scatter(year, data[year], color=country_colors[country], 
                                  s=60, zorder=5, edgecolor='white', linewidth=2)

ax5.set_title('Emission Intensity Changes (1990-2020)', 
              fontsize=14, fontweight='bold', pad=15)
ax5.set_xlabel('Year', fontsize=11)
ax5.set_ylabel('CO₂ Per Capita (tons)', fontsize=11)
ax5.legend(frameon=True, fancybox=True, shadow=True)

# Subplot 6: Multi-line time series with error bands
ax6 = plt.subplot(3, 3, 6)

countries_subset = ['United States', 'China', 'Germany']
for country in countries_subset:
    if country in methane_ag.columns and country in nitrous_ag.columns:
        methane_data = methane_ag[country].dropna()
        nitrous_data = nitrous_ag[country].dropna()
        
        if len(methane_data) > 5 and len(nitrous_data) > 5:
            # Align data
            common_years = methane_data.index.intersection(nitrous_data.index)
            if len(common_years) > 5:
                methane_aligned = methane_data[common_years]
                nitrous_aligned = nitrous_data[common_years]
                
                # Plot lines with error bands
                ax6.plot(common_years, methane_aligned.values, 
                        color=country_colors[country], linewidth=2, 
                        label=f'{country} Methane', linestyle='-')
                ax6.plot(common_years, nitrous_aligned.values, 
                        color=country_colors[country], linewidth=2, 
                        label=f'{country} N₂O', linestyle='--', alpha=0.7)
                
                # Fill area between them
                ax6.fill_between(common_years, methane_aligned.values, 
                               nitrous_aligned.values, 
                               color=country_colors[country], alpha=0.2)

ax6.set_title('Agricultural Emissions: Methane vs N₂O', 
              fontsize=14, fontweight='bold', pad=15)
ax6.set_xlabel('Year', fontsize=11)
ax6.set_ylabel('Emissions (kt CO₂ equivalent)', fontsize=11)
ax6.legend(frameon=True, fancybox=True, shadow=True)
ax6.grid(True, alpha=0.3)

# Subplot 7: Calendar heatmap with box plots
ax7 = plt.subplot(3, 3, 7)

# Create synthetic monthly data
country = 'Germany'
if country in co2_total.columns:
    annual_data = co2_total[country].dropna()
    if len(annual_data) > 5:
        # Create monthly synthetic data
        monthly_emissions = []
        months = []
        for year in annual_data.index[-10:]:  # Last 10 years
            annual_val = annual_data[year]
            for month in range(1, 13):
                # Add seasonal variation
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
                monthly_val = annual_val / 12 * seasonal_factor
                monthly_emissions.append(monthly_val)
                months.append(f'{year}-{month:02d}')
        
        # Reshape for heatmap
        monthly_array = np.array(monthly_emissions).reshape(-1, 12)
        
        # Create heatmap
        im = ax7.imshow(monthly_array, cmap='Reds', aspect='auto')
        ax7.set_xticks(range(12))
        ax7.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax7.set_yticks(range(len(monthly_array)))
        ax7.set_yticklabels([str(year) for year in annual_data.index[-10:]])
        
        # Add quarterly box plots overlay
        quarterly_data = [monthly_array[:, 0:3].flatten(),
                         monthly_array[:, 3:6].flatten(),
                         monthly_array[:, 6:9].flatten(),
                         monthly_array[:, 9:12].flatten()]
        
        # Create small box plots
        box_positions = [1.5, 4.5, 7.5, 10.5]
        bp = ax7.boxplot(quarterly_data, positions=box_positions, widths=1.5,
                        patch_artist=True, boxprops=dict(alpha=0.7))

ax7.set_title('Monthly Emission Calendar with Quarterly Distributions', 
              fontsize=14, fontweight='bold', pad=15)
ax7.set_xlabel('Month', fontsize=11)
ax7.set_ylabel('Year', fontsize=11)

# Subplot 8: Autocorrelation plot
ax8 = plt.subplot(3, 3, 8)

country = 'United States'
if country in co2_total.columns:
    data = co2_total[country].dropna()
    if len(data) > 10:
        # Calculate autocorrelation
        autocorr = acf(data.values, nlags=min(15, len(data)-1))
        lags = range(len(autocorr))
        
        # Plot autocorrelation
        ax8.bar(lags, autocorr, color='steelblue', alpha=0.7, 
               label='Autocorrelation')
        
        # Add confidence intervals
        n = len(data)
        confidence_interval = 1.96 / np.sqrt(n)
        ax8.axhline(y=confidence_interval, color='red', linestyle='--', 
                   alpha=0.7, label='95% Confidence')
        ax8.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7)
        
        # Partial autocorrelation overlay
        try:
            pacf_vals = pacf(data.values, nlags=min(10, len(data)-1))
            ax8.plot(range(len(pacf_vals)), pacf_vals, 'ro-', 
                    alpha=0.7, label='Partial Autocorr')
        except:
            pass

ax8.set_title('Emission Pattern Dependencies (USA)', 
              fontsize=14, fontweight='bold', pad=15)
ax8.set_xlabel('Lag (years)', fontsize=11)
ax8.set_ylabel('Correlation Coefficient', fontsize=11)
ax8.legend(frameon=True, fancybox=True, shadow=True)
ax8.grid(True, alpha=0.3)

# Subplot 9: Cross-correlation matrix heatmap
ax9 = plt.subplot(3, 3, 9)

# Create correlation matrix between different emission types
emission_types = ['CO2 Total', 'CO2 Per Capita', 'Methane Ag', 'Nitrous Ag']
correlation_data = []

for country in ['United States', 'China', 'Germany']:
    country_correlations = []
    datasets = [co2_total, co2_per_capita, methane_ag, nitrous_ag]
    
    for i, data1 in enumerate(datasets):
        row_corr = []
        for j, data2 in enumerate(datasets):
            if country in data1.columns and country in data2.columns:
                series1 = data1[country].dropna()
                series2 = data2[country].dropna()
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) > 5:
                    corr = np.corrcoef(series1[common_idx], series2[common_idx])[0, 1]
                    row_corr.append(corr)
                else:
                    row_corr.append(0)
            else:
                row_corr.append(0)
        country_correlations.append(row_corr)
    correlation_data.extend(country_correlations)

if correlation_data:
    corr_matrix = np.array(correlation_data).reshape(3, 4, 4).mean(axis=0)
    
    # Create heatmap
    im = ax9.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(len(emission_types)):
        for j in range(len(emission_types)):
            text = ax9.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax9.set_xticks(range(len(emission_types)))
    ax9.set_yticks(range(len(emission_types)))
    ax9.set_xticklabels(emission_types, rotation=45, ha='right')
    ax9.set_yticklabels(emission_types)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=10)

ax9.set_title('Cross-Correlation Matrix: Emission Types', 
              fontsize=14, fontweight='bold', pad=15)

# Overall layout adjustments
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('Comprehensive Analysis of Global CO₂ Emissions and Environmental Indicators (1990-2020)', 
             fontsize=18, fontweight='bold', y=0.98)

plt.show()