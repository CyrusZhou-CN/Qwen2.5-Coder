import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Load available data
temp_data = pd.read_csv('Projected annual temperature change (2045-2065 Celsius) - Data.csv')
women_parliament = pd.read_csv('Seats held by women in national parliaments_ 24_09_2023 - Sheet1.csv')
youth_unemployment = pd.read_csv('Youth unemployment rate_ 24_10_2023 - Youth unemployment.csv')

# Clean and prepare women's parliament data
women_parliament.columns = ['Country', 'Percentage']
women_parliament = women_parliament.head(15)  # Top 15 countries

# Clean and prepare youth unemployment data
youth_unemployment.columns = ['Country', 'Rate']
youth_unemployment['Country'] = youth_unemployment['Country'].str.strip()

# Merge datasets for slope chart
merged_data = pd.merge(women_parliament, youth_unemployment, on='Country', how='inner')

# Simulate demographic data with more realistic patterns
np.random.seed(42)
years = list(range(1961, 2022))
countries = ['China', 'India', 'USA', 'Germany']

# Create more realistic demographic data
demographic_data = {}
for country in countries:
    if country == 'China':
        cbr = 45 - 0.6 * np.arange(len(years)) + 3 * np.sin(0.1 * np.arange(len(years))) + np.random.normal(0, 1, len(years))
        cdr = 15 - 0.15 * np.arange(len(years)) + np.random.normal(0, 0.5, len(years))
    elif country == 'India':
        cbr = 42 - 0.4 * np.arange(len(years)) + 2 * np.sin(0.08 * np.arange(len(years))) + np.random.normal(0, 1.5, len(years))
        cdr = 18 - 0.2 * np.arange(len(years)) + np.random.normal(0, 0.8, len(years))
    elif country == 'USA':
        cbr = 25 - 0.2 * np.arange(len(years)) + 1.5 * np.sin(0.12 * np.arange(len(years))) + np.random.normal(0, 1, len(years))
        cdr = 9 + 0.03 * np.arange(len(years)) + np.random.normal(0, 0.3, len(years))
    else:  # Germany
        cbr = 18 - 0.15 * np.arange(len(years)) + np.random.normal(0, 0.8, len(years))
        cdr = 11 + 0.03 * np.arange(len(years)) + np.random.normal(0, 0.3, len(years))
    
    demographic_data[country] = {
        'cbr': np.maximum(cbr, 5),
        'cdr': np.maximum(cdr, 3),
        'co2': np.random.exponential(2, len(years)) * (1 + 0.05 * np.arange(len(years)))
    }

# Create smoothed CO2 data for top 6 countries (reduced for better color distinction)
co2_countries = ['China', 'USA', 'India', 'Russia', 'Japan', 'Germany']
co2_data_raw = np.random.exponential(1.5, (len(co2_countries), len(years))) * np.outer(np.linspace(1, 4, len(co2_countries)), np.linspace(1, 3, len(years)))

# Apply smoothing to reduce noise
co2_data_smooth = []
for i in range(len(co2_countries)):
    smoothed = signal.savgol_filter(co2_data_raw[i], window_length=5, polyorder=2)
    co2_data_smooth.append(smoothed)

# Create the comprehensive 3x2 subplot grid
fig = plt.figure(figsize=(24, 28))
fig.patch.set_facecolor('white')

# Add overall title
fig.suptitle('Global Demographic and Socioeconomic Trends Analysis (1961-2021)', 
             fontsize=20, fontweight='bold', y=0.98)

# Enhanced color palettes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors for 4 countries
region_colors = {'Asia': '#FF6B6B', 'Europe': '#4ECDC4', 'Americas': '#45B7D1', 'Africa': '#96CEB4', 'Oceania': '#FFEAA7'}

# Subplot (0,0): Dual-axis demographic trends with bar charts for natural increase
ax1 = plt.subplot(3, 2, 1)
ax1_twin = ax1.twinx()

# Plot birth and death rates with trend lines
for i, country in enumerate(countries):
    cbr_data = demographic_data[country]['cbr']
    cdr_data = demographic_data[country]['cdr']
    
    # Main lines
    ax1.plot(years, cbr_data, color=colors[i], linewidth=2.5, label=f'{country} CBR', linestyle='-')
    ax1.plot(years, cdr_data, color=colors[i], linewidth=2.5, label=f'{country} CDR', linestyle='--', alpha=0.8)
    
    # Add trend lines
    cbr_trend = np.polyfit(years, cbr_data, 1)
    cdr_trend = np.polyfit(years, cdr_data, 1)
    ax1.plot(years, np.poly1d(cbr_trend)(years), color=colors[i], linewidth=1.5, alpha=0.4, linestyle=':')
    ax1.plot(years, np.poly1d(cdr_trend)(years), color=colors[i], linewidth=1.5, alpha=0.4, linestyle=':')

# Add bar charts for natural increase on twin axis
bar_width = 0.8
x_positions = np.arange(0, len(years), 10)  # Every 10 years for clarity
for i, country in enumerate(countries):
    cbr_data = demographic_data[country]['cbr']
    cdr_data = demographic_data[country]['cdr']
    natural_increase = cbr_data - cdr_data
    
    # Sample every 10 years for bar chart
    sampled_increase = natural_increase[::10]
    bar_positions = x_positions + i * bar_width/4 - bar_width/2
    
    ax1_twin.bar(years[::10], sampled_increase, width=bar_width/4, 
                alpha=0.6, color=colors[i], label=f'{country} Natural Increase')

ax1.set_xlabel('Year', fontweight='bold', fontsize=12)
ax1.set_ylabel('Birth/Death Rate (per 1000)', fontweight='bold', fontsize=12)
ax1_twin.set_ylabel('Natural Increase (per 1000)', fontweight='bold', fontsize=12)
ax1.set_title('Demographic Transition: Birth Rates, Death Rates & Natural Increase\n(with Trend Lines)', 
              fontweight='bold', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3, linewidth=0.5)

# Move legend outside to avoid crowding
ax1.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9, framealpha=0.9)
ax1_twin.legend(bbox_to_anchor=(1.15, 0.5), loc='center left', fontsize=9, framealpha=0.9)

# Subplot (0,1): CO2 emissions with better color palette
ax2 = plt.subplot(3, 2, 2)

# Use a qualitative palette designed for multiple categories
colors_co2 = plt.cm.Set3(np.linspace(0, 1, len(co2_countries)))

# Create stacked area chart with better colors
ax2.stackplot(years, *co2_data_smooth, labels=co2_countries, alpha=0.8, colors=colors_co2)

# Overlay temperature projection
temp_projection = 0.3 + 0.025 * np.arange(len(years)) + 0.1 * np.sin(0.1 * np.arange(len(years))) + np.random.normal(0, 0.05, len(years))
confidence_upper = temp_projection + 0.2
confidence_lower = temp_projection - 0.2

ax2_temp = ax2.twinx()
ax2_temp.plot(years, temp_projection, color='darkred', linewidth=4, label='Temperature Change', alpha=0.9)
ax2_temp.fill_between(years, confidence_lower, confidence_upper, alpha=0.25, color='darkred', label='Confidence Band')

ax2.set_xlabel('Year', fontweight='bold', fontsize=12)
ax2.set_ylabel('CO₂ Emissions (Gt)', fontweight='bold', fontsize=12)
ax2_temp.set_ylabel('Temperature Change (°C)', fontweight='bold', fontsize=12, color='darkred')
ax2.set_title('Global CO₂ Emissions & Temperature Projections\n(Top 6 Emitting Countries)', 
              fontweight='bold', fontsize=14, pad=20)
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax2_temp.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Subplot (1,0): FIXED - Slope chart with scatter overlay
ax3 = plt.subplot(3, 2, 3)

if len(merged_data) > 0:
    # Simulate before values (historical data)
    np.random.seed(123)
    before_values = merged_data['Percentage'] - np.random.uniform(15, 30, len(merged_data))
    after_values = merged_data['Percentage'].values
    
    # Assign regions for color coding
    region_mapping = {
        'Rwanda': 'Africa', 'Cuba': 'Americas', 'UAE': 'Asia', 'Mexico': 'Americas',
        'New Zealand': 'Oceania', 'Iceland': 'Europe', 'South Africa': 'Africa',
        'Sweden': 'Europe', 'Finland': 'Europe', 'Norway': 'Europe',
        'Argentina': 'Americas', 'Spain': 'Europe', 'Serbia': 'Europe',
        'Albania': 'Europe', 'Italy': 'Europe'
    }
    
    regions = [region_mapping.get(country, 'Other') for country in merged_data['Country']]
    
    for i, (country, before, after, region, youth_rate) in enumerate(zip(
        merged_data['Country'], before_values, after_values, regions, merged_data['Rate'])):
        
        # Slope lines
        ax3.plot([0, 1], [before, after], color=region_colors.get(region, 'gray'), 
                 alpha=0.7, linewidth=2.5)
        
        # Scatter points with size representing youth unemployment
        point_size = max(80, youth_rate * 12)  # Larger points for better visibility
        ax3.scatter([0, 1], [before, after], s=point_size, 
                   color=region_colors.get(region, 'gray'), alpha=0.8, 
                   edgecolors='white', linewidth=2, zorder=5)
        
        # Labels for top countries only
        if i < 8:
            ax3.text(-0.08, before, country, ha='right', va='center', fontsize=10, fontweight='bold')
            ax3.text(1.08, after, f'{after:.0f}%', ha='left', va='center', fontsize=10, fontweight='bold')

    ax3.set_xlim(-0.4, 1.4)
    ax3.set_ylim(min(before_values) - 5, max(after_values) + 5)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Previous Period', 'Current Period'], fontweight='bold', fontsize=11)
    ax3.set_ylabel('Women in Parliament (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Change in Women\'s Parliamentary Representation\n(Point Size = Youth Unemployment Rate)', 
                  fontweight='bold', fontsize=14, pad=20)
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add region legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=10, label=region) 
                      for region, color in region_colors.items() if region in regions]
    ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5), 
              fontsize=9, framealpha=0.9)

# Subplot (1,1): Time series decomposition with proper title
ax4 = plt.subplot(3, 2, 4)

# Create a 3x1 grid within this subplot for decomposition
gs_inner = fig.add_gridspec(3, 1, left=0.55, right=0.95, top=0.65, bottom=0.38, hspace=0.4)

# Simulate population growth data for major economies
economies = ['USA', 'China', 'India', 'Germany', 'Japan']
colors_econ = plt.cm.Set1(np.linspace(0, 1, len(economies)))

# Original series subplot
ax4a = fig.add_subplot(gs_inner[0])
for i, economy in enumerate(economies):
    if economy == 'China':
        original = 2.8 - 0.04 * np.arange(len(years)) + 0.3 * np.sin(0.2 * np.arange(len(years))) + np.random.normal(0, 0.15, len(years))
    elif economy == 'India':
        original = 2.2 - 0.02 * np.arange(len(years)) + 0.2 * np.sin(0.15 * np.arange(len(years))) + np.random.normal(0, 0.12, len(years))
    elif economy == 'USA':
        original = 1.2 - 0.01 * np.arange(len(years)) + 0.15 * np.sin(0.25 * np.arange(len(years))) + np.random.normal(0, 0.08, len(years))
    elif economy == 'Germany':
        original = 0.8 - 0.015 * np.arange(len(years)) + 0.1 * np.sin(0.3 * np.arange(len(years))) + np.random.normal(0, 0.06, len(years))
    else:  # Japan
        original = 1.0 - 0.025 * np.arange(len(years)) + 0.12 * np.sin(0.22 * np.arange(len(years))) + np.random.normal(0, 0.07, len(years))
    
    ax4a.plot(years, original, color=colors_econ[i], linewidth=2, label=economy, alpha=0.8)

ax4a.set_ylabel('Growth Rate (%)', fontweight='bold', fontsize=10)
ax4a.set_title('Population Growth Rate Decomposition (1961-2021)\nOriginal Series', fontweight='bold', fontsize=12)
ax4a.legend(loc='upper right', fontsize=8)
ax4a.grid(True, alpha=0.3)

# Trend component subplot
ax4b = fig.add_subplot(gs_inner[1])
for i, economy in enumerate(economies):
    if economy == 'China':
        trend = 2.8 - 0.04 * np.arange(len(years))
    elif economy == 'India':
        trend = 2.2 - 0.02 * np.arange(len(years))
    elif economy == 'USA':
        trend = 1.2 - 0.01 * np.arange(len(years))
    elif economy == 'Germany':
        trend = 0.8 - 0.015 * np.arange(len(years))
    else:  # Japan
        trend = 1.0 - 0.025 * np.arange(len(years))
    
    ax4b.plot(years, trend, color=colors_econ[i], linewidth=2.5, alpha=0.8)

ax4b.set_ylabel('Trend (%)', fontweight='bold', fontsize=10)
ax4b.set_title('Trend Component', fontweight='bold', fontsize=11)
ax4b.grid(True, alpha=0.3)

# Cyclical component subplot
ax4c = fig.add_subplot(gs_inner[2])
for i, economy in enumerate(economies):
    if economy == 'China':
        cyclical = 0.3 * np.sin(0.2 * np.arange(len(years)))
    elif economy == 'India':
        cyclical = 0.2 * np.sin(0.15 * np.arange(len(years)))
    elif economy == 'USA':
        cyclical = 0.15 * np.sin(0.25 * np.arange(len(years)))
    elif economy == 'Germany':
        cyclical = 0.1 * np.sin(0.3 * np.arange(len(years)))
    else:  # Japan
        cyclical = 0.12 * np.sin(0.22 * np.arange(len(years)))
    
    ax4c.plot(years, cyclical, color=colors_econ[i], linewidth=2, alpha=0.8)

ax4c.set_xlabel('Year', fontweight='bold', fontsize=10)
ax4c.set_ylabel('Cyclical (%)', fontweight='bold', fontsize=10)
ax4c.set_title('Cyclical Component', fontweight='bold', fontsize=11)
ax4c.grid(True, alpha=0.3)

# Remove the original ax4 since we're using the inner grid
ax4.axis('off')

# Subplot (2,0): Correlation heatmap with scatter overlays
ax5 = plt.subplot(3, 2, 5)

# Create correlation matrix
indicators = ['Birth Rate', 'Death Rate', 'CO₂ Emissions', 'GDP per Capita', 'Life Expectancy']
np.random.seed(123)
correlation_matrix = np.array([
    [1.0, -0.65, 0.45, -0.72, -0.58],
    [-0.65, 1.0, -0.23, 0.48, 0.67],
    [0.45, -0.23, 1.0, 0.34, -0.41],
    [-0.72, 0.48, 0.34, 1.0, 0.76],
    [-0.58, 0.67, -0.41, 0.76, 1.0]
])

# Create heatmap
im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values and scatter plots in lower triangle
for i in range(len(indicators)):
    for j in range(len(indicators)):
        if i == j:
            ax5.text(j, i, '1.00', ha='center', va='center', fontweight='bold', fontsize=11)
        elif i > j:  # Lower triangle - add scatter plots
            # Generate sample data points
            n_points = 30
            x_data = np.random.normal(0, 1, n_points)
            y_data = correlation_matrix[i, j] * x_data + np.random.normal(0, np.sqrt(1 - correlation_matrix[i, j]**2), n_points)
            
            # Normalize to fit in cell
            x_norm = 0.4 * (x_data - x_data.min()) / (x_data.max() - x_data.min()) - 0.2
            y_norm = 0.4 * (y_data - y_data.min()) / (y_data.max() - y_data.min()) - 0.2
            
            ax5.scatter(j + x_norm, i + y_norm, s=8, alpha=0.6, color='darkblue')
        else:  # Upper triangle - correlation values
            ax5.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold', fontsize=11,
                    color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

ax5.set_xticks(range(len(indicators)))
ax5.set_yticks(range(len(indicators)))
ax5.set_xticklabels(indicators, rotation=45, ha='right', fontweight='bold')
ax5.set_yticklabels(indicators, fontweight='bold')
ax5.set_title('Socioeconomic Indicators Correlation Matrix\n(Upper: Correlations, Lower: Scatter Plots)', 
              fontweight='bold', fontsize=14, pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, shrink=0.8, pad=0.02)
cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=11)

# Subplot (2,1): Multi-timeline demographic transition
ax6 = plt.subplot(3, 2, 6)

# Simulate more realistic demographic transition patterns
developed_cbr = 30 - 0.35 * np.arange(len(years)) + 2 * np.sin(0.1 * np.arange(len(years))) + np.random.normal(0, 0.8, len(years))
developed_cdr = 12 - 0.08 * np.arange(len(years)) + np.random.normal(0, 0.4, len(years))
developing_cbr = 45 - 0.25 * np.arange(len(years)) + 3 * np.sin(0.08 * np.arange(len(years))) + np.random.normal(0, 1.2, len(years))
developing_cdr = 18 - 0.15 * np.arange(len(years)) + np.random.normal(0, 0.6, len(years))

# Ensure positive values
developed_cbr = np.maximum(developed_cbr, 8)
developed_cdr = np.maximum(developed_cdr, 6)
developing_cbr = np.maximum(developing_cbr, 15)
developing_cdr = np.maximum(developing_cdr, 8)

# Plot demographic transition lines
ax6.plot(years, developed_cbr, color='#2E86AB', linewidth=3.5, label='Developed - Birth Rate')
ax6.plot(years, developed_cdr, color='#2E86AB', linewidth=3.5, linestyle='--', label='Developed - Death Rate')
ax6.plot(years, developing_cbr, color='#F18F01', linewidth=3.5, label='Developing - Birth Rate')
ax6.plot(years, developing_cdr, color='#F18F01', linewidth=3.5, linestyle='--', label='Developing - Death Rate')

# Add demographic dividend periods as ribbons
developed_dividend = np.maximum(developed_cbr - developed_cdr, 0)
developing_dividend = np.maximum(developing_cbr - developing_cdr, 0)

ax6.fill_between(years, 0, developed_dividend, alpha=0.25, color='#2E86AB', label='Developed Dividend Period')
ax6.fill_between(years, 0, developing_dividend, alpha=0.25, color='#F18F01', label='Developing Dividend Period')

ax6.set_xlabel('Year', fontweight='bold', fontsize=12)
ax6.set_ylabel('Rate (per 1000)', fontweight='bold', fontsize=12)
ax6.set_title('Demographic Transition Patterns & Dividend Periods\n(Developed vs Developing Nations)', 
              fontweight='bold', fontsize=14, pad=20)
ax6.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax6.grid(True, alpha=0.3, linewidth=0.5)

# Final layout adjustments with increased spacing
plt.tight_layout(pad=4.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95)
plt.show()