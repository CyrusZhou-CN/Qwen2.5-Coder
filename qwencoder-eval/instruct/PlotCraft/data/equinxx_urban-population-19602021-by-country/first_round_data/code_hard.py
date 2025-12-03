import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.signal import detrend
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Load data
df_percent = pd.read_csv('urban_percent.csv')
df_total = pd.read_csv('urban_total.csv')

# Data preprocessing
year_cols = [str(year) for year in range(1960, 2021)]
df_percent_clean = df_percent.dropna(subset=['Country Name']).copy()
df_total_clean = df_total.dropna(subset=['Country Name']).copy()

# Remove regional aggregates and keep only countries
exclude_regions = ['World', 'Africa Eastern and Southern', 'Africa Western and Central', 
                   'Arab World', 'Caribbean small states', 'Central Europe and the Baltics',
                   'East Asia & Pacific', 'Euro area', 'Europe & Central Asia',
                   'European Union', 'Fragile and conflict affected situations',
                   'Heavily indebted poor countries', 'High income', 'IBRD only',
                   'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total',
                   'Latin America & Caribbean', 'Least developed countries',
                   'Low & middle income', 'Low income', 'Lower middle income',
                   'Middle East & North Africa', 'Middle income', 'North America',
                   'OECD members', 'Other small states', 'Pacific island small states',
                   'Post-demographic dividend', 'Pre-demographic dividend',
                   'Small states', 'South Asia', 'Sub-Saharan Africa',
                   'Upper middle income']

df_countries = df_percent_clean[~df_percent_clean['Country Name'].isin(exclude_regions)].copy()
df_total_countries = df_total_clean[~df_total_clean['Country Name'].isin(exclude_regions)].copy()

# Get top 5 most populous countries in 2020
top5_countries = df_total_countries.nlargest(5, '2020')['Country Name'].tolist()

# Define color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#F8961E', '#90323D']

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Subplot 1: Line chart with confidence bands for top 5 countries
ax1 = plt.subplot(3, 3, 1)
years = np.array(range(1960, 2021))

for i, country in enumerate(top5_countries):
    country_data = df_countries[df_countries['Country Name'] == country]
    if not country_data.empty:
        values = country_data[year_cols].values.flatten()
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) > 10:
            # Calculate trend and confidence bands
            x_valid = years[valid_mask]
            y_valid = values[valid_mask]
            
            # Polynomial fit for smooth trend
            z = np.polyfit(x_valid, y_valid, 3)
            p = np.poly1d(z)
            trend = p(years)
            
            # Calculate residuals for confidence bands
            residuals = y_valid - p(x_valid)
            std_residual = np.std(residuals)
            
            # Plot line with confidence bands
            ax1.fill_between(years, trend - 1.96*std_residual, trend + 1.96*std_residual, 
                           alpha=0.2, color=colors[i])
            ax1.plot(years, trend, color=colors[i], linewidth=2.5, label=country)
            
            # Mark inflection points (where second derivative changes sign)
            second_deriv = np.gradient(np.gradient(trend))
            inflection_points = np.where(np.diff(np.sign(second_deriv)))[0]
            if len(inflection_points) > 0:
                ax1.scatter(years[inflection_points], trend[inflection_points], 
                          color=colors[i], s=60, zorder=5, edgecolor='white', linewidth=1)

ax1.set_title('Urban Population Trends: Top 5 Most Populous Countries\nwith Confidence Bands and Inflection Points', 
              fontweight='bold', fontsize=12, pad=15)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Urban Population (%)', fontweight='bold')
ax1.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('white')

# Subplot 2: Stacked area chart by continent with trend lines
ax2 = plt.subplot(3, 3, 2)

# Define continent mapping (simplified)
continent_mapping = {
    'China': 'Asia', 'India': 'Asia', 'United States': 'North America',
    'Indonesia': 'Asia', 'Pakistan': 'Asia', 'Brazil': 'South America',
    'Nigeria': 'Africa', 'Bangladesh': 'Asia', 'Russia': 'Europe',
    'Mexico': 'North America', 'Japan': 'Asia', 'Philippines': 'Asia',
    'Ethiopia': 'Africa', 'Vietnam': 'Asia', 'Egypt, Arab Rep.': 'Africa',
    'Germany': 'Europe', 'Turkey': 'Asia', 'Iran, Islamic Rep.': 'Asia',
    'Thailand': 'Asia', 'United Kingdom': 'Europe', 'France': 'Europe',
    'Italy': 'Europe', 'South Africa': 'Africa', 'Tanzania': 'Africa',
    'Myanmar': 'Asia', 'Kenya': 'Africa', 'South Korea': 'Asia',
    'Colombia': 'South America', 'Spain': 'Europe', 'Uganda': 'Africa',
    'Argentina': 'South America', 'Algeria': 'Africa', 'Sudan': 'Africa',
    'Ukraine': 'Europe', 'Iraq': 'Asia', 'Afghanistan': 'Asia',
    'Poland': 'Europe', 'Canada': 'North America', 'Morocco': 'Africa'
}

# Aggregate by continent
continent_data = {}
continents = ['Asia', 'Africa', 'Europe', 'North America', 'South America']

for continent in continents:
    continent_countries = [k for k, v in continent_mapping.items() if v == continent]
    continent_total = np.zeros(len(year_cols))
    
    for country in continent_countries:
        country_data = df_total_countries[df_total_countries['Country Name'] == country]
        if not country_data.empty:
            values = country_data[year_cols].values.flatten()
            valid_values = np.where(np.isnan(values), 0, values)
            continent_total += valid_values
    
    continent_data[continent] = continent_total

# Create stacked area chart
bottom = np.zeros(len(year_cols))
continent_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, (continent, data) in enumerate(continent_data.items()):
    ax2.fill_between(years, bottom, bottom + data, alpha=0.7, 
                    color=continent_colors[i], label=continent)
    
    # Add trend line for each continent
    if np.sum(data) > 0:
        z = np.polyfit(years, data, 2)
        p = np.poly1d(z)
        trend = p(years)
        ax2.plot(years, bottom + trend, color='white', linewidth=2, alpha=0.8)
    
    bottom += data

ax2.set_title('Urban Population Growth by Continent\nwith Polynomial Trend Lines', 
              fontweight='bold', fontsize=12, pad=15)
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Urban Population (millions)', fontweight='bold')
ax2.legend(fontsize=9, loc='upper left', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('white')

# Subplot 3: Dual-axis plot - decadal growth rates and acceleration
ax3 = plt.subplot(3, 3, 3)
ax3_twin = ax3.twinx()

# Calculate global decadal growth rates
decades = [1970, 1980, 1990, 2000, 2010, 2020]
global_urban_pct = []
growth_rates = []
acceleration = []

for year in decades:
    year_str = str(year)
    if year_str in year_cols:
        valid_data = df_countries[year_str].dropna()
        global_avg = valid_data.mean()
        global_urban_pct.append(global_avg)

for i in range(1, len(global_urban_pct)):
    growth_rate = (global_urban_pct[i] - global_urban_pct[i-1]) / 10  # per year
    growth_rates.append(growth_rate)

for i in range(1, len(growth_rates)):
    accel = growth_rates[i] - growth_rates[i-1]
    acceleration.append(accel)

# Bar chart for growth rates
bars = ax3.bar(decades[1:], growth_rates, alpha=0.7, color='#3498db', 
               label='Decadal Growth Rate', width=7)

# Line plot for acceleration
if len(acceleration) > 0:
    ax3_twin.plot(decades[2:], acceleration, color='#e74c3c', marker='o', 
                 linewidth=3, markersize=8, label='Acceleration/Deceleration')
    ax3_twin.axhline(y=0, color='red', linestyle='--', alpha=0.5)

ax3.set_title('Global Urbanization: Growth Rates vs Acceleration', 
              fontweight='bold', fontsize=12, pad=15)
ax3.set_xlabel('Decade', fontweight='bold')
ax3.set_ylabel('Growth Rate (%/year)', fontweight='bold', color='#3498db')
ax3_twin.set_ylabel('Acceleration (%/year²)', fontweight='bold', color='#e74c3c')
ax3.tick_params(axis='y', labelcolor='#3498db')
ax3_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('white')

# Subplot 4: Multiple histograms with KDE for 1960, 1990, 2020
ax4 = plt.subplot(3, 3, 4)

hist_years = ['1960', '1990', '2020']
hist_colors = ['#FF9999', '#66B2FF', '#99FF99']
hist_alphas = [0.6, 0.6, 0.8]

for i, year in enumerate(hist_years):
    data = df_countries[year].dropna()
    
    # Histogram
    ax4.hist(data, bins=20, alpha=hist_alphas[i], color=hist_colors[i], 
            label=f'{year}', density=True, edgecolor='white', linewidth=0.5)
    
    # KDE curve
    if len(data) > 5:
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax4.plot(x_range, kde(x_range), color=hist_colors[i].replace('FF', 'CC'), 
                linewidth=3, alpha=0.9)

ax4.set_title('Distribution Evolution of Urban Percentages\nwith Kernel Density Estimation', 
              fontweight='bold', fontsize=12, pad=15)
ax4.set_xlabel('Urban Population (%)', fontweight='bold')
ax4.set_ylabel('Density', fontweight='bold')
ax4.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3)
ax4.set_facecolor('white')

# Subplot 5: Box plots with violin overlays by decade
ax5 = plt.subplot(3, 3, 5)

decade_data = []
decade_labels = []
for decade in range(1960, 2021, 10):
    decade_end = min(decade + 9, 2020)
    decade_cols = [str(year) for year in range(decade, decade_end + 1) if str(year) in year_cols]
    
    if decade_cols:
        decade_avg = df_countries[decade_cols].mean(axis=1).dropna()
        decade_data.append(decade_avg)
        decade_labels.append(f"{decade}s")

# Create violin plots
parts = ax5.violinplot(decade_data, positions=range(len(decade_labels)), 
                      showmeans=True, showmedians=True)

# Customize violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_alpha(0.6)

# Overlay box plots
bp = ax5.boxplot(decade_data, positions=range(len(decade_labels)), 
                widths=0.3, patch_artist=True, 
                boxprops=dict(facecolor='white', alpha=0.8),
                medianprops=dict(color='red', linewidth=2))

# Label outliers
for i, data in enumerate(decade_data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    
    if len(outliers) > 0:
        outlier_countries = df_countries[df_countries[decade_cols].mean(axis=1).isin(outliers)]['Country Name'].head(2)
        for j, country in enumerate(outlier_countries):
            if j < 2:  # Limit labels to avoid clutter
                ax5.annotate(country, (i, outliers.iloc[j]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)

ax5.set_title('Urban Percentage Distribution by Decade\nBox-Violin Plots with Outlier Labels', 
              fontweight='bold', fontsize=12, pad=15)
ax5.set_xlabel('Decade', fontweight='bold')
ax5.set_ylabel('Urban Population (%)', fontweight='bold')
ax5.set_xticks(range(len(decade_labels)))
ax5.set_xticklabels(decade_labels)
ax5.grid(True, alpha=0.3)
ax5.set_facecolor('white')

# Subplot 6: Heatmap with marginal histograms
ax6 = plt.subplot(3, 3, 6)

# Create change matrix (countries vs decades)
change_matrix = []
country_names = []

for _, country_row in df_countries.head(50).iterrows():  # Limit to 50 countries for visibility
    country_changes = []
    country_name = country_row['Country Name']
    
    for i in range(len(decades)-1):
        start_year = str(decades[i])
        end_year = str(decades[i+1])
        
        if start_year in year_cols and end_year in year_cols:
            start_val = country_row[start_year]
            end_val = country_row[end_year]
            
            if pd.notna(start_val) and pd.notna(end_val):
                change = end_val - start_val
                country_changes.append(change)
            else:
                country_changes.append(np.nan)
        else:
            country_changes.append(np.nan)
    
    if len(country_changes) > 0 and not all(pd.isna(country_changes)):
        change_matrix.append(country_changes)
        country_names.append(country_name[:15])  # Truncate long names

change_matrix = np.array(change_matrix)

# Create heatmap
im = ax6.imshow(change_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
cbar.set_label('Urban % Change', fontweight='bold')

ax6.set_title('Urban Percentage Changes Heatmap\n(Countries vs Decades)', 
              fontweight='bold', fontsize=12, pad=15)
ax6.set_xlabel('Decade Transition', fontweight='bold')
ax6.set_ylabel('Countries (Top 50)', fontweight='bold')

decade_transitions = [f"{decades[i]}-{decades[i+1]}" for i in range(len(decades)-1)]
ax6.set_xticks(range(len(decade_transitions)))
ax6.set_xticklabels(decade_transitions, rotation=45)
ax6.set_yticks(range(0, len(country_names), 5))
ax6.set_yticklabels([country_names[i] for i in range(0, len(country_names), 5)], fontsize=8)
ax6.set_facecolor('white')

# Subplot 7: Slope chart 1960 vs 2020
ax7 = plt.subplot(3, 3, 7)

# Get data for slope chart
slope_data = []
for _, country_row in df_countries.iterrows():
    val_1960 = country_row['1960']
    val_2020 = country_row['2020']
    
    if pd.notna(val_1960) and pd.notna(val_2020):
        change = val_2020 - val_1960
        slope_data.append({
            'country': country_row['Country Name'],
            'val_1960': val_1960,
            'val_2020': val_2020,
            'change': change
        })

slope_df = pd.DataFrame(slope_data)

# Classify trends
slope_df['trend'] = pd.cut(slope_df['change'], 
                          bins=[-np.inf, -10, 10, 30, np.inf],
                          labels=['Declining', 'Stable', 'Growing', 'Rapid Growth'])

# Plot slope lines with color coding
trend_colors = {'Declining': '#e74c3c', 'Stable': '#95a5a6', 
                'Growing': '#3498db', 'Rapid Growth': '#2ecc71'}

for trend, color in trend_colors.items():
    trend_data = slope_df[slope_df['trend'] == trend]
    
    for _, row in trend_data.iterrows():
        ax7.plot([0, 1], [row['val_1960'], row['val_2020']], 
                color=color, alpha=0.6, linewidth=1)

# Add trend legend
for trend, color in trend_colors.items():
    ax7.plot([], [], color=color, linewidth=3, label=trend)

ax7.set_title('Urbanization Slope Chart: 1960 → 2020\nTrend Classification', 
              fontweight='bold', fontsize=12, pad=15)
ax7.set_xlim(-0.1, 1.1)
ax7.set_xticks([0, 1])
ax7.set_xticklabels(['1960', '2020'], fontweight='bold')
ax7.set_ylabel('Urban Population (%)', fontweight='bold')
ax7.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
ax7.grid(True, alpha=0.3)
ax7.set_facecolor('white')

# Subplot 8: Bubble chart with regression
ax8 = plt.subplot(3, 3, 8)

# Prepare bubble chart data
bubble_data = []
for _, country_row in df_countries.iterrows():
    val_1960 = country_row['1960']
    val_2020 = country_row['2020']
    
    # Get total population from total dataset
    total_row = df_total_countries[df_total_countries['Country Name'] == country_row['Country Name']]
    if not total_row.empty:
        pop_2020 = total_row['2020'].iloc[0]
        
        if pd.notna(val_1960) and pd.notna(val_2020) and pd.notna(pop_2020):
            bubble_data.append({
                'x': val_1960,
                'y': val_2020,
                'size': pop_2020,
                'country': country_row['Country Name']
            })

bubble_df = pd.DataFrame(bubble_data)

if len(bubble_df) > 0:
    # Normalize bubble sizes
    bubble_df['size_norm'] = (bubble_df['size'] / bubble_df['size'].max()) * 1000 + 20
    
    # Create scatter plot
    scatter = ax8.scatter(bubble_df['x'], bubble_df['y'], s=bubble_df['size_norm'], 
                         alpha=0.6, c=bubble_df['y'] - bubble_df['x'], 
                         cmap='RdYlGn', edgecolors='white', linewidth=0.5)
    
    # Add regression line
    X = bubble_df['x'].values.reshape(-1, 1)
    y = bubble_df['y'].values
    reg = LinearRegression().fit(X, y)
    x_range = np.linspace(bubble_df['x'].min(), bubble_df['x'].max(), 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    
    ax8.plot(x_range, y_pred, color='red', linewidth=2, label='Regression Line')
    
    # Add prediction intervals (simplified)
    residuals = y - reg.predict(X)
    std_residual = np.std(residuals)
    ax8.fill_between(x_range, y_pred - 1.96*std_residual, y_pred + 1.96*std_residual, 
                    alpha=0.2, color='red', label='95% Prediction Interval')
    
    # Add diagonal line
    ax8.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='No Change Line')

ax8.set_title('Urban % 1960 vs 2020 Bubble Chart\nBubble Size = Population, Color = Change', 
              fontweight='bold', fontsize=12, pad=15)
ax8.set_xlabel('Urban % in 1960', fontweight='bold')
ax8.set_ylabel('Urban % in 2020', fontweight='bold')
ax8.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
ax8.grid(True, alpha=0.3)
ax8.set_facecolor('white')

# Subplot 9: Time series decomposition
ax9 = plt.subplot(3, 3, 9)

# Calculate global urbanization rate
global_urban_rate = []
for year in year_cols:
    valid_data = df_countries[year].dropna()
    if len(valid_data) > 0:
        global_rate = valid_data.mean()
        global_urban_rate.append(global_rate)
    else:
        global_urban_rate.append(np.nan)

global_urban_rate = np.array(global_urban_rate)
valid_mask = ~np.isnan(global_urban_rate)
years_valid = np.array(years)[valid_mask]
rates_valid = global_urban_rate[valid_mask]

if len(rates_valid) > 10:
    # Trend component (polynomial fit)
    z = np.polyfit(years_valid, rates_valid, 3)
    p = np.poly1d(z)
    trend = p(years_valid)
    
    # Residuals
    residuals = rates_valid - trend
    
    # Plot original data
    ax9.plot(years_valid, rates_valid, 'b-', linewidth=2, label='Original', alpha=0.7)
    
    # Plot trend
    ax9.plot(years_valid, trend, 'r-', linewidth=3, label='Trend')
    
    # Plot residuals (scaled and shifted)
    residuals_scaled = residuals * 2 + trend.mean()
    ax9.plot(years_valid, residuals_scaled, 'g-', linewidth=1.5, 
            label='Residuals (scaled)', alpha=0.8)
    
    # Add autocorrelation info as text
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        ax9.text(0.02, 0.98, f'Autocorr: {autocorr:.3f}', 
                transform=ax9.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax9.set_title('Global Urbanization Time Series Decomposition\nTrend, Residuals & Autocorrelation', 
              fontweight='bold', fontsize=12, pad=15)
ax9.set_xlabel('Year', fontweight='bold')
ax9.set_ylabel('Urban Population (%)', fontweight='bold')
ax9.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
ax9.grid(True, alpha=0.3)
ax9.set_facecolor('white')

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()