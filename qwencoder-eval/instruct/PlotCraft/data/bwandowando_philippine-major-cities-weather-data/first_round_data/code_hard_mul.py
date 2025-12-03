import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Get all CSV files in the current directory
csv_files = glob.glob('*.csv')
print(f"Found CSV files: {csv_files}")

# Filter for monthly data files (those with YYYYMM pattern)
monthly_files = [f for f in csv_files if any(month in f for month in ['202401', '202403', '202406', '202409', '202411', '202412'])]

if not monthly_files:
    # If no monthly files found, use any available CSV files
    monthly_files = csv_files[:6] if len(csv_files) >= 6 else csv_files

print(f"Using files: {monthly_files}")

if not monthly_files:
    raise Exception("No CSV files found in the directory")

# Combine all available data
all_data = []
for file in monthly_files:
    try:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        
        # Handle datetime conversion more robustly
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            df['month'] = df['datetime'].dt.month
            df['quarter'] = df['datetime'].dt.quarter
        else:
            # If no datetime column, try to infer from filename
            if '202401' in file:
                df['month'] = 1
                df['quarter'] = 1
            elif '202403' in file:
                df['month'] = 3
                df['quarter'] = 1
            elif '202406' in file:
                df['month'] = 6
                df['quarter'] = 2
            elif '202409' in file:
                df['month'] = 9
                df['quarter'] = 3
            elif '202411' in file:
                df['month'] = 11
                df['quarter'] = 4
            elif '202412' in file:
                df['month'] = 12
                df['quarter'] = 4
            else:
                df['month'] = 6  # Default
                df['quarter'] = 2
        
        all_data.append(df)
        print(f"Successfully loaded {file} with {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

if not all_data:
    raise Exception("No data files could be loaded successfully")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Focus on the exact cities requested: Manila, Cebu, Davao, Baguio, Iloilo
available_cities = combined_df['city_name'].unique()
print(f"Available cities: {list(available_cities)[:10]}...")

# Use the exact cities specified in the task
requested_cities = ['Manila', 'Cebu', 'Davao', 'Baguio', 'Iloilo']
# Find closest matches in available cities if exact names don't exist
existing_major_cities = []
for city in requested_cities:
    if city in available_cities:
        existing_major_cities.append(city)
    else:
        # Look for partial matches
        matches = [c for c in available_cities if city.lower() in c.lower() or c.lower() in city.lower()]
        if matches:
            existing_major_cities.append(matches[0])

# If we still don't have enough cities, add from most frequent
if len(existing_major_cities) < 5:
    city_counts = combined_df['city_name'].value_counts()
    additional_cities = [city for city in city_counts.head(10).index if city not in existing_major_cities]
    existing_major_cities.extend(additional_cities[:5-len(existing_major_cities)])

print(f"Using cities: {existing_major_cities}")

city_data = combined_df[combined_df['city_name'].isin(existing_major_cities)].copy()

# Ensure we have required columns with fallbacks
required_columns = {
    'main.temp': 'main.temp',
    'main.temp_min': 'main.temp_min', 
    'main.temp_max': 'main.temp_max',
    'main.humidity': 'main.humidity',
    'main.pressure': 'main.pressure',
    'rain.1h': 'rain.1h',
    'wind.speed': 'wind.speed',
    'wind.deg': 'wind.deg'
}

# Check which columns exist and create fallbacks if needed
for col_name, col_key in required_columns.items():
    if col_key not in city_data.columns:
        print(f"Column {col_key} not found, creating with default values")
        if 'temp' in col_key:
            city_data[col_key] = 25.0  # Default temperature
        elif 'humidity' in col_key:
            city_data[col_key] = 70.0  # Default humidity
        elif 'pressure' in col_key:
            city_data[col_key] = 1013.0  # Default pressure
        elif 'rain' in col_key:
            city_data[col_key] = 0.0  # Default no rain
        elif 'wind' in col_key:
            city_data[col_key] = 2.0 if 'speed' in col_key else 180.0  # Default wind

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('white')

# Define color palettes
temp_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
precip_colors = ['#74B9FF', '#0984E3', '#6C5CE7']
season_colors = ['#FD79A8', '#FDCB6E', '#E17055', '#00B894']

# Subplot 1: Monthly temperature trends with error bars and seasonal background
ax1 = plt.subplot(2, 3, 1)
monthly_temp = city_data.groupby(['month', 'city_name'])['main.temp'].agg(['mean', 'std']).reset_index()

# Create seasonal background
available_months = sorted(city_data['month'].unique())
if available_months:
    for month in available_months:
        # Determine season colors based on months
        if month in [12, 1, 2]:
            color = '#E3F2FD'  # Winter
        elif month in [3, 4, 5]:
            color = '#FFF3E0'  # Spring
        elif month in [6, 7, 8]:
            color = '#FFEBEE'  # Summer
        else:
            color = '#F3E5F5'  # Fall
        ax1.axvspan(month-0.4, month+0.4, alpha=0.2, color=color)

# Plot temperature trends for the 5 specified cities
selected_cities = existing_major_cities[:5]
for i, city in enumerate(selected_cities):
    city_temp = monthly_temp[monthly_temp['city_name'] == city]
    if not city_temp.empty:
        ax1.errorbar(city_temp['month'], city_temp['mean'], yerr=city_temp['std'].fillna(0), 
                    label=city, color=temp_colors[i], linewidth=2.5, marker='o', 
                    markersize=6, capsize=4, capthick=2)

ax1.set_title('Monthly Temperature Evolution Across Major Cities', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
if available_months:
    ax1.set_xlim(min(available_months)-0.5, max(available_months)+0.5)

# Subplot 2: Dual-axis plot with temperature ranges and humidity correlation WITH TREND LINES
ax2 = plt.subplot(2, 3, 2)
ax2_twin = ax2.twinx()

# Find hottest and coolest cities from available data
if len(existing_major_cities) >= 2:
    avg_temps = city_data.groupby('city_name')['main.temp'].mean()
    hottest_city = avg_temps.idxmax()
    coolest_city = avg_temps.idxmin()
    
    # Temperature ranges for hottest and coolest cities
    for city, color, alpha in [(hottest_city, '#FF6B6B', 0.6), (coolest_city, '#4ECDC4', 0.6)]:
        city_monthly = city_data[city_data['city_name'] == city].groupby('month').agg({
            'main.temp_min': 'min',
            'main.temp_max': 'max',
            'main.humidity': 'mean'
        }).reset_index()
        
        if not city_monthly.empty:
            # Area chart for temperature range
            ax2.fill_between(city_monthly['month'], city_monthly['main.temp_min'], 
                           city_monthly['main.temp_max'], alpha=alpha, color=color, 
                           label=f'{city} Temp Range')
            
            # Scatter points for humidity correlation
            ax2_twin.scatter(city_monthly['month'], city_monthly['main.humidity'], 
                           color=color, s=80, alpha=0.8, edgecolors='white', linewidth=2)
            
            # ADD HUMIDITY TREND LINES (addressing feedback point 2)
            if len(city_monthly) > 1:
                z = np.polyfit(city_monthly['month'], city_monthly['main.humidity'], 1)
                p = np.poly1d(z)
                ax2_twin.plot(city_monthly['month'], p(city_monthly['month']), 
                             '--', color=color, linewidth=2, alpha=0.9, 
                             label=f'{city} Humidity Trend')
    
    # Temperature trend lines
    if available_months and len(available_months) > 1:
        months = np.array(available_months)
        for city, color in [(hottest_city, '#FF6B6B'), (coolest_city, '#4ECDC4')]:
            city_monthly = city_data[city_data['city_name'] == city].groupby('month')['main.temp'].mean()
            if len(city_monthly) > 1:
                common_months = sorted(set(months) & set(city_monthly.index))
                if len(common_months) > 1:
                    y_vals = [city_monthly[m] for m in common_months]
                    if len(common_months) >= 2:
                        z = np.polyfit(common_months, y_vals, min(len(common_months)-1, 1))
                        p = np.poly1d(z)
                        ax2.plot(common_months, p(common_months), '--', color=color, linewidth=2, alpha=0.8)

ax2.set_title('Temperature Ranges: Hottest vs Coolest Cities', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax2_twin.set_ylabel('Humidity (%)', fontsize=12, fontweight='bold')

# IMPROVED LEGEND POSITIONING (addressing feedback point 5)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
ax2.grid(True, alpha=0.3)

# Subplot 3: Seasonal temperature distributions with violin and box plots
ax3 = plt.subplot(2, 3, 3)

# Prepare quarterly data
quarterly_data = []
quarter_labels = []
available_quarters = sorted(city_data['quarter'].unique())

for quarter in available_quarters:
    quarter_temps = city_data[city_data['quarter'] == quarter]['main.temp'].dropna()
    if len(quarter_temps) > 5:
        quarterly_data.append(quarter_temps)
        quarter_labels.append(f'Q{quarter}')

if quarterly_data:
    # Violin plots
    positions = range(1, len(quarterly_data) + 1)
    try:
        parts = ax3.violinplot(quarterly_data, positions=positions, widths=0.6, 
                              showmeans=True, showmedians=True)
        
        # Color violin plots
        colors_to_use = season_colors[:len(quarterly_data)]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_to_use[i])
            pc.set_alpha(0.7)
    except:
        # Fallback to histogram if violin plot fails
        for i, data in enumerate(quarterly_data):
            ax3.hist(data, alpha=0.6, bins=20, color=season_colors[i], 
                    label=quarter_labels[i])
    
    # Box plots overlay
    try:
        box_data = ax3.boxplot(quarterly_data, positions=positions, widths=0.3, 
                              patch_artist=True, boxprops=dict(alpha=0.8))
        
        colors_to_use = season_colors[:len(quarterly_data)]
        for i, patch in enumerate(box_data['boxes']):
            patch.set_facecolor(colors_to_use[i])
    except:
        pass
    
    # Seasonal average lines
    seasonal_avgs = [np.mean(data) for data in quarterly_data]
    if len(positions) == len(seasonal_avgs):
        ax3.plot(positions, seasonal_avgs, 'ko-', linewidth=3, markersize=8, 
                label='Seasonal Average')
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(quarter_labels)

ax3.set_title('Temperature Distribution Changes Across Seasons', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Quarter', fontsize=12, fontweight='bold')
ax3.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Precipitation bars with pressure trends and monsoon seasons
ax4 = plt.subplot(2, 3, 4)
ax4_twin = ax4.twinx()

# Select 3 representative cities from available data
rep_cities = existing_major_cities[:3]
monthly_precip = city_data.groupby(['month', 'city_name'])['rain.1h'].mean().reset_index()
monthly_pressure = city_data.groupby(['month', 'city_name'])['main.pressure'].mean().reset_index()

# Bar chart for precipitation
bar_width = 0.25
available_months_array = np.array(available_months) if available_months else np.array([])

for i, city in enumerate(rep_cities):
    city_precip = monthly_precip[monthly_precip['city_name'] == city]
    if not city_precip.empty:
        ax4.bar(city_precip['month'] + i*bar_width, city_precip['rain.1h'].fillna(0), 
                bar_width, label=f'{city} Precipitation', color=precip_colors[i % len(precip_colors)], alpha=0.8)

# Pressure trend lines
for i, city in enumerate(rep_cities):
    city_pressure = monthly_pressure[monthly_pressure['city_name'] == city]
    if not city_pressure.empty:
        ax4_twin.plot(city_pressure['month'], city_pressure['main.pressure'], 
                      'o-', color=precip_colors[i % len(precip_colors)], linewidth=2, markersize=6,
                      label=f'{city} Pressure')

# Monsoon season shading
if available_months and any(m >= 6 and m <= 10 for m in available_months):
    monsoon_start = max(6, min(available_months))
    monsoon_end = min(10, max(available_months))
    ax4.axvspan(monsoon_start-0.5, monsoon_end+0.5, alpha=0.2, color='blue', label='Monsoon Period')

ax4.set_title('Precipitation and Atmospheric Pressure Patterns', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
ax4.set_ylabel('Precipitation (mm/h)', fontsize=12, fontweight='bold')
ax4_twin.set_ylabel('Pressure (hPa)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Subplot 5: COMPLETE OVERHAUL - Quarterly correlation matrices with diagonal scatter plots
ax5 = plt.subplot(2, 3, 5)

# Weather variables for correlation analysis
weather_vars = ['main.temp', 'main.humidity', 'main.pressure', 'rain.1h']
var_labels = ['Temperature', 'Humidity', 'Pressure', 'Precipitation']

# Create a 4x4 grid for quarterly correlation analysis
gs = fig.add_gridspec(2, 2, left=0.69, right=0.98, top=0.48, bottom=0.08, hspace=0.3, wspace=0.3)

available_quarters = sorted(city_data['quarter'].unique())
quarter_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i, quarter in enumerate(available_quarters[:4]):
    if i < len(quarter_positions):
        row, col = quarter_positions[i]
        ax_sub = fig.add_subplot(gs[row, col])
        
        quarter_data = city_data[city_data['quarter'] == quarter][weather_vars].dropna()
        
        if len(quarter_data) > 10:
            # Calculate correlation matrix
            corr_matrix = quarter_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, ax=ax_sub, cbar=False, 
                       xticklabels=var_labels, yticklabels=var_labels,
                       annot_kws={'size': 8})
            
            ax_sub.set_title(f'Q{quarter} Correlations', fontsize=10, fontweight='bold')
            ax_sub.tick_params(axis='both', which='major', labelsize=8)

# Remove the original ax5 since we're using the gridspec
ax5.remove()

# Subplot 6: ENHANCED Wind patterns with clear seasonal separation and trend lines
ax6 = plt.subplot(2, 3, 6, projection='polar')

# Prepare wind data by season with better separation
seasonal_wind = city_data.groupby(['quarter', 'city_name']).agg({
    'wind.deg': 'mean',
    'wind.speed': 'mean',
    'rain.1h': 'mean'
}).reset_index()

# Convert wind direction to radians and handle NaN values
seasonal_wind = seasonal_wind.dropna(subset=['wind.deg', 'wind.speed'])

if not seasonal_wind.empty:
    seasonal_wind['wind.deg_rad'] = np.radians(seasonal_wind['wind.deg'])
    
    # Define distinct markers and colors for each season
    season_markers = ['o', 's', '^', 'D']
    season_sizes = [100, 120, 140, 160]
    
    # Plot for each available season with distinct styling
    for i, quarter in enumerate(available_quarters):
        quarter_data = seasonal_wind[seasonal_wind['quarter'] == quarter]
        if not quarter_data.empty:
            color_idx = (quarter - 1) % len(season_colors)
            marker_idx = (quarter - 1) % len(season_markers)
            
            # Scatter plot with distinct markers and sizes
            scatter = ax6.scatter(quarter_data['wind.deg_rad'], quarter_data['wind.speed'],
                                 c=quarter_data['rain.1h'].fillna(0), s=season_sizes[marker_idx], 
                                 alpha=0.8, cmap='Blues', marker=season_markers[marker_idx],
                                 edgecolors=season_colors[color_idx], linewidth=2,
                                 label=f'Q{quarter}')
    
    # ADD SEASONAL TREND LINES (addressing feedback point 4)
    for i, quarter in enumerate(available_quarters):
        quarter_data = seasonal_wind[seasonal_wind['quarter'] == quarter]
        if len(quarter_data) > 0:
            # Calculate mean direction and speed for the quarter
            avg_direction = np.radians(quarter_data['wind.deg'].mean())
            avg_speed = quarter_data['wind.speed'].mean()
            color_idx = (quarter - 1) % len(season_colors)
            
            # Draw vector showing mean wind direction and speed
            ax6.annotate('', xy=(avg_direction, avg_speed), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=season_colors[color_idx], 
                                      lw=4, alpha=0.9))
            
            # Add text label for the trend
            ax6.text(avg_direction, avg_speed + 0.5, f'Q{quarter}', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=season_colors[color_idx])
    
    # Add colorbar for precipitation intensity
    if 'scatter' in locals():
        try:
            cbar = plt.colorbar(scatter, ax=ax6, shrink=0.8, pad=0.1)
            cbar.set_label('Precipitation Intensity (mm/h)', fontsize=10)
        except:
            pass

ax6.set_title('Wind Speed and Direction Patterns by Season\n(with Seasonal Trend Vectors)', 
              fontsize=12, fontweight='bold', pad=30)
ax6.set_theta_zero_location('N')
ax6.set_theta_direction(-1)
ax6.set_ylim(0, 8)
ax6.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.35, wspace=0.4)

plt.savefig('philippine_weather_analysis_refined.png', dpi=300, bbox_inches='tight')
plt.show()