import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Get all CSV files in the current directory
csv_files = glob.glob("*.csv")
print(f"Found CSV files: {csv_files}")

# Load and combine all datasets
dfs = []
for file in csv_files:
    try:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"Loaded {file} with shape {df.shape}")
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

# Check if we have any data
if len(dfs) == 0:
    print("No CSV files could be loaded. Creating sample data for demonstration.")
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='6H')
    cities = ['Manila', 'Cebu', 'Davao', 'Quezon City', 'Makati', 'Taguig']
    
    sample_data = []
    for date in dates[:1000]:  # Limit to 1000 records for demo
        for city in cities[:3]:  # Use 3 cities
            sample_data.append({
                'datetime': date,
                'city_name': city,
                'main.temp': np.random.normal(27, 3),
                'main.temp_min': np.random.normal(25, 2),
                'main.temp_max': np.random.normal(29, 2),
                'main.humidity': np.random.normal(75, 10),
                'main.pressure': np.random.normal(1013, 5),
                'wind.speed': np.random.exponential(3),
                'wind.deg': np.random.uniform(0, 360),
                'wind.gust': np.random.exponential(4),
                'clouds.all': np.random.uniform(0, 100),
                'rain.1h': np.random.exponential(0.5) if np.random.random() > 0.7 else 0,
                'weather.main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Thunderstorm'], p=[0.3, 0.4, 0.2, 0.1])
            })
    
    df = pd.DataFrame(sample_data)
else:
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

# Data preprocessing
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['quarter'] = df['datetime'].dt.quarter
df['temp_amplitude'] = df['main.temp_max'] - df['main.temp_min']

# Fill missing values
df['rain.1h'] = df['rain.1h'].fillna(0)
df['wind.gust'] = df['wind.gust'].fillna(df['wind.speed'])

# Select major cities with sufficient data
city_counts = df['city_name'].value_counts()
major_cities = city_counts.head(10).index.tolist()
df_major = df[df['city_name'].isin(major_cities)].copy()

print(f"Major cities: {major_cities}")
print(f"Data shape after filtering: {df_major.shape}")

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Define color palettes
temp_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
season_colors = ['#4A90E2', '#7ED321', '#F5A623', '#D0021B']
weather_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Subplot 1: Monthly temperature trends with precipitation overlay
ax1 = plt.subplot(3, 3, 1)
monthly_temp = df_major.groupby(['month', 'city_name'])['main.temp'].agg(['mean', 'std']).reset_index()
monthly_precip = df_major.groupby('month')['rain.1h'].mean()

# Temperature trends with error bands
cities_to_plot = major_cities[:min(4, len(major_cities))]
for i, city in enumerate(cities_to_plot):
    city_data = monthly_temp[monthly_temp['city_name'] == city]
    if len(city_data) > 0:
        ax1.plot(city_data['month'], city_data['mean'], 
                 color=temp_colors[i % len(temp_colors)], linewidth=2.5, 
                 label=city, marker='o', markersize=6)
        ax1.fill_between(city_data['month'], 
                         city_data['mean'] - city_data['std'],
                         city_data['mean'] + city_data['std'],
                         color=temp_colors[i % len(temp_colors)], alpha=0.2)

# Precipitation overlay
ax1_twin = ax1.twinx()
if len(monthly_precip) > 0:
    bars = ax1_twin.bar(monthly_precip.index, monthly_precip.values, 
                        alpha=0.3, color='steelblue', width=0.6)
    ax1_twin.set_ylabel('Precipitation (mm/h)', fontweight='bold', fontsize=11)
    ax1_twin.set_ylim(0, monthly_precip.max() * 1.2)

ax1.set_title('Monthly Temperature Trends with Precipitation Patterns', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Month', fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fancybox=True)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 12.5)

# Subplot 2: Seasonal humidity distribution with violin and box plots
ax2 = plt.subplot(3, 3, 2)
humidity_data = []
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

for q in [1, 2, 3, 4]:
    q_data = df_major[df_major['quarter'] == q]['main.humidity'].dropna()
    if len(q_data) > 0:
        humidity_data.append(q_data)
    else:
        humidity_data.append(pd.Series([75]))  # Default value

if len(humidity_data) > 0:
    # Violin plots
    try:
        parts = ax2.violinplot(humidity_data, positions=range(1, 5), widths=0.6, showmeans=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(season_colors[i])
            pc.set_alpha(0.7)
    except:
        pass

    # Box plots overlay
    try:
        bp = ax2.boxplot(humidity_data, positions=range(1, 5), widths=0.3, 
                         patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor('white')
            patch.set_alpha(0.8)
    except:
        pass

ax2.set_title('Seasonal Humidity Distribution Patterns', fontweight='bold', fontsize=14, pad=20)
ax2.set_xlabel('Quarter', fontweight='bold')
ax2.set_ylabel('Humidity (%)', fontweight='bold')
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(quarters)
ax2.grid(True, alpha=0.3)

# Subplot 3: Wind speed and direction polar scatter plot
ax3 = plt.subplot(3, 3, 3, projection='polar')
wind_data = df_major.dropna(subset=['wind.deg', 'wind.speed'])
wind_data = wind_data[wind_data['wind.speed'] > 0]

if len(wind_data) > 0:
    # Convert wind direction to radians
    theta = np.radians(wind_data['wind.deg'])
    r = wind_data['wind.speed']
    quarters_wind = wind_data['quarter']

    # Scatter plot with seasonal colors
    for q in [1, 2, 3, 4]:
        mask = quarters_wind == q
        if mask.sum() > 0:
            ax3.scatter(theta[mask], r[mask], c=season_colors[q-1], 
                       alpha=0.6, s=30, label=f'Q{q}')

    ax3.set_ylim(0, wind_data['wind.speed'].quantile(0.95))

ax3.set_title('Wind Speed & Direction by Season', fontweight='bold', fontsize=14, pad=30)
ax3.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))

# Subplot 4: Daily temperature amplitude evolution
ax4 = plt.subplot(3, 3, 4)
daily_amplitude = df_major.groupby(['month', 'day'])['temp_amplitude'].mean().reset_index()
if len(daily_amplitude) > 0:
    daily_amplitude['date_num'] = daily_amplitude['month'] * 30 + daily_amplitude['day']

    # Area chart with moving average
    ax4.fill_between(daily_amplitude['date_num'], daily_amplitude['temp_amplitude'], 
                     alpha=0.4, color='coral')
    ax4.plot(daily_amplitude['date_num'], daily_amplitude['temp_amplitude'], 
             color='darkred', linewidth=1.5)

    # Moving average overlay
    window = min(30, len(daily_amplitude))
    if len(daily_amplitude) >= window:
        moving_avg = daily_amplitude['temp_amplitude'].rolling(window=window, center=True).mean()
        ax4.plot(daily_amplitude['date_num'], moving_avg, 
                 color='navy', linewidth=3, label=f'{window}-day Moving Average')

ax4.set_title('Daily Temperature Amplitude Evolution', fontweight='bold', fontsize=14, pad=20)
ax4.set_xlabel('Time (Month-Day)', fontweight='bold')
ax4.set_ylabel('Temperature Amplitude (°C)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Pressure system changes with storm events
ax5 = plt.subplot(3, 3, 5)
pressure_data = df_major.groupby('month')['main.pressure'].agg(['mean', 'min', 'max']).reset_index()

if len(pressure_data) > 0:
    # Line chart with filled areas
    ax5.plot(pressure_data['month'], pressure_data['mean'], 
             color='darkblue', linewidth=3, marker='o', markersize=8, label='Mean Pressure')
    ax5.fill_between(pressure_data['month'], pressure_data['min'], pressure_data['max'], 
                     alpha=0.3, color='lightblue', label='Pressure Range')

    # Mark extreme events
    if len(pressure_data) > 0:
        extreme_low = pressure_data[pressure_data['min'] < pressure_data['min'].quantile(0.1)]
        if len(extreme_low) > 0:
            ax5.scatter(extreme_low['month'], extreme_low['min'], 
                       color='red', s=100, marker='v', label='Low Pressure Events', zorder=5)

ax5.set_title('Pressure System Changes with Extreme Events', fontweight='bold', fontsize=14, pad=20)
ax5.set_xlabel('Month', fontweight='bold')
ax5.set_ylabel('Pressure (hPa)', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Weather condition frequency heatmap
ax6 = plt.subplot(3, 3, 6)
try:
    weather_freq = pd.crosstab(df_major['month'], df_major['weather.main'], normalize='index') * 100
    weather_freq = weather_freq.fillna(0)

    if weather_freq.shape[0] > 0 and weather_freq.shape[1] > 0:
        # Heatmap
        sns.heatmap(weather_freq, annot=True, fmt='.1f', cmap='YlOrRd', 
                    ax=ax6, cbar_kws={'label': 'Frequency (%)'})
except:
    # Fallback: simple bar chart
    weather_counts = df_major['weather.main'].value_counts()
    ax6.bar(range(len(weather_counts)), weather_counts.values)
    ax6.set_xticks(range(len(weather_counts)))
    ax6.set_xticklabels(weather_counts.index, rotation=45)

ax6.set_title('Weather Condition Frequency by Month', fontweight='bold', fontsize=14, pad=20)
ax6.set_xlabel('Weather Condition', fontweight='bold')
ax6.set_ylabel('Month', fontweight='bold')

# Subplot 7: City-wise temperature anomaly patterns
ax7 = plt.subplot(3, 3, 7)
if len(df_major) > 0:
    annual_mean = df_major.groupby('city_name')['main.temp'].mean()
    monthly_city_temp = df_major.groupby(['city_name', 'month'])['main.temp'].mean().reset_index()
    
    if len(monthly_city_temp) > 0:
        monthly_city_temp['anomaly'] = monthly_city_temp.apply(
            lambda x: x['main.temp'] - annual_mean.get(x['city_name'], x['main.temp']), axis=1)

        # Diverging bar chart for selected cities
        cities_subset = major_cities[:min(6, len(major_cities))]
        for i, city in enumerate(cities_subset):
            city_data = monthly_city_temp[monthly_city_temp['city_name'] == city]
            if len(city_data) > 0:
                colors = ['red' if x < 0 else 'blue' for x in city_data['anomaly']]
                ax7.barh([i] * len(city_data), city_data['anomaly'], 
                         left=city_data['month'], height=0.1, color=colors, alpha=0.7)

        ax7.set_yticks(range(len(cities_subset)))
        ax7.set_yticklabels(cities_subset)

ax7.set_title('City-wise Temperature Anomaly Patterns', fontweight='bold', fontsize=14, pad=20)
ax7.set_xlabel('Temperature Anomaly (°C)', fontweight='bold')
ax7.set_ylabel('Cities', fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax7.grid(True, alpha=0.3)

# Subplot 8: Precipitation intensity distribution
ax8 = plt.subplot(3, 3, 8)
rain_data = df_major[df_major['rain.1h'] > 0]['rain.1h']
quarters_rain = df_major[df_major['rain.1h'] > 0]['quarter']

if len(rain_data) > 0:
    # Histogram with KDE overlay for each season
    for q in [1, 2, 3, 4]:
        season_rain = rain_data[quarters_rain == q]
        if len(season_rain) > 10:
            ax8.hist(season_rain, bins=20, alpha=0.5, density=True, 
                    color=season_colors[q-1], label=f'Q{q}')

    # Mark extreme events
    if len(rain_data) > 0:
        extreme_rain = rain_data.quantile(0.95)
        ax8.axvline(x=extreme_rain, color='red', linestyle='--', linewidth=2, 
                   label=f'95th percentile: {extreme_rain:.2f}mm')

ax8.set_title('Precipitation Intensity Distribution by Season', fontweight='bold', fontsize=14, pad=20)
ax8.set_xlabel('Precipitation (mm/h)', fontweight='bold')
ax8.set_ylabel('Density', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Subplot 9: Multi-variable correlation matrix and weather index
ax9 = plt.subplot(3, 3, 9)

# Create correlation matrix
weather_vars = ['main.temp', 'main.humidity', 'main.pressure', 'wind.speed', 'clouds.all']
available_vars = [var for var in weather_vars if var in df_major.columns]

if len(available_vars) >= 2:
    corr_data = df_major[available_vars].corr()

    # Heatmap
    im = ax9.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(available_vars)))
    ax9.set_yticks(range(len(available_vars)))
    
    var_labels = [var.split('.')[-1].title() for var in available_vars]
    ax9.set_xticklabels(var_labels, rotation=45)
    ax9.set_yticklabels(var_labels)

    # Add correlation values
    for i in range(len(available_vars)):
        for j in range(len(available_vars)):
            text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontweight='bold')

ax9.set_title('Weather Variables Correlation Matrix', fontweight='bold', fontsize=14, pad=20)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('Comprehensive Weather Pattern Analysis - Philippine Major Cities 2024', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('weather_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()