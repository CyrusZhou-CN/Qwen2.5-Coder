import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load data from available files - simplified approach
try:
    df1 = pd.read_csv('PRSA_Data_Aotizhongxin_20130301-20170228.csv')
    df2 = pd.read_csv('PRSA_Data_Dongsi_20130301-20170228.csv')
    df3 = pd.read_csv('PRSA_Data_Wanliu_20130301-20170228.csv')
    df4 = pd.read_csv('PRSA_Data_Tiantan_20130301-20170228.csv')
    
    # Combine data
    all_data = pd.concat([df1, df2, df3, df4], ignore_index=True)
except:
    # Fallback to single file if others not available
    all_data = pd.read_csv('PRSA_Data_Aotizhongxin_20130301-20170228.csv')

# Data preprocessing
all_data['datetime'] = pd.to_datetime(all_data[['year', 'month', 'day', 'hour']])
all_data = all_data.dropna(subset=['PM2.5', 'PM10', 'TEMP'])

# Create figure
fig = plt.figure(figsize=(18, 14), facecolor='white')

# Row 1: Daily Pattern Analysis

# Subplot 1: Hourly PM2.5 patterns for stations with error bands
ax1 = plt.subplot(3, 3, 1)
available_stations = all_data['station'].unique()[:3]
colors = ['#2E86AB', '#A23B72', '#F18F01']

for i, station in enumerate(available_stations):
    station_data = all_data[all_data['station'] == station]
    hourly_stats = station_data.groupby('hour')['PM2.5'].agg(['mean', 'std']).reset_index()
    
    ax1.plot(hourly_stats['hour'], hourly_stats['mean'], 
             color=colors[i], linewidth=2, label=station, marker='o', markersize=3)
    ax1.fill_between(hourly_stats['hour'], 
                     hourly_stats['mean'] - hourly_stats['std'],
                     hourly_stats['mean'] + hourly_stats['std'],
                     color=colors[i], alpha=0.2)

ax1.set_title('Hourly PM2.5 Patterns Across Stations', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('PM2.5 (μg/m³)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Box plots for PM10 by hour (simplified)
ax2 = plt.subplot(3, 3, 2)
hourly_pm10_data = []
hour_labels = []

for hour in range(0, 24, 4):  # Every 4 hours to reduce computation
    hour_data = all_data[all_data['hour'] == hour]['PM10'].dropna()
    if len(hour_data) > 10:
        hourly_pm10_data.append(hour_data.values[:500])  # Limit data size
        hour_labels.append(hour)

bp = ax2.boxplot(hourly_pm10_data, positions=hour_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#4ECDC4')
    patch.set_alpha(0.7)

ax2.set_title('PM10 Distribution by Hour', fontsize=12, fontweight='bold')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('PM10 (μg/m³)')
ax2.grid(True, alpha=0.3)

# Subplot 3: Temperature-PM2.5 correlation
ax3 = plt.subplot(3, 3, 3)
sample_data = all_data.sample(n=min(1000, len(all_data))).dropna(subset=['TEMP', 'PM2.5'])

scatter = ax3.scatter(sample_data['TEMP'], sample_data['PM2.5'], 
                     alpha=0.5, s=15, c='#FF6B6B')

# Add trend line
z = np.polyfit(sample_data['TEMP'], sample_data['PM2.5'], 1)
p = np.poly1d(z)
ax3.plot(sample_data['TEMP'].sort_values(), p(sample_data['TEMP'].sort_values()), 
         "r--", alpha=0.8, linewidth=2)

ax3.set_title('Temperature vs PM2.5 Correlation', fontsize=12, fontweight='bold')
ax3.set_xlabel('Temperature (°C)')
ax3.set_ylabel('PM2.5 (μg/m³)')
ax3.grid(True, alpha=0.3)

# Row 2: Seasonal Variation Analysis

# Subplot 4: Monthly PM2.5 trends
ax4 = plt.subplot(3, 3, 4)
stations_to_plot = all_data['station'].unique()[:4]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, station in enumerate(stations_to_plot):
    station_data = all_data[all_data['station'] == station]
    monthly_pm25 = station_data.groupby('month')['PM2.5'].mean()
    
    ax4.plot(monthly_pm25.index, monthly_pm25.values, 
             color=colors[i], linewidth=2, label=station, marker='o', markersize=4)
    ax4.fill_between(monthly_pm25.index, 0, monthly_pm25.values, 
                     color=colors[i], alpha=0.2)

ax4.set_title('Monthly PM2.5 Trends by Station', fontsize=12, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Average PM2.5 (μg/m³)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Monthly pollutant composition
ax5 = plt.subplot(3, 3, 5)
ax5_twin = ax5.twinx()

monthly_data = all_data.groupby('month')[['PM2.5', 'PM10', 'SO2', 'NO2', 'TEMP']].mean()

# Stacked bar chart (scaled for visibility)
bottom = np.zeros(12)
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2']
colors_poll = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, pollutant in enumerate(pollutants):
    values = monthly_data[pollutant].values
    if pollutant in ['PM2.5', 'PM10']:
        values = values / 3  # Scale for visibility
    ax5.bar(range(1, 13), values, bottom=bottom, 
           color=colors_poll[i], label=pollutant, alpha=0.8)
    bottom += values

# Temperature line
ax5_twin.plot(range(1, 13), monthly_data['TEMP'].values, 
              color='red', linewidth=2, marker='s', markersize=4, label='Temperature')

ax5.set_title('Monthly Pollutants + Temperature', fontsize=12, fontweight='bold')
ax5.set_xlabel('Month')
ax5.set_ylabel('Pollutant Concentration (scaled)')
ax5_twin.set_ylabel('Temperature (°C)', color='red')
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')

# Subplot 6: Rainfall vs AQI
ax6 = plt.subplot(3, 3, 6)
ax6_twin = ax6.twinx()

monthly_rain = all_data.groupby('month')['RAIN'].sum()
all_data['AQI_approx'] = all_data['PM2.5'] * 2 + all_data['PM10'] * 0.5
monthly_aqi = all_data.groupby('month')['AQI_approx'].mean()

bars = ax6.bar(range(1, 13), monthly_rain.values, 
               color='skyblue', alpha=0.7, label='Rainfall')
line = ax6_twin.plot(range(1, 13), monthly_aqi.values, 
                     color='darkred', linewidth=2, marker='o', markersize=4, label='AQI')

ax6.set_title('Monthly Rainfall vs Air Quality', fontsize=12, fontweight='bold')
ax6.set_xlabel('Month')
ax6.set_ylabel('Rainfall (mm)', color='blue')
ax6_twin.set_ylabel('AQI (approx)', color='darkred')
ax6.legend(loc='upper left')
ax6_twin.legend(loc='upper right')

# Row 3: Weather Impact Analysis

# Subplot 7: Wind speed vs PM2.5
ax7 = plt.subplot(3, 3, 7)
sample_data = all_data.sample(n=min(800, len(all_data))).dropna(subset=['WSPM', 'PM2.5', 'TEMP'])

scatter = ax7.scatter(sample_data['WSPM'], sample_data['PM2.5'], 
                     c=sample_data['TEMP'], cmap='RdYlBu_r', 
                     alpha=0.6, s=20)

# Regression line
z = np.polyfit(sample_data['WSPM'], sample_data['PM2.5'], 1)
p = np.poly1d(z)
ax7.plot(sample_data['WSPM'].sort_values(), p(sample_data['WSPM'].sort_values()), 
         "r--", alpha=0.8, linewidth=2)

ax7.set_title('Wind Speed vs PM2.5', fontsize=12, fontweight='bold')
ax7.set_xlabel('Wind Speed (m/s)')
ax7.set_ylabel('PM2.5 (μg/m³)')
plt.colorbar(scatter, ax=ax7, label='Temp (°C)')
ax7.grid(True, alpha=0.3)

# Subplot 8: Wind direction impact
ax8 = plt.subplot(3, 3, 8)

# Simplified wind categorization
def categorize_wind(wd):
    if pd.isna(wd):
        return 'Calm'
    elif wd in ['N', 'NNE', 'NNW']:
        return 'North'
    elif wd in ['S', 'SSE', 'SSW']:
        return 'South'
    elif wd in ['E', 'ENE', 'ESE']:
        return 'East'
    elif wd in ['W', 'WNW', 'WSW']:
        return 'West'
    else:
        return 'Other'

all_data['wind_category'] = all_data['wd'].apply(categorize_wind)
wind_categories = ['North', 'South', 'East', 'West']
colors_wind = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, category in enumerate(wind_categories):
    cat_data = all_data[all_data['wind_category'] == category]['PM2.5'].dropna()
    if len(cat_data) > 50:
        # Sample data to avoid memory issues
        cat_sample = cat_data.sample(n=min(500, len(cat_data)))
        ax8.hist(cat_sample, bins=20, alpha=0.6, color=colors_wind[i], 
                label=f'{category}', density=True)

ax8.set_title('PM2.5 Distribution by Wind Direction', fontsize=12, fontweight='bold')
ax8.set_xlabel('PM2.5 (μg/m³)')
ax8.set_ylabel('Density')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, 200)

# Subplot 9: Time series decomposition (simplified)
ax9 = plt.subplot(3, 3, 9)

# Monthly time series
monthly_data = all_data.groupby(['year', 'month'])['PM2.5'].mean().reset_index()
monthly_data['period'] = range(len(monthly_data))

# Simple trend (moving average)
window = 6
monthly_data['trend'] = monthly_data['PM2.5'].rolling(window=window, center=True).mean()

# Plot original and trend
ax9.plot(monthly_data['period'], monthly_data['PM2.5'], 
         'k-', linewidth=1.5, label='Original', alpha=0.7)
ax9.plot(monthly_data['period'], monthly_data['trend'], 
         'r-', linewidth=2, label='Trend')

# Add seasonal pattern
seasonal_avg = all_data.groupby('month')['PM2.5'].mean()
seasonal_pattern = [seasonal_avg[row['month']] for _, row in monthly_data.iterrows()]
ax9.plot(monthly_data['period'], seasonal_pattern, 
         'g-', linewidth=1.5, label='Seasonal', alpha=0.7)

ax9.set_title('PM2.5 Time Series Components', fontsize=12, fontweight='bold')
ax9.set_xlabel('Time Period')
ax9.set_ylabel('PM2.5 (μg/m³)')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Final layout
plt.tight_layout(pad=2.0)
plt.show()