import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data more efficiently - only load what we need
def load_station_data(station_name):
    """Load and preprocess station data efficiently"""
    df = pd.read_csv(f'PRSA_Data_{station_name}_20130301-20170228.csv')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df

# Load only required stations to reduce memory usage
key_stations = ['Aotizhongxin', 'Tiantan', 'Wanliu', 'Dongsi', 'Wanshouxigong', 'Guanyuan']
dfs = {}
for station in key_stations:
    dfs[station] = load_station_data(station)

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('white')

# Define color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#4CC9F0']

# 1. Top-left: PM2.5 time series with rolling averages
ax1 = plt.subplot(3, 2, 1)
selected_stations = ['Aotizhongxin', 'Dongsi', 'Wanliu', 'Tiantan']

for i, station in enumerate(selected_stations):
    df = dfs[station].copy()
    # Sample data to reduce processing time - take every 24th hour (daily)
    daily_data = df[df['hour'] == 12].copy()  # Use noon data as daily representative
    daily_data = daily_data.set_index('datetime')['PM2.5'].dropna()
    
    # Calculate 7-day rolling average
    rolling_avg = daily_data.rolling(window=7, center=True).mean()
    rolling_std = daily_data.rolling(window=7, center=True).std()
    
    # Plot with reduced data points
    sample_indices = np.arange(0, len(daily_data), 5)  # Every 5th point
    
    ax1.plot(daily_data.index[sample_indices], daily_data.iloc[sample_indices], 
             alpha=0.3, color=colors[i], linewidth=0.5)
    ax1.plot(rolling_avg.index, rolling_avg, color=colors[i], linewidth=2, label=station)
    ax1.fill_between(rolling_avg.index, 
                     rolling_avg - rolling_std, 
                     rolling_avg + rolling_std, 
                     alpha=0.2, color=colors[i])

ax1.set_title('PM2.5 Time Series with 7-Day Rolling Averages', fontsize=12, fontweight='bold')
ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=10)
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 300)

# 2. Top-right: Monthly PM2.5 bars with temperature line
ax2 = plt.subplot(3, 2, 2)

# Combine data from all loaded stations
all_data = pd.concat([df.assign(station=name) for name, df in dfs.items()])
monthly_data = all_data.groupby(['year', 'month']).agg({
    'PM2.5': 'mean',
    'TEMP': 'mean'
}).reset_index()
monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))

# Bar chart for PM2.5
bars = ax2.bar(monthly_data['date'], monthly_data['PM2.5'], 
               color='#2E86AB', alpha=0.7, width=20, label='PM2.5')

# Dual axis for temperature
ax2_temp = ax2.twinx()
ax2_temp.plot(monthly_data['date'], monthly_data['TEMP'], 
              color='#F18F01', linewidth=2, marker='o', markersize=3, label='Temperature')

ax2.set_title('Monthly PM2.5 Levels and Temperature Trends', fontsize=12, fontweight='bold')
ax2.set_ylabel('PM2.5 (μg/m³)', fontsize=10, color='#2E86AB')
ax2_temp.set_ylabel('Temperature (°C)', fontsize=10, color='#F18F01')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_temp.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Middle-left: Stacked area chart for pollutant composition
ax3 = plt.subplot(3, 2, 3)
central_stations = ['Dongsi', 'Tiantan', 'Wanshouxigong']
central_data = pd.concat([dfs[station] for station in central_stations if station in dfs])

# Sample data monthly to reduce computation
monthly_pollutants = central_data.groupby(['year', 'month']).agg({
    'PM2.5': 'mean', 'PM10': 'mean', 'SO2': 'mean', 
    'NO2': 'mean', 'CO': 'mean', 'O3': 'mean'
}).reset_index()
monthly_pollutants['date'] = pd.to_datetime(monthly_pollutants[['year', 'month']].assign(day=1))

# Clean and normalize data
pollutant_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
for col in pollutant_cols:
    monthly_pollutants[col] = monthly_pollutants[col].fillna(0)

# Scale pollutants for better visualization
monthly_pollutants['CO_scaled'] = monthly_pollutants['CO'] / 100
monthly_pollutants['O3_scaled'] = monthly_pollutants['O3'] / 2

ax3.stackplot(monthly_pollutants['date'], 
              monthly_pollutants['PM2.5'], monthly_pollutants['PM10'], 
              monthly_pollutants['SO2'], monthly_pollutants['NO2'], 
              monthly_pollutants['CO_scaled'], monthly_pollutants['O3_scaled'],
              labels=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO (÷100)', 'O3 (÷2)'],
              colors=colors, alpha=0.8)

ax3.set_title('Pollutant Composition Over Time (Central Beijing)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Concentration', fontsize=10)
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Middle-right: Seasonal slope chart
ax4 = plt.subplot(3, 2, 4)
seasonal_data = []
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}

for i, (station, df) in enumerate(dfs.items()):
    df_copy = df.copy()
    df_copy['season'] = df_copy['month'].map(season_map)
    seasonal_avg = df_copy.groupby('season')['PM2.5'].mean()
    
    seasons_order = ['Spring', 'Summer', 'Fall', 'Winter']
    values = [seasonal_avg.get(season, 0) for season in seasons_order]
    
    ax4.plot(range(4), values, 'o-', color=colors[i], 
             linewidth=2, markersize=5, alpha=0.8, label=station)

ax4.set_title('Seasonal PM2.5 Patterns by Station', fontsize=12, fontweight='bold')
ax4.set_ylabel('PM2.5 (μg/m³)', fontsize=10)
ax4.set_xticks(range(4))
ax4.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Bottom-left: Calendar heatmap for 2015
ax5 = plt.subplot(3, 2, 5)
dongsi_2015 = dfs['Dongsi'][dfs['Dongsi']['year'] == 2015].copy()
daily_2015 = dongsi_2015.groupby(['month', 'day'])['PM2.5'].mean().reset_index()

# Create simplified calendar heatmap
calendar_data = np.full((12, 31), np.nan)
for _, row in daily_2015.iterrows():
    if row['day'] <= 31:
        calendar_data[int(row['month'])-1, int(row['day'])-1] = row['PM2.5']

im = ax5.imshow(calendar_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=200)
ax5.set_title('Daily PM2.5 Calendar Heatmap (2015, Dongsi)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Month', fontsize=10)
ax5.set_xlabel('Day', fontsize=10)
ax5.set_yticks(range(0, 12, 2))
ax5.set_yticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
ax5.set_xticks(range(0, 31, 10))
ax5.set_xticklabels(['1', '11', '21', '31'])

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, shrink=0.6)
cbar.set_label('PM2.5 (μg/m³)', fontsize=9)

# 6. Bottom-right: Simplified time series analysis
ax6 = plt.subplot(3, 2, 6)

# Find most polluted station
station_pollution = {name: df['PM2.5'].mean() for name, df in dfs.items()}
most_polluted = max(station_pollution, key=station_pollution.get)

# Use monthly data for trend analysis
monthly_trend = dfs[most_polluted].groupby(['year', 'month'])['PM2.5'].mean().reset_index()
monthly_trend['date'] = pd.to_datetime(monthly_trend[['year', 'month']].assign(day=1))

# Calculate simple trend and seasonal patterns
monthly_trend = monthly_trend.set_index('date')['PM2.5']
trend = monthly_trend.rolling(window=12, center=True).mean()

# Calculate seasonal component (simplified)
monthly_trend_df = monthly_trend.reset_index()
monthly_trend_df['month'] = monthly_trend_df['date'].dt.month
seasonal_pattern = monthly_trend_df.groupby('month')['PM2.5'].mean()
seasonal_component = monthly_trend_df['date'].dt.month.map(seasonal_pattern)

ax6.plot(monthly_trend.index, monthly_trend, color='lightgray', alpha=0.5, label='Original', linewidth=1)
ax6.plot(trend.index, trend, color='#2E86AB', linewidth=3, label='Trend')
ax6.plot(monthly_trend.index, seasonal_component, color='#F18F01', linewidth=2, label='Seasonal Pattern')

ax6.set_title(f'Time Series Analysis ({most_polluted})', fontsize=12, fontweight='bold')
ax6.set_ylabel('PM2.5 (μg/m³)', fontsize=10)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)
plt.show()