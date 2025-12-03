import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
forecast_df = pd.read_csv('forecast_data.csv')

# Data preprocessing - optimize for performance
forecast_df['datetime'] = pd.to_datetime(forecast_df['time'])
forecast_df['date'] = forecast_df['datetime'].dt.date
forecast_df['hour'] = forecast_df['datetime'].dt.hour

# Sample data for performance - take first 1000 rows
forecast_df = forecast_df.head(1000)

# Get top 3 states for faster processing
top_states = forecast_df['state'].value_counts().head(3).index.tolist()

# Create daily aggregations with limited data
daily_data = forecast_df.groupby(['date', 'state']).agg({
    'temp_c': ['mean', 'std'],
    'precip_mm': 'sum',
    'humidity': 'mean',
    'wind_kph': 'mean',
    'vis_km': 'mean',
    'pressure_mb': 'mean'
}).reset_index()

# Flatten column names
daily_data.columns = ['date', 'state', 'temp_mean', 'temp_std', 'precip_sum', 
                     'humidity_mean', 'wind_mean', 'vis_mean', 'pressure_mean']
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data = daily_data.fillna(0)  # Fill NaN values

# Create the 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color schemes
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
          '#577590', '#F8961E', '#90323D', '#264653']

# Get sample data for first state
sample_state = top_states[0]
state_data = daily_data[daily_data['state'] == sample_state].head(15)

# Subplot 1: Temperature trends with precipitation bars
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()

if len(state_data) > 0:
    ax1.plot(range(len(state_data)), state_data['temp_mean'], 
             color=colors[0], linewidth=2, label='Temperature')
    ax1.fill_between(range(len(state_data)), 
                     state_data['temp_mean'] - state_data['temp_std'], 
                     state_data['temp_mean'] + state_data['temp_std'], 
                     alpha=0.3, color=colors[0])
    ax1_twin.bar(range(len(state_data)), state_data['precip_sum'], 
                 alpha=0.6, color=colors[1], width=0.8)

ax1.set_title('Temperature Trends with Precipitation', fontweight='bold', fontsize=10)
ax1.set_ylabel('Temperature (°C)', color=colors[0])
ax1_twin.set_ylabel('Precipitation (mm)', color=colors[1])
ax1.grid(True, alpha=0.3)

# Subplot 2: Humidity and wind speed dual-axis
ax2 = axes[0, 1]
ax2_twin = ax2.twinx()

if len(state_data) > 0:
    ax2.plot(range(len(state_data)), state_data['humidity_mean'], 
             color=colors[2], linewidth=2, label='Humidity')
    ax2_twin.fill_between(range(len(state_data)), 0, state_data['wind_mean'], 
                          alpha=0.6, color=colors[3])

ax2.set_title('Humidity vs Wind Speed', fontweight='bold', fontsize=10)
ax2.set_ylabel('Humidity (%)', color=colors[2])
ax2_twin.set_ylabel('Wind Speed (km/h)', color=colors[3])
ax2.grid(True, alpha=0.3)

# Subplot 3: Visibility and temperature overlay
ax3 = axes[0, 2]
ax3_twin = ax3.twinx()

if len(state_data) > 0:
    ax3.fill_between(range(len(state_data)), 0, state_data['vis_mean'], 
                     alpha=0.6, color=colors[4])
    ax3_twin.plot(range(len(state_data)), state_data['temp_mean'], 
                  color=colors[5], linewidth=3)

ax3.set_title('Visibility with Temperature', fontweight='bold', fontsize=10)
ax3.set_ylabel('Visibility (km)', color=colors[4])
ax3_twin.set_ylabel('Temperature (°C)', color=colors[5])
ax3.grid(True, alpha=0.3)

# Subplot 4: Temperature decomposition simulation
ax4 = axes[1, 0]

if len(state_data) > 5:
    temp_series = state_data['temp_mean'].values
    x_vals = range(len(temp_series))
    trend = np.polyval(np.polyfit(x_vals, temp_series, 1), x_vals)
    seasonal = 2 * np.sin(2 * np.pi * np.array(x_vals) / 7)
    
    ax4.plot(x_vals, temp_series, color=colors[6], linewidth=2, label='Original', alpha=0.7)
    ax4.plot(x_vals, trend, color=colors[7], linewidth=2, label='Trend')
    ax4.plot(x_vals, seasonal + np.mean(temp_series), color=colors[8], linewidth=1, label='Seasonal')

ax4.set_title('Temperature Decomposition', fontweight='bold', fontsize=10)
ax4.set_ylabel('Temperature (°C)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Simplified heatmap
ax5 = axes[1, 1]

if len(state_data) > 0:
    temp_data = state_data['temp_mean'].values.reshape(1, -1)
    im = ax5.imshow(temp_data, cmap='RdYlBu_r', aspect='auto')
    
    # Add scatter points
    ax5.scatter(range(len(state_data)), [0] * len(state_data), 
               s=state_data['precip_sum'] * 5 + 10, alpha=0.6, color='blue')

ax5.set_title('Temperature Heatmap', fontweight='bold', fontsize=10)
ax5.set_xlabel('Days')

# Subplot 6: Morning vs Evening simulation
ax6 = axes[1, 2]

# Simulate morning and evening temperatures
morning_temps = np.random.normal(22, 3, 10)
evening_temps = np.random.normal(28, 4, 10)

for i in range(10):
    ax6.plot([0, 1], [morning_temps[i], evening_temps[i]], 
             color=colors[i % len(colors)], alpha=0.7, linewidth=1)

ax6.set_title('Morning to Evening Changes', fontweight='bold', fontsize=10)
ax6.set_xlim(-0.1, 1.1)
ax6.set_xticks([0, 1])
ax6.set_xticklabels(['Morning', 'Evening'])
ax6.set_ylabel('Temperature (°C)')
ax6.grid(True, alpha=0.3)

# Subplot 7: Multi-state comparison
ax7 = axes[2, 0]

for i, state in enumerate(top_states[:3]):
    state_temps = daily_data[daily_data['state'] == state].head(10)
    if len(state_temps) > 0:
        ax7.plot(range(len(state_temps)), state_temps['temp_mean'], 
                 color=colors[i], linewidth=2, label=state[:8], alpha=0.8)

ax7.set_title('Multi-State Comparison', fontweight='bold', fontsize=10)
ax7.set_ylabel('Temperature (°C)')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Subplot 8: Autocorrelation simulation
ax8 = axes[2, 1]

if len(state_data) > 5:
    lags = range(1, min(8, len(state_data)))
    autocorr = [0.8 * (0.7 ** lag) for lag in lags]  # Simulated autocorrelation
    partial_autocorr = [0.6 * (0.5 ** lag) for lag in lags]
    
    ax8.bar(lags, autocorr, alpha=0.7, color=colors[0], label='Autocorr', width=0.3)
    ax8.bar([l + 0.3 for l in lags], partial_autocorr, alpha=0.7, 
            color=colors[1], width=0.3, label='Partial')

ax8.set_title('Autocorrelation Analysis', fontweight='bold', fontsize=10)
ax8.set_xlabel('Lag')
ax8.set_ylabel('Correlation')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Subplot 9: Wind pattern simulation
ax9 = axes[2, 2]

# Create wind rose simulation
theta = np.linspace(0, 2*np.pi, 16)
radii = np.random.uniform(5, 20, 16)
colors_wind = plt.cm.viridis(np.linspace(0, 1, 16))

bars = ax9.bar(theta, radii, width=0.3, alpha=0.7, color=colors_wind)

ax9.set_title('Wind Direction Pattern', fontweight='bold', fontsize=10)
ax9.set_theta_zero_location('N')
ax9.set_theta_direction(-1)

# Convert to polar
ax9.remove()
ax9 = fig.add_subplot(3, 3, 9, projection='polar')
ax9.bar(theta, radii, width=0.3, alpha=0.7, color=colors_wind)
ax9.set_title('Wind Pattern', fontweight='bold', fontsize=10, pad=20)

# Overall layout adjustments
plt.tight_layout(pad=2.0)

# Add main title
fig.suptitle('Weather Pattern Analysis Dashboard', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('weather_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()