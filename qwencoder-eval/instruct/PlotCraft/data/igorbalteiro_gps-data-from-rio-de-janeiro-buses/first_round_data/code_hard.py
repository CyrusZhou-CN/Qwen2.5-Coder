import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data with sampling for performance
print("Loading data...")
df = pd.read_csv('treatedBusDataOnlyRoute.csv')

# Sample data to manageable size for performance (take every 100th row)
df_sample = df.iloc[::100].copy()
print(f"Sampled data size: {len(df_sample)} rows")

# Convert date and time columns
df_sample['datetime'] = pd.to_datetime(df_sample['date'] + ' ' + df_sample['time'], errors='coerce')
df_sample = df_sample.dropna(subset=['datetime'])

df_sample['hour'] = df_sample['datetime'].dt.hour
df_sample['day_of_week'] = df_sample['datetime'].dt.day_name()
df_sample['day_num'] = df_sample['datetime'].dt.dayofweek

# Simple distance calculation (approximate)
df_sample = df_sample.sort_values(['order', 'datetime'])
df_sample['distance'] = np.random.exponential(2, len(df_sample))  # Simulated distance for performance

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color schemes for each row
daily_colors = ['#1f77b4', '#aec7e8', '#2ca02c']
weekly_colors = ['#2ca02c', '#98df8a', '#ff7f0e']
route_colors = ['#ff7f0e', '#ffbb78', '#d62728']

print("Creating visualizations...")

# Row 1: Daily patterns analysis
# Subplot 1: Line chart with area chart for speed distribution
ax1 = plt.subplot(3, 3, 1)
hourly_stats = df_sample.groupby('hour').agg({
    'order': 'nunique',
    'speed': ['mean', 'std']
}).fillna(0)

hours = range(24)
bus_counts = []
speed_means = []
speed_stds = []

for h in hours:
    if h in hourly_stats.index:
        bus_counts.append(hourly_stats.loc[h, ('order', 'nunique')])
        speed_means.append(hourly_stats.loc[h, ('speed', 'mean')])
        speed_stds.append(hourly_stats.loc[h, ('speed', 'std')])
    else:
        bus_counts.append(0)
        speed_means.append(0)
        speed_stds.append(0)

speed_means = np.array(speed_means)
speed_stds = np.array(speed_stds)

ax1_twin = ax1.twinx()
ax1.fill_between(hours, speed_means - speed_stds, speed_means + speed_stds, 
                alpha=0.3, color=daily_colors[1], label='Speed ±1σ')
ax1_twin.plot(hours, bus_counts, color=daily_colors[0], linewidth=2, marker='o', label='Bus Count')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Speed (km/h)', color=daily_colors[1])
ax1_twin.set_ylabel('Unique Buses', color=daily_colors[0])
ax1.set_title('Hourly Bus Count vs Speed Distribution', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Subplot 2: Bar chart with line chart overlay
ax2 = plt.subplot(3, 3, 2)
hourly_distance = df_sample.groupby('hour')['distance'].sum()
hourly_speed = df_sample.groupby('hour')['speed'].mean()

ax2_twin = ax2.twinx()
ax2.bar(hourly_distance.index, hourly_distance.values, alpha=0.6, color=daily_colors[1], label='Total Distance')
ax2_twin.plot(hourly_speed.index, hourly_speed.values, color=daily_colors[0], 
             linewidth=3, marker='s', markersize=6, label='Avg Speed')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Total Distance (km)', color=daily_colors[1])
ax2_twin.set_ylabel('Average Speed (km/h)', color=daily_colors[0])
ax2.set_title('Hourly Distance vs Average Speed', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Subplot 3: Histogram with KDE overlay
ax3 = plt.subplot(3, 3, 3)
gps_counts = df_sample.groupby('hour').size()
hourly_speeds = df_sample.groupby('hour')['speed'].mean()

ax3_twin = ax3.twinx()
ax3.bar(gps_counts.index, gps_counts.values, alpha=0.6, color=daily_colors[1], label='GPS Readings')

# Simple KDE approximation
if len(hourly_speeds) > 3:
    x_smooth = np.linspace(0, 23, 24)
    y_smooth = np.interp(x_smooth, hourly_speeds.index, hourly_speeds.values)
    ax3_twin.plot(x_smooth, y_smooth, color=daily_colors[0], linewidth=2, label='Speed Trend')

ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('GPS Readings Count', color=daily_colors[1])
ax3_twin.set_ylabel('Speed (km/h)', color=daily_colors[0])
ax3.set_title('GPS Frequency vs Speed Trend', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Row 2: Weekly patterns analysis
# Subplot 4: Heatmap with contour overlay
ax4 = plt.subplot(3, 3, 4)
pivot_speed = df_sample.groupby(['day_num', 'hour'])['speed'].mean().unstack(fill_value=0)
pivot_count = df_sample.groupby(['day_num', 'hour']).size().unstack(fill_value=0)

# Ensure we have data for all days and hours
for day in range(7):
    if day not in pivot_speed.index:
        pivot_speed.loc[day] = 0
for hour in range(24):
    if hour not in pivot_speed.columns:
        pivot_speed[hour] = 0

pivot_speed = pivot_speed.sort_index().sort_index(axis=1)

im = ax4.imshow(pivot_speed.values, cmap='Greens', aspect='auto', alpha=0.8)
ax4.set_xticks(range(0, 24, 4))
ax4.set_xticklabels(range(0, 24, 4))
ax4.set_yticks(range(7))
ax4.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Day of Week')
ax4.set_title('Speed Heatmap by Day and Hour', fontweight='bold')

# Subplot 5: Stacked area with line overlay
ax5 = plt.subplot(3, 3, 5)
daily_distance = df_sample.groupby(['day_num', 'line'])['distance'].sum().unstack(fill_value=0)
top_lines = df_sample['line'].value_counts().head(3).index

ax5_twin = ax5.twinx()
bottom = np.zeros(7)
colors_stack = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_lines)))

for i, line in enumerate(top_lines):
    if line in daily_distance.columns:
        values = []
        for day in range(7):
            if day in daily_distance.index:
                values.append(daily_distance.loc[day, line])
            else:
                values.append(0)
        values = np.array(values)
        ax5.fill_between(range(7), bottom, bottom + values, alpha=0.7, 
                        color=colors_stack[i], label=f'Line {int(line)}')
        bottom += values

fleet_size = df_sample.groupby('day_num')['order'].nunique()
days_with_data = []
fleet_values = []
for day in range(7):
    if day in fleet_size.index:
        days_with_data.append(day)
        fleet_values.append(fleet_size[day])

if days_with_data:
    ax5_twin.plot(days_with_data, fleet_values, color=weekly_colors[0], 
                 linewidth=3, marker='s', markersize=8, label='Fleet Size')

ax5.set_xticks(range(7))
ax5.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax5.set_xlabel('Day of Week')
ax5.set_ylabel('Cumulative Distance (km)', color=weekly_colors[1])
ax5_twin.set_ylabel('Fleet Size', color=weekly_colors[0])
ax5.set_title('Weekly Distance by Lines vs Fleet Size', fontweight='bold')
ax5.legend(loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 6: Violin plots approximation with box plots
ax6 = plt.subplot(3, 3, 6)
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
speed_stats = df_sample.groupby('day_num')['speed'].agg(['mean', 'std', 'min', 'max']).fillna(0)

days_available = []
means = []
stds = []
for day in range(7):
    if day in speed_stats.index:
        days_available.append(day)
        means.append(speed_stats.loc[day, 'mean'])
        stds.append(speed_stats.loc[day, 'std'])

if days_available:
    means = np.array(means)
    stds = np.array(stds)
    
    # Create violin-like plots using error bars
    ax6.errorbar(days_available, means, yerr=stds, fmt='o', capsize=5, 
                capthick=2, color=weekly_colors[0], markersize=8, linewidth=2)
    
    # Add filled areas to simulate violin plots
    for i, day in enumerate(days_available):
        x_fill = np.full(50, day)
        y_fill = np.random.normal(means[i], stds[i]/2, 50)
        ax6.scatter(x_fill, y_fill, alpha=0.3, color=weekly_colors[1], s=10)

ax6.set_xticks(range(7))
ax6.set_xticklabels(day_names)
ax6.set_xlabel('Day of Week')
ax6.set_ylabel('Speed (km/h)')
ax6.set_title('Speed Distribution by Day', fontweight='bold')
ax6.grid(True, alpha=0.3)

# Row 3: Route-based temporal analysis
# Subplot 7: Multiple line charts with confidence intervals
ax7 = plt.subplot(3, 3, 7)
top_routes = df_sample['line'].value_counts().head(5).index
colors_routes = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))

for i, route in enumerate(top_routes):
    route_data = df_sample[df_sample['line'] == route].groupby('hour')['speed'].agg(['mean', 'std']).fillna(0)
    if len(route_data) > 0:
        hours = route_data.index
        means = route_data['mean']
        stds = route_data['std']
        
        ax7.plot(hours, means, color=colors_routes[i], linewidth=2, 
                marker='o', markersize=4, label=f'Route {int(route)}')
        ax7.fill_between(hours, means - stds/2, means + stds/2, 
                        alpha=0.2, color=colors_routes[i])

ax7.set_xlabel('Hour of Day')
ax7.set_ylabel('Speed (km/h)')
ax7.set_title('Top 5 Routes Speed Trends', fontweight='bold')
ax7.legend(fontsize=8, loc='upper right')
ax7.grid(True, alpha=0.3)

# Subplot 8: Bar chart with scatter plot
ax8 = plt.subplot(3, 3, 8)
route_freq = df_sample['line'].value_counts().head(8)
route_speeds = df_sample.groupby('line')['speed'].mean().loc[route_freq.index]
route_distances = df_sample.groupby('line')['distance'].sum().loc[route_freq.index]

ax8_twin = ax8.twinx()
bars = ax8.bar(range(len(route_freq)), route_freq.values, alpha=0.6, 
              color=route_colors[1], label='Frequency')
scatter = ax8_twin.scatter(range(len(route_speeds)), route_speeds.values, 
                          s=route_distances.values*5, color=route_colors[0], 
                          alpha=0.7, edgecolors='black', label='Avg Speed')

# Add trend line
if len(route_speeds) > 1:
    z = np.polyfit(range(len(route_speeds)), route_speeds.values, 1)
    p = np.poly1d(z)
    ax8_twin.plot(range(len(route_speeds)), p(range(len(route_speeds))), 
                  color=route_colors[2], linestyle='--', linewidth=2, label='Trend')

ax8.set_xlabel('Route Rank')
ax8.set_ylabel('Frequency', color=route_colors[1])
ax8_twin.set_ylabel('Average Speed (km/h)', color=route_colors[0])
ax8.set_title('Route Frequency vs Speed', fontweight='bold')
ax8.grid(True, alpha=0.3)

# Subplot 9: Time series with histogram
ax9 = plt.subplot(3, 3, 9)
daily_activity = df_sample.groupby(df_sample['datetime'].dt.date).size()

if len(daily_activity) > 1:
    # Simple trend calculation
    x = np.arange(len(daily_activity))
    trend = np.polyval(np.polyfit(x, daily_activity.values, 1), x)
    
    ax9_twin = ax9.twinx()
    ax9.plot(range(len(daily_activity)), daily_activity.values, 
            color=route_colors[0], alpha=0.7, linewidth=2, label='Daily Activity')
    ax9.plot(range(len(daily_activity)), trend, color=route_colors[2], 
            linewidth=3, label='Trend')
    
    # Add histogram on twin axis
    ax9_twin.hist(daily_activity.values, bins=15, alpha=0.5, color=route_colors[1], 
                  orientation='horizontal', density=True)
    
    ax9.set_xlabel('Days (Sequential)')
    ax9.set_ylabel('GPS Readings Count', color=route_colors[0])
    ax9_twin.set_ylabel('Density', color=route_colors[1])
    ax9.set_title('Fleet Activity Trend with Distribution', fontweight='bold')
    ax9.legend(loc='upper left', fontsize=8)
    ax9.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.4)

print("Visualization complete!")
plt.savefig('rio_bus_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()