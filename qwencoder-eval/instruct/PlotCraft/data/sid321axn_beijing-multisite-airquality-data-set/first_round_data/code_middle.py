import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

# Load and combine all station data efficiently
station_files = [
    'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    'PRSA_Data_Tiantan_20130301-20170228.csv',
    'PRSA_Data_Wanliu_20130301-20170228.csv',
    'PRSA_Data_Shunyi_20130301-20170228.csv',
    'PRSA_Data_Dongsi_20130301-20170228.csv',
    'PRSA_Data_Dingling_20130301-20170228.csv'
]

# Read and combine datasets with sampling to reduce processing time
dfs = []
for file in station_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        # Sample 20% of data from each station to improve performance
        df_sample = df.sample(n=min(7000, len(df)), random_state=42)
        dfs.append(df_sample)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Clean data - remove rows with missing values for key pollutants
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
clean_df = combined_df.dropna(subset=pollutants)

# Further sample for performance
clean_df = clean_df.sample(n=min(15000, len(clean_df)), random_state=42)

# Create figure with 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palette for stations
stations = clean_df['station'].unique()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
station_color_map = dict(zip(stations, colors[:len(stations)]))

# Top-left: Correlation heatmap
ax1 = axes[0, 0]
corr_matrix = clean_df[pollutants].corr()
im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(len(pollutants)))
ax1.set_yticks(range(len(pollutants)))
ax1.set_xticklabels(pollutants, rotation=45, ha='right')
ax1.set_yticklabels(pollutants)
ax1.set_title('Pollutant Correlation Matrix', fontweight='bold', fontsize=12, pad=15)

# Add correlation values to heatmap
for i in range(len(pollutants)):
    for j in range(len(pollutants)):
        text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=9)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

# Top-right: Scatter plot for PM2.5 vs NO2 by station
ax2 = axes[0, 1]
sample_df = clean_df.sample(n=min(3000, len(clean_df)), random_state=42)

for station in stations:
    station_data = sample_df[sample_df['station'] == station]
    if len(station_data) > 0:
        ax2.scatter(station_data['PM2.5'], station_data['NO2'], 
                   c=station_color_map[station], alpha=0.6, s=15, label=station)

ax2.set_xlabel('PM2.5 (μg/m³)', fontweight='bold')
ax2.set_ylabel('NO2 (μg/m³)', fontweight='bold')
ax2.set_title('PM2.5 vs NO2 by Station', fontweight='bold', fontsize=12, pad=15)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Bottom-left: Bubble plot (PM2.5 vs NO2, bubble size = CO, color = station)
ax3 = axes[1, 0]
bubble_df = clean_df.sample(n=min(1500, len(clean_df)), random_state=42)

for station in stations:
    station_data = bubble_df[bubble_df['station'] == station]
    if len(station_data) > 0:
        # Normalize CO values for bubble sizes (scale between 10-80)
        co_min, co_max = bubble_df['CO'].min(), bubble_df['CO'].max()
        if co_max > co_min:
            sizes = ((station_data['CO'] - co_min) / (co_max - co_min)) * 70 + 10
        else:
            sizes = 30
        
        ax3.scatter(station_data['PM2.5'], station_data['NO2'], 
                   s=sizes, c=station_color_map[station], alpha=0.6, label=station)

ax3.set_xlabel('PM2.5 (μg/m³)', fontweight='bold')
ax3.set_ylabel('NO2 (μg/m³)', fontweight='bold')
ax3.set_title('PM2.5 vs NO2 (Bubble Size = CO Level)', fontweight='bold', fontsize=12, pad=15)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Bottom-right: PM2.5 vs PM10 with regression lines by station
ax4 = axes[1, 1]
regression_df = clean_df.sample(n=min(2000, len(clean_df)), random_state=42)

for station in stations:
    station_data = regression_df[regression_df['station'] == station]
    if len(station_data) > 20:  # Need sufficient data for regression
        # Plot scatter points
        ax4.scatter(station_data['PM2.5'], station_data['PM10'], 
                   c=station_color_map[station], alpha=0.5, s=12, label=station)
        
        # Calculate and plot regression line
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(station_data['PM2.5'], station_data['PM10'])
            x_min, x_max = station_data['PM2.5'].min(), station_data['PM2.5'].max()
            line_x = np.linspace(x_min, x_max, 50)
            line_y = slope * line_x + intercept
            ax4.plot(line_x, line_y, color=station_color_map[station], linewidth=2, alpha=0.8)
        except:
            continue

ax4.set_xlabel('PM2.5 (μg/m³)', fontweight='bold')
ax4.set_ylabel('PM10 (μg/m³)', fontweight='bold')
ax4.set_title('PM2.5 vs PM10 with Station-Specific Trends', fontweight='bold', fontsize=12, pad=15)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.45)

# Save the plot
plt.savefig('beijing_air_pollution_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()