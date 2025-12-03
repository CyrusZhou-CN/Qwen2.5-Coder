import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data from all stations
station_files = [
    'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    'PRSA_Data_Dongsi_20130301-20170228.csv',
    'PRSA_Data_Dingling_20130301-20170228.csv',
    'PRSA_Data_Gucheng_20130301-20170228.csv',
    'PRSA_Data_Huairou_20130301-20170228.csv',
    'PRSA_Data_Nongzhanguan_20130301-20170228.csv'
]

# Combine data from 6 stations (optimized for performance)
all_data = []
for file in station_files:
    try:
        df = pd.read_csv(file)
        # Sample data to reduce processing time
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        all_data.append(df)
    except FileNotFoundError:
        print(f"File {file} not found, skipping...")
        continue

if not all_data:
    print("No data files found")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Clean data - remove rows with missing values in key pollutants
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
clean_df = combined_df.dropna(subset=pollutants)

# Further sample for performance if dataset is still large
if len(clean_df) > 20000:
    clean_df = clean_df.sample(n=20000, random_state=42)

print(f"Working with {len(clean_df)} data points from {clean_df['station'].nunique()} stations")

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palette for stations
stations = clean_df['station'].unique()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
station_colors = dict(zip(stations, colors[:len(stations)]))

# Top-left: Correlation heatmap for all pollutants
corr_matrix = clean_df[pollutants].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, ax=ax1, cbar_kws={'shrink': 0.8}, fmt='.2f')
ax1.set_title('Pollutant Correlation Matrix\n(All Stations Combined)', 
              fontweight='bold', fontsize=12, pad=15)
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=0)

# Top-right: Scatter plot for PM2.5 vs O3 by station
sample_size = min(2000, len(clean_df))
sample_df = clean_df.sample(n=sample_size, random_state=42)

for station in stations:
    station_data = sample_df[sample_df['station'] == station]
    if len(station_data) > 0:
        ax2.scatter(station_data['PM2.5'], station_data['O3'], 
                   c=station_colors[station], alpha=0.6, s=20, label=station)

ax2.set_xlabel('PM2.5 (μg/m³)')
ax2.set_ylabel('O3 (μg/m³)')
ax2.set_title('PM2.5 vs O3 Relationship by Station', 
              fontweight='bold', fontsize=12, pad=15)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# Bottom-left: Bubble plot (PM2.5 vs NO2, bubble size = CO, color = station)
bubble_sample = clean_df.sample(n=min(1500, len(clean_df)), random_state=42)

for station in stations:
    station_data = bubble_sample[bubble_sample['station'] == station]
    if len(station_data) > 0:
        # Normalize CO values for bubble size (scale between 10-200)
        co_normalized = (station_data['CO'] - station_data['CO'].min()) / (station_data['CO'].max() - station_data['CO'].min() + 1e-8)
        bubble_sizes = 10 + co_normalized * 100
        
        ax3.scatter(station_data['PM2.5'], station_data['NO2'], 
                   s=bubble_sizes, c=station_colors[station], 
                   alpha=0.6, label=station, edgecolors='white', linewidth=0.5)

ax3.set_xlabel('PM2.5 (μg/m³)')
ax3.set_ylabel('NO2 (μg/m³)')
ax3.set_title('PM2.5 vs NO2 Relationship\n(Bubble Size = CO Level)', 
              fontweight='bold', fontsize=12, pad=15)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', fontsize=9)

# Bottom-right: PM2.5 vs PM10 with regression lines by station
for station in stations:
    station_data = clean_df[clean_df['station'] == station]
    if len(station_data) > 50:  # Ensure sufficient data
        # Sample for performance
        station_sample = station_data.sample(n=min(800, len(station_data)), random_state=42)
        
        x = station_sample['PM2.5'].values
        y = station_sample['PM10'].values
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) > 10:
            # Plot scatter points
            ax4.scatter(x, y, c=station_colors[station], alpha=0.5, s=15, label=station)
            
            # Calculate and plot regression line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line_x = np.linspace(x.min(), x.max(), 50)
                line_y = slope * line_x + intercept
                ax4.plot(line_x, line_y, color=station_colors[station], 
                        linewidth=2, alpha=0.8)
            except:
                continue

ax4.set_xlabel('PM2.5 (μg/m³)')
ax4.set_ylabel('PM10 (μg/m³)')
ax4.set_title('PM2.5 vs PM10 with Regression Lines\n(By Station)', 
              fontweight='bold', fontsize=12, pad=15)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left', fontsize=9)

# Add main title
fig.suptitle('Beijing Air Quality: Multi-Station Pollutant Correlation Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)

# Save the plot
plt.savefig('beijing_air_quality_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()