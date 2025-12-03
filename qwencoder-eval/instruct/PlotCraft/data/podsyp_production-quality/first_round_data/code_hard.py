import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df_X = pd.read_csv('data_X.csv')
df_Y = pd.read_csv('data_Y.csv')

# Convert date_time columns to datetime
df_X['date_time'] = pd.to_datetime(df_X['date_time'])
df_Y['date_time'] = pd.to_datetime(df_Y['date_time'])

# Sample data heavily for performance (take every 1000th row from X data)
df_X_sampled = df_X.iloc[::1000, :].reset_index(drop=True)

# Simple merge by rounding to nearest hour for faster processing
df_X_sampled['hour'] = df_X_sampled['date_time'].dt.floor('H')
df_Y['hour'] = df_Y['date_time'].dt.floor('H')

# Merge on hour
merged_df = pd.merge(df_X_sampled, df_Y[['hour', 'quality']], on='hour', how='inner')

# Further sample if still too large (take max 5000 points)
if len(merged_df) > 5000:
    merged_df = merged_df.sample(n=5000, random_state=42).reset_index(drop=True)

# Remove any remaining NaN values
merged_df = merged_df.dropna()

# Define sensor columns for each chamber
chamber_sensors = {
    1: ['T_data_1_1', 'T_data_1_2', 'T_data_1_3'],
    2: ['T_data_2_1', 'T_data_2_2', 'T_data_2_3'],
    3: ['T_data_3_1', 'T_data_3_2', 'T_data_3_3']
}

# Define colors for each chamber
chamber_colors = {1: '#2E86AB', 2: '#A23B72', 3: '#F18F01'}

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.patch.set_facecolor('white')

# Function to create scatter plot with regression line
def create_subplot(ax, x_data, y_data, sensor_name, chamber_num, color):
    # Scatter plot
    ax.scatter(x_data, y_data, alpha=0.5, s=15, color=color, edgecolors='none')
    
    # Polynomial regression (degree 2) with error handling
    try:
        if len(x_data) > 3:  # Need at least 3 points for degree 2 polynomial
            z = np.polyfit(x_data, y_data, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(x_data.min(), x_data.max(), 50)
            ax.plot(x_smooth, p(x_smooth), color='darkred', linewidth=2, alpha=0.8)
    except:
        # Fallback to linear regression
        try:
            slope, intercept, _, _, _ = stats.linregress(x_data, y_data)
            x_smooth = np.linspace(x_data.min(), x_data.max(), 50)
            ax.plot(x_smooth, slope * x_smooth + intercept, color='darkred', linewidth=2, alpha=0.8)
        except:
            pass
    
    # Calculate correlation coefficient
    try:
        corr_coef = np.corrcoef(x_data, y_data)[0, 1]
        if not np.isnan(corr_coef):
            ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=9, fontweight='bold', verticalalignment='top')
    except:
        pass
    
    # Styling
    ax.set_xlabel(f'{sensor_name}', fontsize=9)
    ax.set_ylabel('Quality', fontsize=9)
    ax.set_title(f'Chamber {chamber_num}: {sensor_name}', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor('white')

# Create 3x3 grid of subplots
for chamber_idx, chamber_num in enumerate([1, 2, 3]):
    for sensor_idx, sensor in enumerate(chamber_sensors[chamber_num]):
        ax = axes[chamber_idx, sensor_idx]
        
        # Get data for this sensor
        x_data = merged_df[sensor].values
        y_data = merged_df['quality'].values
        
        # Remove any NaN or infinite values
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) > 0:  # Only plot if we have data
            create_subplot(ax, x_data, y_data, sensor, chamber_num, chamber_colors[chamber_num])
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Chamber {chamber_num}: {sensor}', fontweight='bold', fontsize=10)

# Create correlation heatmap in a separate small subplot
# Add a small inset for correlation heatmap
heatmap_ax = fig.add_axes([0.02, 0.02, 0.25, 0.25])  # [left, bottom, width, height]

# Calculate correlation matrix for sensors vs quality
sensor_cols = [sensor for sensors in chamber_sensors.values() for sensor in sensors]
corr_matrix = merged_df[sensor_cols + ['quality']].corr()

# Create simplified heatmap
im = heatmap_ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
heatmap_ax.set_title('Correlation\nHeatmap', fontsize=8, fontweight='bold')
heatmap_ax.set_xticks([])
heatmap_ax.set_yticks([])

# Add colorbar
cbar = plt.colorbar(im, ax=heatmap_ax, shrink=0.6)
cbar.ax.tick_params(labelsize=6)

# Add overall title
fig.suptitle('Temperature Sensors vs Production Quality Analysis\n(3x3 Chamber Grid with Polynomial Regression)', 
             fontsize=14, fontweight='bold', y=0.95)

# Add chamber legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=chamber_colors[i], 
                             markersize=8, label=f'Chamber {i}') for i in [1, 2, 3]]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88), fontsize=10)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.95, hspace=0.35, wspace=0.3)

plt.savefig('temperature_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()