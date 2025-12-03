import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data for available ponds only
pond_data = {}

# Load IoTPond3
try:
    df3 = pd.read_csv('IoTPond3.csv')
    df3['created_at'] = pd.to_datetime(df3['created_at'])
    df3 = df3.sort_values('created_at')
    # Sample every 500th point for performance
    df3 = df3.iloc[::500].copy()
    pond_data['IoTPond3'] = df3
except:
    pass

# Load IoTPond6
try:
    df6 = pd.read_csv('IoTPond6.csv')
    df6['created_at'] = pd.to_datetime(df6['created_at'])
    df6 = df6.sort_values('created_at')
    # Sample every 500th point for performance
    df6 = df6.iloc[::500].copy()
    pond_data['IoTPond6'] = df6
except:
    pass

# Load IoTPond9
try:
    df9 = pd.read_csv('IoTPond9.csv')
    df9['created_at'] = pd.to_datetime(df9['created_at'])
    df9 = df9.sort_values('created_at')
    # Standardize column names
    df9 = df9.rename(columns={'Fish_length(cm)': 'Fish_Length(cm)', 'Fish_weight(g)': 'Fish_Weight(g)'})
    # Sample every 500th point for performance
    df9 = df9.iloc[::500].copy()
    pond_data['IoTPond9'] = df9
except:
    pass

# Create simulated data for IoTPond1 (lightweight)
np.random.seed(42)
n_points = 200  # Reduced for performance
base_dates = pd.date_range(start='2021-06-18', end='2021-12-31', periods=n_points)

pond1_data = {
    'created_at': base_dates,
    'Temperature(C)': 25 + 3 * np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 0.5, n_points),
    'PH': 7.5 + 0.5 * np.sin(np.linspace(0, 6*np.pi, n_points)) + np.random.normal(0, 0.1, n_points),
    'Dissolved Oxygen(g/ml)': 8 + 2 * np.cos(np.linspace(0, 5*np.pi, n_points)) + np.random.normal(0, 0.3, n_points),
    'Ammonia(g/ml)': 3 + 1.5 * np.sin(np.linspace(0, 3*np.pi, n_points)) + np.random.normal(0, 0.2, n_points),
    'Nitrate(g/ml)': 200 + 50 * np.cos(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 10, n_points),
    'Fish_Length(cm)': 6.5 + 0.5 * np.linspace(0, 1, n_points) + np.random.normal(0, 0.1, n_points),
    'Fish_Weight(g)': 3.5 + 0.8 * np.linspace(0, 1, n_points) + np.random.normal(0, 0.1, n_points)
}
pond_data['IoTPond1'] = pd.DataFrame(pond1_data)

# Define color palette for ponds
pond_colors = {
    'IoTPond1': '#1f77b4',  # Blue
    'IoTPond3': '#ff7f0e',  # Orange
    'IoTPond6': '#2ca02c',  # Green
    'IoTPond9': '#d62728'   # Red
}

# Create the comprehensive dashboard
fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor('white')

# Subplot 1: Temperature trends with rolling averages
ax1 = plt.subplot(3, 2, 1)
for pond_name, df in pond_data.items():
    if len(df) > 10:
        # Simple rolling average with smaller window
        window_size = min(10, len(df)//5)
        if window_size >= 3:
            df_temp = df.copy()
            df_temp['temp_rolling'] = df_temp['Temperature(C)'].rolling(window=window_size, center=True).mean()
            df_temp['temp_std'] = df_temp['Temperature(C)'].rolling(window=window_size, center=True).std()
            
            # Plot original data (lighter)
            ax1.plot(df_temp['created_at'], df_temp['Temperature(C)'], 
                    color=pond_colors[pond_name], alpha=0.3, linewidth=0.5)
            
            # Plot rolling average
            ax1.plot(df_temp['created_at'], df_temp['temp_rolling'], 
                    color=pond_colors[pond_name], linewidth=2, label=f'{pond_name}')
            
            # Add confidence bands
            valid_mask = ~df_temp['temp_std'].isna()
            if valid_mask.sum() > 0:
                ax1.fill_between(df_temp['created_at'][valid_mask], 
                                (df_temp['temp_rolling'] - df_temp['temp_std'])[valid_mask], 
                                (df_temp['temp_rolling'] + df_temp['temp_std'])[valid_mask],
                                color=pond_colors[pond_name], alpha=0.2)

ax1.set_title('Temperature Trends with Rolling Averages & Confidence Bands', fontweight='bold', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (°C)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: pH and Dissolved Oxygen dual-axis
ax2 = plt.subplot(3, 2, 2)
ax2_twin = ax2.twinx()

for pond_name, df in pond_data.items():
    if len(df) > 0:
        # pH on left axis
        ax2.plot(df['created_at'], df['PH'], 
                color=pond_colors[pond_name], linewidth=2, 
                label=f'{pond_name} pH', linestyle='-')
        
        # Dissolved Oxygen on right axis
        ax2_twin.plot(df['created_at'], df['Dissolved Oxygen(g/ml)'], 
                     color=pond_colors[pond_name], linewidth=1.5, 
                     label=f'{pond_name} DO', linestyle='--', alpha=0.7)

ax2.set_title('pH Levels & Dissolved Oxygen Trends', fontweight='bold', fontsize=12)
ax2.set_xlabel('Time')
ax2.set_ylabel('pH Level', color='black')
ax2_twin.set_ylabel('Dissolved Oxygen (g/ml)', color='gray')
ax2.legend(loc='upper left', fontsize=8)
ax2_twin.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Ammonia concentration with simplified analysis
ax3 = plt.subplot(3, 2, 3)

# Create 5 time periods for analysis
all_dates = []
for df in pond_data.values():
    if len(df) > 0:
        all_dates.extend(df['created_at'].tolist())

if all_dates:
    min_date, max_date = min(all_dates), max(all_dates)
    date_bins = pd.date_range(start=min_date, end=max_date, periods=6)
    
    ammonia_by_period = []
    period_labels = []
    
    for i in range(len(date_bins)-1):
        period_data = []
        for pond_name, df in pond_data.items():
            if len(df) > 0:
                mask = (df['created_at'] >= date_bins[i]) & (df['created_at'] < date_bins[i+1])
                period_data.extend(df[mask]['Ammonia(g/ml)'].tolist())
        
        if period_data:
            ammonia_by_period.append(period_data)
            period_labels.append(f'{date_bins[i].strftime("%m/%d")}')
    
    # Create box plots
    if ammonia_by_period:
        bp = ax3.boxplot(ammonia_by_period, patch_artist=True, alpha=0.6)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('#lightblue')
        
        # Overlay mean lines for each pond
        for pond_name, df in pond_data.items():
            if len(df) > 0:
                period_means = []
                x_positions = []
                
                for i, period_start in enumerate(date_bins[:-1]):
                    period_end = date_bins[i+1]
                    mask = (df['created_at'] >= period_start) & (df['created_at'] < period_end)
                    period_ammonia = df[mask]['Ammonia(g/ml)']
                    
                    if len(period_ammonia) > 0:
                        period_means.append(period_ammonia.mean())
                        x_positions.append(i+1)
                
                if len(period_means) > 1:
                    ax3.plot(x_positions, period_means, 
                            color=pond_colors[pond_name], marker='o', linewidth=2,
                            label=f'{pond_name}', markersize=6)
        
        ax3.set_xticklabels(period_labels, rotation=45)

ax3.set_title('Ammonia Concentration Evolution', fontweight='bold', fontsize=12)
ax3.set_xlabel('Time Period')
ax3.set_ylabel('Ammonia (g/ml)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Nitrate trends for representative pond
ax4 = plt.subplot(3, 2, 4)

# Use the pond with most data
best_pond = max(pond_data.keys(), key=lambda x: len(pond_data[x]))
df_decomp = pond_data[best_pond].copy()

if len(df_decomp) > 20:
    # Simple trend analysis
    nitrate_values = df_decomp['Nitrate(g/ml)'].values
    time_values = np.arange(len(nitrate_values))
    
    # Calculate trend using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_values, nitrate_values)
    trend_line = slope * time_values + intercept
    
    # Calculate moving average for seasonal component
    window = min(10, len(nitrate_values)//3)
    if window >= 3:
        seasonal = pd.Series(nitrate_values).rolling(window=window, center=True).mean()
        residual = nitrate_values - seasonal.fillna(nitrate_values.mean())
    else:
        seasonal = pd.Series([nitrate_values.mean()] * len(nitrate_values))
        residual = nitrate_values - seasonal
    
    # Plot components
    ax4.plot(df_decomp['created_at'], nitrate_values, label='Original', color='black', linewidth=2)
    ax4.plot(df_decomp['created_at'], trend_line, label='Trend', color='red', linewidth=2)
    ax4.plot(df_decomp['created_at'], seasonal, label='Seasonal', color='blue', linewidth=1.5)

ax4.set_title(f'Nitrate Analysis ({best_pond})', fontweight='bold', fontsize=12)
ax4.set_xlabel('Time')
ax4.set_ylabel('Nitrate (g/ml)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Parameter correlation heatmap
ax5 = plt.subplot(3, 2, 5)

# Combine data from all ponds for correlation
combined_data = []
for pond_name, df in pond_data.items():
    if len(df) > 0:
        df_corr = df[['Temperature(C)', 'PH', 'Dissolved Oxygen(g/ml)', 'Ammonia(g/ml)']].copy()
        combined_data.append(df_corr)

if combined_data:
    all_data = pd.concat(combined_data, ignore_index=True)
    corr_matrix = all_data.corr()
    
    # Create heatmap
    im = ax5.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax5.set_xticks(range(len(corr_matrix.columns)))
    ax5.set_yticks(range(len(corr_matrix.columns)))
    ax5.set_xticklabels(['Temp', 'pH', 'DO', 'NH3'], rotation=45)
    ax5.set_yticklabels(['Temp', 'pH', 'DO', 'NH3'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Correlation')

ax5.set_title('Parameter Correlation Heatmap', fontweight='bold', fontsize=12)

# Subplot 6: Fish growth indicators
ax6 = plt.subplot(3, 2, 6)
ax6_twin = ax6.twinx()

for pond_name, df in pond_data.items():
    if len(df) > 0 and 'Fish_Length(cm)' in df.columns and 'Fish_Weight(g)' in df.columns:
        # Fish length on left axis
        ax6.plot(df['created_at'], df['Fish_Length(cm)'], 
                color=pond_colors[pond_name], linewidth=2, 
                label=f'{pond_name} Length', marker='o', markersize=3)
        
        # Fish weight on right axis
        ax6_twin.plot(df['created_at'], df['Fish_Weight(g)'], 
                     color=pond_colors[pond_name], linewidth=2, 
                     linestyle='--', label=f'{pond_name} Weight', 
                     marker='s', markersize=3, alpha=0.7)

ax6.set_title('Fish Growth Indicators', fontweight='bold', fontsize=12)
ax6.set_xlabel('Time')
ax6.set_ylabel('Fish Length (cm)', color='black')
ax6_twin.set_ylabel('Fish Weight (g)', color='gray')
ax6.legend(loc='upper left', fontsize=8)
ax6_twin.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Save the plot
plt.savefig('aquaponics_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()