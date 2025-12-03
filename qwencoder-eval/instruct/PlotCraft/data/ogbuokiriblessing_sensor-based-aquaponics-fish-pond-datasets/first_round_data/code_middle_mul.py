import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and standardize all pond data
pond_files = ['IoTPond10.csv', 'IoTPond6.csv', 'IoTPond3.csv', 'IoTPond8.csv', 
              'IoTPond9.csv', 'IoTPond7.csv', 'IoTPond11.csv', 'IoTpond1.csv', 
              'IoTPond4.csv', 'IoTPond2.csv', 'IoTPond12.csv']

def standardize_columns(df, pond_id):
    """Standardize column names across all ponds"""
    column_mapping = {
        'TEMPERATURE': 'Temperature',
        'Temperature(C)': 'Temperature',
        'Temperature (C)': 'Temperature',
        'temperature(C)': 'Temperature',
        'TURBIDITY': 'Turbidity',
        'Turbidity(NTU)': 'Turbidity',
        'Turbidity (NTU)': 'Turbidity',
        'turbidity (NTU)': 'Turbidity',
        'DISOLVED OXYGEN': 'Dissolved_Oxygen',
        'Dissolved Oxygen(g/ml)': 'Dissolved_Oxygen',
        'Dissolved Oxygen (mg/L)': 'Dissolved_Oxygen',
        'Dissolved Oxygen (g/ml)': 'Dissolved_Oxygen',
        'pH': 'pH',
        'PH': 'pH',
        'AMMONIA': 'Ammonia',
        'Ammonia(g/ml)': 'Ammonia',
        'Ammonia (mg/L)': 'Ammonia',
        'ammonia(g/ml)': 'Ammonia',
        'NITRATE': 'Nitrate',
        'Nitrate(g/ml)': 'Nitrate',
        'Nitrate (mg/L)': 'Nitrate',
        'nitrate(g/ml)': 'Nitrate',
        'Length': 'Fish_Length',
        'Lenght': 'Fish_Length',
        'Fish_Length(cm)': 'Fish_Length',
        'Fish_Length (cm)': 'Fish_Length',
        'Fish_length(cm)': 'Fish_Length',
        'Total_length (cm)': 'Fish_Length',
        'Weight': 'Fish_Weight',
        'Fish_Weight(g)': 'Fish_Weight',
        'Fish_Weight (g)': 'Fish_Weight',
        'Fish_weight(g)': 'Fish_Weight',
        'Weight (g)': 'Fish_Weight'
    }
    
    df = df.rename(columns=column_mapping)
    df['Pond_ID'] = pond_id
    
    # Select only the standardized columns we need
    required_cols = ['Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Fish_Length', 'Fish_Weight', 'Pond_ID']
    available_cols = [col for col in required_cols if col in df.columns]
    
    return df[available_cols]

# Load and process all pond data
all_ponds = []
loaded_files = []

for i, file in enumerate(pond_files, 1):
    try:
        print(f"Attempting to load {file}...")
        df = pd.read_csv(file)
        print(f"Successfully loaded {file} with shape {df.shape}")
        df_std = standardize_columns(df, f'Pond_{i}')
        
        # Only add if we have some required columns
        if len(df_std.columns) > 1:  # At least Pond_ID + one other column
            all_ponds.append(df_std)
            loaded_files.append(file)
            print(f"Added {file} to dataset")
        else:
            print(f"Skipped {file} - insufficient columns after standardization")
            
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Successfully loaded {len(all_ponds)} files: {loaded_files}")

# Check if we have any data to work with
if len(all_ponds) == 0:
    print("No data files could be loaded. Creating sample data for demonstration.")
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = []
    for i in range(3):
        n_samples = 1000
        data = {
            'Temperature': np.random.normal(25 + i, 2, n_samples),
            'Turbidity': np.random.normal(50 + i*10, 10, n_samples),
            'Dissolved_Oxygen': np.random.normal(8 + i, 1, n_samples),
            'pH': np.random.normal(7.5 + i*0.2, 0.3, n_samples),
            'Ammonia': np.random.exponential(2 + i, n_samples),
            'Nitrate': np.random.normal(100 + i*20, 20, n_samples),
            'Fish_Length': np.random.normal(10 + i, 2, n_samples),
            'Fish_Weight': np.random.normal(15 + i*3, 3, n_samples),
            'Pond_ID': f'Pond_{i+1}'
        }
        sample_data.append(pd.DataFrame(data))
    all_ponds = sample_data

# Combine all pond data
combined_df = pd.concat(all_ponds, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Clean anomalous values
combined_df = combined_df.replace([np.inf, -np.inf], np.nan)

# Clean specific parameters if they exist
if 'Temperature' in combined_df.columns:
    combined_df.loc[combined_df['Temperature'] < -50, 'Temperature'] = np.nan
    combined_df.loc[combined_df['Temperature'] > 50, 'Temperature'] = np.nan

if 'Turbidity' in combined_df.columns:
    combined_df.loc[combined_df['Turbidity'] < 0, 'Turbidity'] = np.nan

if 'Dissolved_Oxygen' in combined_df.columns:
    combined_df.loc[combined_df['Dissolved_Oxygen'] < 0, 'Dissolved_Oxygen'] = np.nan

if 'pH' in combined_df.columns:
    combined_df.loc[combined_df['pH'] < 0, 'pH'] = np.nan
    combined_df.loc[combined_df['pH'] > 14, 'pH'] = np.nan

# Remove rows with too many missing values
combined_df = combined_df.dropna(thresh=3)  # Keep rows with at least 3 non-null values

print(f"Dataset after cleaning: {combined_df.shape}")
print(f"Available columns: {combined_df.columns.tolist()}")

# Create figure with 2x2 subplot layout
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('white')

# Define water quality parameters for correlation analysis
water_params = ['Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate']
available_water_params = [param for param in water_params if param in combined_df.columns]

print(f"Available water parameters: {available_water_params}")

# 1. Top-left: Correlation heatmap of water quality parameters
ax1 = plt.subplot(2, 2, 1)

if len(available_water_params) >= 2:
    water_data = combined_df[available_water_params].dropna()
    
    if len(water_data) > 0:
        correlation_matrix = water_data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax1)
        ax1.set_title('Water Quality Parameters\nCorrelation Heatmap', fontweight='bold', fontsize=12, pad=15)
    else:
        ax1.text(0.5, 0.5, 'No water quality data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Water Quality Parameters\nCorrelation Heatmap', fontweight='bold', fontsize=12, pad=15)
else:
    ax1.text(0.5, 0.5, 'Insufficient water quality parameters', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Water Quality Parameters\nCorrelation Heatmap', fontweight='bold', fontsize=12, pad=15)

# 2. Top-right: Scatter plot of most correlated parameters
ax2 = plt.subplot(2, 2, 2)

if len(available_water_params) >= 2:
    water_data = combined_df[available_water_params + ['Pond_ID']].dropna()
    
    if len(water_data) > 0:
        # Find most correlated pair
        corr_matrix = water_data[available_water_params].corr()
        corr_pairs = []
        
        for i in range(len(available_water_params)):
            for j in range(i+1, len(available_water_params)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if not np.isnan(corr_val):
                    corr_pairs.append((available_water_params[i], available_water_params[j], corr_val))
        
        if corr_pairs:
            # Sort by correlation strength and take the top pair
            top_pair = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[0]
            param1, param2, corr_val = top_pair
            
            # Create color map for ponds
            unique_ponds = water_data['Pond_ID'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ponds)))
            pond_colors = dict(zip(unique_ponds, colors))
            
            for pond in unique_ponds:
                pond_data = water_data[water_data['Pond_ID'] == pond]
                ax2.scatter(pond_data[param1], pond_data[param2], 
                           c=[pond_colors[pond]], label=pond, alpha=0.6, s=30)
            
            ax2.set_xlabel(param1, fontweight='bold')
            ax2.set_ylabel(param2, fontweight='bold')
            ax2.set_title(f'Top Correlation: {param1} vs {param2}\n(r={corr_val:.2f})', 
                          fontweight='bold', fontsize=12, pad=15)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No valid correlations found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Correlation Scatter', fontweight='bold', fontsize=12, pad=15)
    else:
        ax2.text(0.5, 0.5, 'No data available for correlation', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Parameter Correlation Scatter', fontweight='bold', fontsize=12, pad=15)
else:
    ax2.text(0.5, 0.5, 'Insufficient parameters for correlation', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Parameter Correlation Scatter', fontweight='bold', fontsize=12, pad=15)

# 3. Bottom-left: Bubble chart - Fish Weight vs Length with DO as bubble size
ax3 = plt.subplot(2, 2, 3)

required_bubble_cols = ['Fish_Weight', 'Fish_Length', 'Dissolved_Oxygen', 'Pond_ID']
available_bubble_cols = [col for col in required_bubble_cols if col in combined_df.columns]

if len(available_bubble_cols) >= 3:  # Need at least 3 columns including Pond_ID
    bubble_data = combined_df[available_bubble_cols].dropna()
    
    if len(bubble_data) > 0 and 'Fish_Weight' in bubble_data.columns and 'Fish_Length' in bubble_data.columns:
        # Create color map for ponds
        unique_ponds = bubble_data['Pond_ID'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ponds)))
        pond_colors = dict(zip(unique_ponds, colors))
        
        # Normalize bubble sizes
        if 'Dissolved_Oxygen' in bubble_data.columns:
            do_min, do_max = bubble_data['Dissolved_Oxygen'].min(), bubble_data['Dissolved_Oxygen'].max()
            if do_max > do_min:
                do_normalized = (bubble_data['Dissolved_Oxygen'] - do_min) / (do_max - do_min) * 100 + 20
            else:
                do_normalized = pd.Series([50] * len(bubble_data))
            size_label = 'Dissolved Oxygen'
        else:
            do_normalized = pd.Series([50] * len(bubble_data))
            size_label = 'Fixed Size'
        
        for pond in unique_ponds:
            pond_data = bubble_data[bubble_data['Pond_ID'] == pond]
            pond_sizes = do_normalized[bubble_data['Pond_ID'] == pond]
            
            ax3.scatter(pond_data['Fish_Length'], pond_data['Fish_Weight'],
                       s=pond_sizes, c=[pond_colors[pond]], alpha=0.6, 
                       label=pond, edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('Fish Length (cm)', fontweight='bold')
        ax3.set_ylabel('Fish Weight (g)', fontweight='bold')
        ax3.set_title(f'Fish Weight vs Length\n(Bubble size = {size_label})', 
                      fontweight='bold', fontsize=12, pad=15)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No fish measurement data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Fish Weight vs Length\n(Bubble Chart)', fontweight='bold', fontsize=12, pad=15)
else:
    ax3.text(0.5, 0.5, 'Insufficient data for bubble chart', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Fish Weight vs Length\n(Bubble Chart)', fontweight='bold', fontsize=12, pad=15)

# 4. Bottom-right: Ammonia vs Fish Weight with regression lines
ax4 = plt.subplot(2, 2, 4)

required_reg_cols = ['Ammonia', 'Fish_Weight', 'Pond_ID']
available_reg_cols = [col for col in required_reg_cols if col in combined_df.columns]

if len(available_reg_cols) >= 3:
    regression_data = combined_df[available_reg_cols].dropna()
    
    if len(regression_data) > 0:
        # Create color map for ponds
        unique_ponds = regression_data['Pond_ID'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ponds)))
        pond_colors = dict(zip(unique_ponds, colors))
        
        for pond in unique_ponds:
            pond_data = regression_data[regression_data['Pond_ID'] == pond]
            
            if len(pond_data) > 5:  # Only plot if enough data points
                ax4.scatter(pond_data['Ammonia'], pond_data['Fish_Weight'],
                           c=[pond_colors[pond]], alpha=0.6, s=30, label=pond)
                
                # Fit regression line with confidence interval
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(pond_data['Ammonia'], pond_data['Fish_Weight'])
                    x_range = np.linspace(pond_data['Ammonia'].min(), pond_data['Ammonia'].max(), 100)
                    y_pred = slope * x_range + intercept
                    
                    ax4.plot(x_range, y_pred, color=pond_colors[pond], linewidth=2, alpha=0.8)
                    
                    # Simple confidence interval estimation
                    residuals = pond_data['Fish_Weight'] - (slope * pond_data['Ammonia'] + intercept)
                    mse = np.mean(residuals**2)
                    confidence_interval = 1.96 * np.sqrt(mse)
                    
                    ax4.fill_between(x_range, y_pred - confidence_interval, y_pred + confidence_interval, 
                                   color=pond_colors[pond], alpha=0.2)
                except Exception as e:
                    print(f"Regression failed for {pond}: {e}")
        
        ax4.set_xlabel('Ammonia Levels', fontweight='bold')
        ax4.set_ylabel('Fish Weight (g)', fontweight='bold')
        ax4.set_title('Ammonia vs Fish Weight\n(with Regression Lines & Confidence Intervals)', 
                      fontweight='bold', fontsize=12, pad=15)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data available for regression', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Ammonia vs Fish Weight\n(Regression Analysis)', fontweight='bold', fontsize=12, pad=15)
else:
    ax4.text(0.5, 0.5, 'Insufficient data for regression analysis', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Ammonia vs Fish Weight\n(Regression Analysis)', fontweight='bold', fontsize=12, pad=15)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.4)

# Add overall title
fig.suptitle('Comprehensive Aquaponics Correlation Analysis', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('aquaponics_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()