import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Find all CSV files in the current directory
csv_files = glob.glob('*.csv')
print(f"Found CSV files: {csv_files}")

def standardize_columns(df, pond_id):
    """Standardize column names across different pond datasets"""
    # Create a mapping for common variations
    column_mapping = {
        'TEMPERATURE': 'temperature',
        'Temperature(C)': 'temperature',
        'Temperature (C)': 'temperature',
        'temperature(C)': 'temperature',
        'Temperature (C)': 'temperature',
        'TURBIDITY': 'turbidity',
        'Turbidity(NTU)': 'turbidity',
        'Turbidity (NTU)': 'turbidity',
        'turbidity (NTU)': 'turbidity',
        'Turbidity(NTU)': 'turbidity',
        'DISOLVED OXYGEN': 'dissolved_oxygen',
        'Dissolved Oxygen(g/ml)': 'dissolved_oxygen',
        'Dissolved Oxygen (mg/L)': 'dissolved_oxygen',
        'Dissolved Oxygen (g/ml)': 'dissolved_oxygen',
        'pH': 'ph',
        'PH': 'ph',
        'AMMONIA': 'ammonia',
        'Ammonia(g/ml)': 'ammonia',
        'Ammonia (mg/L)': 'ammonia',
        'ammonia(g/ml)': 'ammonia',
        'NITRATE': 'nitrate',
        'Nitrate(g/ml)': 'nitrate',
        'Nitrate (mg/L)': 'nitrate',
        'nitrate(g/ml)': 'nitrate',
        'Length': 'fish_length',
        'Lenght': 'fish_length',
        'Fish_Length(cm)': 'fish_length',
        'Fish_Length (cm)': 'fish_length',
        'Total_length (cm)': 'fish_length',
        'Fish_length(cm)': 'fish_length',
        'Weight': 'fish_weight',
        'Fish_Weight(g)': 'fish_weight',
        'Fish_Weight (g)': 'fish_weight',
        'Weight (g)': 'fish_weight',
        'Fish_weight(g)': 'fish_weight',
        'Population': 'population'
    }
    
    # Rename columns
    df_clean = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    required_cols = ['temperature', 'turbidity', 'dissolved_oxygen', 'ph', 
                    'ammonia', 'nitrate', 'fish_length', 'fish_weight', 'population']
    
    # Keep only available columns
    available_cols = [col for col in required_cols if col in df_clean.columns]
    df_clean = df_clean[available_cols].copy()
    
    # Add pond ID
    df_clean['pond_id'] = pond_id
    
    return df_clean

# Load and combine all pond data
all_ponds = []
for file in csv_files:
    try:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        print(f"Loaded {file} with shape: {df.shape}")
        
        # Extract pond ID from filename
        if 'Pond' in file or 'pond' in file:
            pond_id = ''.join(filter(str.isdigit, file))
            pond_id = int(pond_id) if pond_id else len(all_ponds) + 1
        else:
            pond_id = len(all_ponds) + 1
            
        df_clean = standardize_columns(df, pond_id)
        
        if len(df_clean) > 0:
            all_ponds.append(df_clean)
            print(f"Successfully processed {file} with {len(df_clean)} rows")
        else:
            print(f"No valid data in {file}")
            
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Check if we have any data
if len(all_ponds) == 0:
    print("No valid data found. Creating sample data for demonstration...")
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = []
    
    for pond_id in range(1, 4):
        n_samples = 100
        data = {
            'temperature': np.random.normal(25, 2, n_samples),
            'turbidity': np.random.normal(50, 15, n_samples),
            'dissolved_oxygen': np.random.normal(8, 2, n_samples),
            'ph': np.random.normal(7.5, 0.5, n_samples),
            'ammonia': np.random.exponential(2, n_samples),
            'nitrate': np.random.normal(100, 30, n_samples),
            'fish_length': np.random.normal(8 + pond_id, 1.5, n_samples),
            'fish_weight': np.random.normal(4 + pond_id, 1, n_samples),
            'population': np.random.choice([50, 75, 100], n_samples),
            'pond_id': pond_id
        }
        sample_data.append(pd.DataFrame(data))
    
    all_ponds = sample_data

# Combine all data
combined_df = pd.concat(all_ponds, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")
print(f"Available columns: {combined_df.columns.tolist()}")

# Clean data - remove invalid values and outliers
combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
combined_df = combined_df.dropna()

# Remove extreme outliers (values beyond 3 standard deviations)
numeric_cols = ['temperature', 'dissolved_oxygen', 'ph', 'ammonia', 'fish_length', 'fish_weight']
for col in numeric_cols:
    if col in combined_df.columns:
        mean_val = combined_df[col].mean()
        std_val = combined_df[col].std()
        if std_val > 0:  # Avoid division by zero
            combined_df = combined_df[
                (combined_df[col] >= mean_val - 3*std_val) & 
                (combined_df[col] <= mean_val + 3*std_val)
            ]

# Sample data for better performance if dataset is large
if len(combined_df) > 1000:
    combined_df = combined_df.sample(n=1000, random_state=42).reset_index(drop=True)

# Create population density (fish per unit - normalized)
if 'population' in combined_df.columns:
    combined_df['population_density'] = combined_df['population'] / 100  # Normalize for sizing

# Set up the figure with white background
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define consistent color palette for ponds
unique_ponds = sorted(combined_df['pond_id'].unique())
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ponds)))
pond_colors = dict(zip(unique_ponds, colors))

# 1. Top-left: Dissolved Oxygen vs Fish Weight (colored by pond, sized by population density)
if all(col in combined_df.columns for col in ['dissolved_oxygen', 'fish_weight']):
    pop_density_col = 'population_density' if 'population_density' in combined_df.columns else None
    
    for pond in unique_ponds:
        pond_data = combined_df[combined_df['pond_id'] == pond]
        if len(pond_data) > 0:
            sizes = pond_data[pop_density_col] * 100 if pop_density_col else 50
            ax1.scatter(pond_data['dissolved_oxygen'], pond_data['fish_weight'], 
                       c=[pond_colors[pond]], s=sizes, 
                       alpha=0.6, label=f'Pond {pond}', edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel('Dissolved Oxygen (mg/L)', fontweight='bold')
    ax1.set_ylabel('Fish Weight (g)', fontweight='bold')
    ax1.set_title('Dissolved Oxygen vs Fish Weight\n(Size = Population Density)', fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'Data not available\nfor this plot', ha='center', va='center', 
             transform=ax1.transAxes, fontsize=12)
    ax1.set_title('Dissolved Oxygen vs Fish Weight', fontweight='bold')

# 2. Top-right: Correlation heatmap of water quality parameters
water_quality_cols = ['temperature', 'ph', 'dissolved_oxygen', 'ammonia', 'nitrate', 'turbidity']
available_wq_cols = [col for col in water_quality_cols if col in combined_df.columns]

if len(available_wq_cols) >= 3:
    corr_matrix = combined_df[available_wq_cols].corr()
    
    # Create heatmap
    im = ax2.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax2.set_xticks(range(len(available_wq_cols)))
    ax2.set_yticks(range(len(available_wq_cols)))
    ax2.set_xticklabels([col.replace('_', ' ').title() for col in available_wq_cols], 
                       rotation=45, ha='right')
    ax2.set_yticklabels([col.replace('_', ' ').title() for col in available_wq_cols])
    
    # Add correlation values
    for i in range(len(available_wq_cols)):
        for j in range(len(available_wq_cols)):
            text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax2.set_title('Water Quality Parameters\nCorrelation Matrix', fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Insufficient water quality\nparameters for correlation', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Water Quality Correlation Matrix', fontweight='bold')

# 3. Bottom-left: pH vs Fish Length with regression line
if all(col in combined_df.columns for col in ['ph', 'fish_length']):
    # Different markers for different population sizes
    populations = sorted(combined_df['population'].unique()) if 'population' in combined_df.columns else [50]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, pop in enumerate(populations[:len(markers)]):
        pop_data = combined_df[combined_df['population'] == pop] if 'population' in combined_df.columns else combined_df
        if len(pop_data) > 0:
            for pond in unique_ponds:
                pond_pop_data = pop_data[pop_data['pond_id'] == pond]
                if len(pond_pop_data) > 0:
                    ax3.scatter(pond_pop_data['ph'], pond_pop_data['fish_length'],
                               c=[pond_colors[pond]], marker=markers[i % len(markers)], 
                               s=60, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # Add regression line
    if len(combined_df) > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(combined_df['ph'], combined_df['fish_length'])
        line_x = np.linspace(combined_df['ph'].min(), combined_df['ph'].max(), 100)
        line_y = slope * line_x + intercept
        ax3.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.8, 
                label=f'R² = {r_value**2:.3f}')
    
    ax3.set_xlabel('pH Level', fontweight='bold')
    ax3.set_ylabel('Fish Length (cm)', fontweight='bold')
    ax3.set_title('pH vs Fish Length\n(Different Markers = Population Sizes)', fontweight='bold', pad=20)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'Data not available\nfor this plot', ha='center', va='center', 
             transform=ax3.transAxes, fontsize=12)
    ax3.set_title('pH vs Fish Length', fontweight='bold')

# 4. Bottom-right: Temperature vs Ammonia bubble plot (bubble size = fish weight)
if all(col in combined_df.columns for col in ['temperature', 'ammonia', 'fish_weight']):
    for pond in unique_ponds:
        pond_data = combined_df[combined_df['pond_id'] == pond]
        if len(pond_data) > 0:
            # Normalize bubble sizes
            min_weight = combined_df['fish_weight'].min()
            max_weight = combined_df['fish_weight'].max()
            if max_weight > min_weight:
                bubble_sizes = ((pond_data['fish_weight'] - min_weight) / (max_weight - min_weight) + 0.1) * 100
            else:
                bubble_sizes = np.full(len(pond_data), 50)
                
            ax4.scatter(pond_data['temperature'], pond_data['ammonia'], 
                       s=bubble_sizes, c=[pond_colors[pond]], alpha=0.6, 
                       label=f'Pond {pond}', edgecolors='white', linewidth=0.5)
    
    ax4.set_xlabel('Temperature (°C)', fontweight='bold')
    ax4.set_ylabel('Ammonia (mg/L)', fontweight='bold')
    ax4.set_title('Temperature vs Ammonia\n(Bubble Size = Fish Weight)', fontweight='bold', pad=20)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Data not available\nfor this plot', ha='center', va='center', 
             transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Temperature vs Ammonia', fontweight='bold')

# Set white background for all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('white')

# Overall title
fig.suptitle('Aquaponics Water Quality and Fish Growth Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig('aquaponics_analysis.png', dpi=300, bbox_inches='tight')
plt.show()