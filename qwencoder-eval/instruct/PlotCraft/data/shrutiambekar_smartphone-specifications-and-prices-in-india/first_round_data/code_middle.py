import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re
import os
import glob

# Load data with robust file finding
def find_csv_file():
    """Find the smartphones CSV file in the current directory or subdirectories"""
    # Try current directory first
    if os.path.exists('smartphones.csv'):
        return 'smartphones.csv'
    
    # Search for CSV files with 'smartphone' in the name
    csv_files = glob.glob('*smartphone*.csv')
    if csv_files:
        return csv_files[0]
    
    # Search in subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'smartphone' in file.lower() and file.endswith('.csv'):
                return os.path.join(root, file)
    
    # If no specific smartphone file found, try any CSV file
    csv_files = glob.glob('*.csv')
    if csv_files:
        return csv_files[0]
    
    raise FileNotFoundError("No CSV file found in the current directory")

# Load data
try:
    csv_file = find_csv_file()
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded data from: {csv_file}")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data for demonstration if file not found
    df = pd.DataFrame({
        'model': ['Phone A', 'Phone B', 'Phone C', 'Phone D', 'Phone E'] * 20,
        'price': ['₹25,999', '₹45,999', '₹15,999', '₹35,999', '₹55,999'] * 20,
        'rating': np.random.uniform(70, 95, 100),
        'ram': ['6 GB RAM, 128 GB inbuilt', '8 GB RAM, 256 GB inbuilt', 
                '4 GB RAM, 64 GB inbuilt', '12 GB RAM, 256 GB inbuilt',
                '8 GB RAM, 128 GB inbuilt'] * 20,
        'battery': ['5000 mAh Battery with Fast Charging', '4500 mAh Battery with 80W Fast Charging',
                   '4000 mAh Battery with 25W Fast Charging', '5000 mAh Battery with 120W Fast Charging',
                   '4800 mAh Battery with 67W Fast Charging'] * 20
    })

# Data preprocessing
def extract_numeric_value(text, pattern):
    """Extract numeric value from text using regex pattern"""
    if pd.isna(text):
        return np.nan
    match = re.search(pattern, str(text))
    return float(match.group(1)) if match else np.nan

def convert_price_to_numeric(price_str):
    """Convert price string to numeric value"""
    if pd.isna(price_str):
        return np.nan
    # Remove currency symbol and commas, extract numeric value
    price_clean = re.sub(r'[₹,]', '', str(price_str))
    try:
        return float(price_clean)
    except:
        return np.nan

# Extract RAM capacity (in GB)
df['ram_gb'] = df['ram'].apply(lambda x: extract_numeric_value(x, r'(\d+)\s*GB\s*RAM'))

# Extract battery capacity (in mAh)
df['battery_mah'] = df['battery'].apply(lambda x: extract_numeric_value(x, r'(\d+)\s*mAh'))

# Convert price to numeric
df['price_numeric'] = df['price'].apply(convert_price_to_numeric)

# Remove rows with missing essential data
df_clean = df.dropna(subset=['ram_gb', 'price_numeric', 'battery_mah', 'rating'])

print(f"Data shape after cleaning: {df_clean.shape}")
print(f"RAM range: {df_clean['ram_gb'].min()}-{df_clean['ram_gb'].max()} GB")
print(f"Price range: ₹{df_clean['price_numeric'].min():,.0f}-₹{df_clean['price_numeric'].max():,.0f}")

# Create the composite visualization
fig, ax1 = plt.subplots(figsize=(14, 8))
plt.style.use('default')
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Create scatter plot with rating-based color coding
scatter = ax1.scatter(df_clean['ram_gb'], df_clean['price_numeric'], 
                     c=df_clean['rating'], cmap='viridis', 
                     alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

# Add regression line
if len(df_clean) > 1:
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['ram_gb'], df_clean['price_numeric'])
        line_x = np.linspace(df_clean['ram_gb'].min(), df_clean['ram_gb'].max(), 100)
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, color='red', linewidth=2, alpha=0.8, 
                 label=f'Regression Line (R² = {r_value**2:.3f})')
    except Exception as e:
        print(f"Could not calculate regression line: {e}")

# Customize primary axis
ax1.set_xlabel('RAM Capacity (GB)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
ax1.set_title('Smartphone Specifications vs Pricing Analysis\nRAM-Price Correlation with Rating Colors and Battery Distribution', 
              fontsize=14, fontweight='bold', pad=20)

# Add colorbar for ratings
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Rating Score', fontsize=11, fontweight='bold')

# Create secondary y-axis for battery histogram
ax2 = ax1.twinx()

# Create histogram of battery capacity with transparency
n_bins = min(25, len(df_clean) // 4)  # Adjust bins based on data size
if n_bins < 5:
    n_bins = 5

counts, bins, patches = ax2.hist(df_clean['battery_mah'], bins=n_bins, 
                                alpha=0.3, color='orange', edgecolor='darkorange', 
                                linewidth=1, label='Battery Capacity Distribution')

# Customize secondary axis
ax2.set_ylabel('Frequency (Battery Distribution)', fontsize=12, fontweight='bold', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

# Add legends
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# Add grid for better readability
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add statistics text box
stats_text = f'Dataset: {len(df_clean)} smartphones\n'
stats_text += f'RAM Range: {df_clean["ram_gb"].min():.0f}-{df_clean["ram_gb"].max():.0f} GB\n'
stats_text += f'Price Range: ₹{df_clean["price_numeric"].min():,.0f}-₹{df_clean["price_numeric"].max():,.0f}\n'
stats_text += f'Battery Range: {df_clean["battery_mah"].min():.0f}-{df_clean["battery_mah"].max():.0f} mAh'

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Adjust layout to prevent overlap
plt.tight_layout()

# Final adjustments for professional appearance
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Format price axis with proper currency formatting
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))

plt.show()