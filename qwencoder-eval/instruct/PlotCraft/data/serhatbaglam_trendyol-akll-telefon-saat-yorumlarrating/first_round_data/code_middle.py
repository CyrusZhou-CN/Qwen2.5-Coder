import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re
import os

# Define datasets with error handling
datasets = {
    'Xiaomi_Watch': 'trendyol_xiaomi_saat_yorum_rating.xlsx',
    'Reeder': 'trendyol_reeder_yorum_rating.xlsx',
    'Mateo_Watch': 'trendyol_mateo_saat_yorum_rating.xlsx',
    'Samsung_Phone': 'trendyol_samsung_telefon_yorum.xlsx',
    'Xiaomi_Phone': 'trendyol_xiaomi_yorum_rating.xlsx',
    'Huawei_Watch': 'trendyol_huawei_saat_yorum_rating.xlsx',
    'Apple_Watch': 'trendyol_apple_watch_yorum_rating.xlsx',
    'Samsung_Watch': 'trendyol_samsung_watch_yorum_rating.xlsx',
    'iPhone': 'trendyol_iphone_yorum.xlsx'
}

# Function to extract price from product name
def extract_price(product_name):
    """Extract price information from product name using regex patterns"""
    if pd.isna(product_name):
        return None
    
    # Look for price patterns like "128 GB", "512 GB", "4 GB", etc.
    # Higher storage/memory typically means higher price
    storage_patterns = re.findall(r'(\d+)\s*GB', str(product_name), re.IGNORECASE)
    if storage_patterns:
        # Use the highest storage value found as price indicator
        max_storage = max([int(x) for x in storage_patterns])
        # Estimate price based on storage (simplified mapping)
        if max_storage >= 512:
            return 25000  # High-end
        elif max_storage >= 256:
            return 15000  # Mid-high
        elif max_storage >= 128:
            return 10000  # Mid
        elif max_storage >= 64:
            return 7000   # Mid-low
        else:
            return 5000   # Entry level
    
    # For watches and other products without clear storage indicators
    # Use brand-based estimation
    product_lower = str(product_name).lower()
    if 'apple' in product_lower or 'iphone' in product_lower:
        if 'ultra' in product_lower or 'pro' in product_lower:
            return 20000
        elif 'series' in product_lower:
            return 12000
        else:
            return 8000
    elif 'samsung' in product_lower:
        if 'ultra' in product_lower:
            return 18000
        elif 'galaxy' in product_lower:
            return 10000
        else:
            return 6000
    elif 'huawei' in product_lower:
        return 4000
    elif 'xiaomi' in product_lower:
        return 3000
    elif 'mateo' in product_lower:
        return 1500
    elif 'reeder' in product_lower:
        return 2000
    else:
        return 5000  # Default

# Load and combine all data with error handling
all_data = []
successfully_loaded = []

for brand_category, filename in datasets.items():
    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename)
            df['Brand_Category'] = brand_category
            # Extract estimated price
            df['Estimated_Price'] = df['Telefon'].apply(extract_price)
            all_data.append(df)
            successfully_loaded.append(brand_category)
            print(f"Successfully loaded: {filename}")
        else:
            print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")

if not all_data:
    print("No data files could be loaded. Please check file paths.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Extract brand names from brand_category
brand_mapping = {
    'Xiaomi_Watch': 'Xiaomi',
    'Reeder': 'Reeder', 
    'Mateo_Watch': 'Mateo',
    'Samsung_Phone': 'Samsung',
    'Xiaomi_Phone': 'Xiaomi',
    'Huawei_Watch': 'Huawei',
    'Apple_Watch': 'Apple',
    'Samsung_Watch': 'Samsung',
    'iPhone': 'Apple'
}

combined_df['Brand'] = combined_df['Brand_Category'].map(brand_mapping)

# Calculate brand-level metrics
brand_metrics = combined_df.groupby('Brand').agg({
    'Yıldız': 'mean',
    'Estimated_Price': 'mean',
    'Telefon': 'count'  # Total reviews count
}).round(2)

brand_metrics.columns = ['Avg_Rating', 'Avg_Price', 'Total_Reviews']
brand_metrics = brand_metrics.reset_index()

# Remove any brands with insufficient data
brand_metrics = brand_metrics[brand_metrics['Total_Reviews'] >= 10]

if len(brand_metrics) == 0:
    print("Insufficient data for analysis")
    exit()

print("Brand Metrics Summary:")
print(brand_metrics)

# Create the composite visualization
fig, ax1 = plt.subplots(figsize=(14, 10))

# Color palette for brands
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#2D5016', '#8B4513', '#4B0082']
brand_colors = dict(zip(brand_metrics['Brand'], colors[:len(brand_metrics)]))

# Primary plot: Scatter plot with point sizes representing total reviews
# Scale point sizes appropriately
min_size = 100
max_size = 1000
size_range = max_size - min_size
review_range = brand_metrics['Total_Reviews'].max() - brand_metrics['Total_Reviews'].min()

if review_range > 0:
    point_sizes = min_size + (brand_metrics['Total_Reviews'] - brand_metrics['Total_Reviews'].min()) / review_range * size_range
else:
    point_sizes = [min_size] * len(brand_metrics)

scatter = ax1.scatter(brand_metrics['Avg_Rating'], 
                     brand_metrics['Avg_Price'],
                     s=point_sizes,
                     c=[brand_colors[brand] for brand in brand_metrics['Brand']],
                     alpha=0.7, 
                     edgecolors='white', 
                     linewidth=2)

# Add brand labels to points
for i, row in brand_metrics.iterrows():
    ax1.annotate(f"{row['Brand']}\n({row['Total_Reviews']} reviews)", 
                (row['Avg_Rating'], row['Avg_Price']),
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=9, 
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Fit and plot regression line if we have enough data points
if len(brand_metrics) >= 2:
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(brand_metrics['Avg_Rating'], 
                                                                      brand_metrics['Avg_Price'])
        line_x = np.linspace(brand_metrics['Avg_Rating'].min(), brand_metrics['Avg_Rating'].max(), 100)
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, '--', color='#333333', alpha=0.8, linewidth=2, 
                 label=f'Trend Line (R² = {r_value**2:.3f})')
        
        # Add correlation info
        correlation_text = f'Correlation: r = {r_value:.3f}'
    except:
        correlation_text = 'Correlation: Unable to calculate'
        r_value = 0
else:
    correlation_text = 'Insufficient data for correlation'
    r_value = 0

# Primary axis settings
ax1.set_xlabel('Average Rating (Stars)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Estimated Price (TL)', fontsize=12, fontweight='bold', color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Format y-axis to show prices in thousands
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

# Secondary y-axis: Bar chart showing review counts
ax2 = ax1.twinx()

# Create bar positions
bar_positions = np.arange(len(brand_metrics))
bar_width = 0.6

# Position bars at the bottom of the plot
bars = ax2.bar(bar_positions, 
               brand_metrics['Total_Reviews'],
               alpha=0.3,
               color=[brand_colors[brand] for brand in brand_metrics['Brand']],
               width=bar_width)

ax2.set_ylabel('Total Review Count', fontsize=12, fontweight='bold', color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')

# Set x-axis for bars
ax2.set_xlim(-0.5, len(brand_metrics) - 0.5)
ax2.set_xticks(bar_positions)
ax2.set_xticklabels(brand_metrics['Brand'], rotation=45, ha='right')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, brand_metrics['Total_Reviews'])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
             f'{value:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Title
plt.suptitle('Brand Performance Analysis: Rating vs Price Correlation\nTrendyol Technology Products (Smartphones & Smartwatches)', 
             fontsize=16, fontweight='bold', y=0.95)

# Create custom legend for brands
legend_elements = []
for brand in brand_metrics['Brand']:
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=brand_colors[brand], 
                                    markersize=10, label=brand))

# Add trend line to legend if it exists
if len(brand_metrics) >= 2 and 'r_value' in locals():
    legend_elements.append(plt.Line2D([0], [0], color='#333333', 
                                    linestyle='--', label=f'Trend Line (R² = {r_value**2:.3f})'))

ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Add analysis text box
analysis_text = f"""Analysis Summary:
• Point size = Review count
• {correlation_text}
• Brands: {len(brand_metrics)} analyzed
• Total reviews: {brand_metrics['Total_Reviews'].sum():,}"""

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax1.text(0.98, 0.02, analysis_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Set background
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the plot
plt.savefig('brand_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()