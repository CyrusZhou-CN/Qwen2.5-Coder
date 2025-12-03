import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
amazon_df = pd.read_csv('Amazon Sale Report.csv')
may_df = pd.read_csv('May-2022.csv')
march_df = pd.read_csv('P  L March 2021.csv')

# Data preprocessing
# Convert price columns to numeric, handling non-numeric values
price_columns = ['Amazon MRP', 'Myntra MRP', 'Final MRP Old', 'Ajio MRP', 'Flipkart MRP', 'Paytm MRP', 'Snapdeal MRP']
for col in price_columns:
    if col in may_df.columns:
        may_df[col] = pd.to_numeric(may_df[col], errors='coerce')
    if col in march_df.columns:
        march_df[col] = pd.to_numeric(march_df[col], errors='coerce')

# Clean Amazon sales data
amazon_df['Amount'] = pd.to_numeric(amazon_df['Amount'], errors='coerce')
amazon_df = amazon_df.dropna(subset=['Amount'])

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Subplot 1: Amazon MRP deviation from average MRP by category (Diverging Bar Chart)
# Calculate average MRP across all platforms for each category
platform_cols = ['Amazon MRP', 'Myntra MRP', 'Ajio MRP', 'Flipkart MRP', 'Paytm MRP', 'Snapdeal MRP']
available_cols = [col for col in platform_cols if col in may_df.columns]

may_df['Average_MRP'] = may_df[available_cols].mean(axis=1, skipna=True)
category_deviations = may_df.groupby('Category').agg({
    'Amazon MRP': 'mean',
    'Average_MRP': 'mean'
}).reset_index()
category_deviations['Deviation'] = category_deviations['Amazon MRP'] - category_deviations['Average_MRP']
category_deviations = category_deviations.sort_values('Deviation')

colors1 = ['#e74c3c' if x < 0 else '#2ecc71' for x in category_deviations['Deviation']]
bars1 = ax1.barh(category_deviations['Category'], category_deviations['Deviation'], color=colors1, alpha=0.8)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Deviation from Average MRP (₹)', fontweight='bold')
ax1.set_ylabel('Product Category', fontweight='bold')
ax1.set_title('Amazon MRP Deviation from Platform Average by Category', fontweight='bold', fontsize=14)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Subplot 2: Myntra MRP deviation from Final MRP Old (Diverging Lollipop Chart)
myntra_deviations = may_df.groupby('Category').agg({
    'Myntra MRP': 'mean',
    'Final MRP Old': 'mean'
}).reset_index()
myntra_deviations['Deviation'] = myntra_deviations['Myntra MRP'] - myntra_deviations['Final MRP Old']
myntra_deviations = myntra_deviations.sort_values('Deviation')

colors2 = ['#e74c3c' if x < 0 else '#3498db' for x in myntra_deviations['Deviation']]
ax2.hlines(y=range(len(myntra_deviations)), xmin=0, xmax=myntra_deviations['Deviation'], 
           colors=colors2, alpha=0.8, linewidth=3)
ax2.scatter(myntra_deviations['Deviation'], range(len(myntra_deviations)), 
           c=colors2, s=100, alpha=0.9, zorder=3)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax2.set_yticks(range(len(myntra_deviations)))
ax2.set_yticklabels(myntra_deviations['Category'])
ax2.set_xlabel('Deviation from Final MRP Old (₹)', fontweight='bold')
ax2.set_ylabel('Product Category', fontweight='bold')
ax2.set_title('Myntra MRP Deviation from Final MRP Old Baseline', fontweight='bold', fontsize=14)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Subplot 3: Price range (min-max) across platforms by Style ID (Dumbbell Plot)
# Calculate min and max MRP for each Style ID - Fixed approach
style_groups = may_df.groupby('Style Id')
min_prices = []
max_prices = []
style_ids = []

for style_id, group in style_groups:
    # Get all price values for this style across platforms
    price_values = []
    for col in available_cols:
        values = group[col].dropna()
        if len(values) > 0:
            price_values.extend(values.tolist())
    
    if price_values:  # Only include if we have price data
        min_prices.append(min(price_values))
        max_prices.append(max(price_values))
        style_ids.append(style_id)

# Create DataFrame for price ranges
style_price_range = pd.DataFrame({
    'Style Id': style_ids,
    'Min_MRP': min_prices,
    'Max_MRP': max_prices
})
style_price_range['Price_Spread'] = style_price_range['Max_MRP'] - style_price_range['Min_MRP']
style_price_range = style_price_range.sort_values('Price_Spread', ascending=False).head(15)

y_pos = range(len(style_price_range))
ax3.hlines(y=y_pos, xmin=style_price_range['Min_MRP'], xmax=style_price_range['Max_MRP'], 
           colors='#95a5a6', alpha=0.8, linewidth=2)
ax3.scatter(style_price_range['Min_MRP'], y_pos, c='#e74c3c', s=80, alpha=0.9, label='Min MRP', zorder=3)
ax3.scatter(style_price_range['Max_MRP'], y_pos, c='#2ecc71', s=80, alpha=0.9, label='Max MRP', zorder=3)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(style_price_range['Style Id'], fontsize=9)
ax3.set_xlabel('MRP (₹)', fontweight='bold')
ax3.set_ylabel('Style ID', fontweight='bold')
ax3.set_title('Price Range Across Platforms by Style ID', fontweight='bold', fontsize=14)
ax3.legend(loc='lower right')
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Subplot 4: Sales amount deviation from expected (based on MRP) by order status (Area Chart)
# Create a simplified mapping between Amazon styles and pricing data
# Extract base style from Amazon Style column
amazon_df['Style_Base'] = amazon_df['Style'].str.extract(r'([A-Za-z]+\d+)')[0]

# Create a mapping from May data
style_mrp_map = may_df.groupby('Style Id')['Amazon MRP'].mean().to_dict()

# Map expected amounts
amazon_df['Expected_Amount'] = amazon_df['Style_Base'].map(style_mrp_map)

# Filter out rows without expected amounts and calculate deviation
amazon_with_expected = amazon_df.dropna(subset=['Expected_Amount']).copy()
amazon_with_expected['Amount_Deviation'] = amazon_with_expected['Amount'] - amazon_with_expected['Expected_Amount']

# Group by status and calculate statistics
if len(amazon_with_expected) > 0:
    status_stats = amazon_with_expected.groupby('Status').agg({
        'Amount_Deviation': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    status_stats.columns = ['Status', 'Mean_Deviation', 'Std_Deviation', 'Count']
    
    # Calculate confidence intervals
    status_stats['SE'] = status_stats['Std_Deviation'] / np.sqrt(status_stats['Count'])
    status_stats['CI_Lower'] = status_stats['Mean_Deviation'] - 1.96 * status_stats['SE']
    status_stats['CI_Upper'] = status_stats['Mean_Deviation'] + 1.96 * status_stats['SE']
    
    # Handle NaN values in confidence intervals
    status_stats['CI_Lower'] = status_stats['CI_Lower'].fillna(status_stats['Mean_Deviation'])
    status_stats['CI_Upper'] = status_stats['CI_Upper'].fillna(status_stats['Mean_Deviation'])
    
    x_pos = range(len(status_stats))
    ax4.fill_between(x_pos, status_stats['CI_Lower'], status_stats['CI_Upper'], 
                     alpha=0.3, color='#3498db', label='95% Confidence Interval')
    ax4.plot(x_pos, status_stats['Mean_Deviation'], color='#2c3e50', linewidth=3, 
             marker='o', markersize=8, label='Mean Deviation')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(status_stats['Status'], rotation=45, ha='right')
    ax4.set_xlabel('Order Status', fontweight='bold')
    ax4.set_ylabel('Amount Deviation from Expected (₹)', fontweight='bold')
    ax4.set_title('Sales Amount Deviation from Expected by Order Status', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(alpha=0.3, linestyle='--')
else:
    # If no matching data, create a placeholder
    ax4.text(0.5, 0.5, 'No matching data found\nbetween Amazon sales and pricing data', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Sales Amount Deviation from Expected by Order Status', fontweight='bold', fontsize=14)

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.savefig('pricing_deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()