import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
amazon_df = pd.read_csv('Amazon Sale Report.csv')
may_df = pd.read_csv('May-2022.csv')

# Clean and prepare Amazon sales data
# Remove cancelled orders and handle missing amounts
amazon_clean = amazon_df[amazon_df['Status'] != 'Cancelled'].copy()
amazon_clean = amazon_clean.dropna(subset=['Amount', 'Category'])

# Calculate total sales by category
sales_by_category = amazon_clean.groupby('Category')['Amount'].sum().reset_index()
sales_by_category = sales_by_category.sort_values('Amount', ascending=False)

# Get top 10 categories
top_10_categories = sales_by_category.head(10)

# Prepare MRP data - calculate average MRP across all platforms
mrp_columns = ['Ajio MRP', 'Amazon MRP', 'Amazon FBA MRP', 'Flipkart MRP', 
               'Limeroad MRP', 'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP']

# Convert MRP columns to numeric, handling any non-numeric values
for col in mrp_columns:
    may_df[col] = pd.to_numeric(may_df[col], errors='coerce')

# Calculate average MRP across platforms for each product
may_df['Avg_MRP'] = may_df[mrp_columns].mean(axis=1, skipna=True)

# Calculate average MRP by category
avg_mrp_by_category = may_df.groupby('Category')['Avg_MRP'].mean().reset_index()

# Merge sales data with MRP data
merged_data = pd.merge(top_10_categories, avg_mrp_by_category, on='Category', how='left')

# Fill any missing MRP values with overall average
overall_avg_mrp = may_df['Avg_MRP'].mean()
merged_data['Avg_MRP'].fillna(overall_avg_mrp, inplace=True)

# Create horizontal bar chart with white background
plt.figure(figsize=(14, 10))
plt.style.use('default')  # Ensure white background

# Create the horizontal bar chart
bars = plt.barh(range(len(merged_data)), merged_data['Amount'], 
                color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
                       '#577590', '#F8961E', '#90323D', '#4D5382', '#8B5A3C'])

# Customize the chart
plt.xlabel('Total Sales Amount (INR)', fontsize=12, fontweight='bold')
plt.ylabel('Product Category', fontsize=12, fontweight='bold')
plt.title('Top 10 Product Categories by Total Sales Amount\nwith Average MRP Context', 
          fontsize=16, fontweight='bold', pad=20)

# Set category labels
plt.yticks(range(len(merged_data)), merged_data['Category'], fontsize=11)

# Add value labels on bars and MRP annotations
for i, (sales, mrp) in enumerate(zip(merged_data['Amount'], merged_data['Avg_MRP'])):
    # Sales amount label
    plt.text(sales + sales * 0.01, i, f'₹{sales:,.0f}', 
             va='center', ha='left', fontsize=10, fontweight='bold')
    
    # MRP annotation
    plt.text(sales * 0.5, i, f'Avg MRP: ₹{mrp:,.0f}', 
             va='center', ha='center', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

# Format x-axis to show values in millions/thousands
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000000:.1f}M' if x >= 1000000 else f'₹{x/1000:.0f}K'))

# Add subtle gridlines
plt.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Invert y-axis to show highest sales at top
plt.gca().invert_yaxis()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()