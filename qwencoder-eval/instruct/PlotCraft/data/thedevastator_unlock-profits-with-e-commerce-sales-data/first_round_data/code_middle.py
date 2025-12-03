import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('P  L March 2021.csv')

# Data preprocessing - convert price columns to numeric
price_columns = ['Ajio MRP', 'Amazon MRP', 'Amazon FBA MRP', 'Flipkart MRP', 
                'Limeroad MRP', 'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP']

# Convert TP 1 to numeric for cost calculation
df['TP 1'] = pd.to_numeric(df['TP 1'], errors='coerce')

# Convert price columns to numeric
for col in price_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Group by Style Id and calculate averages
style_data = df.groupby('Style Id').agg({
    'TP 1': 'first',  # Take first value as cost
    **{col: 'mean' for col in price_columns}  # Average MRP across sizes
}).reset_index()

# Remove rows with missing data
style_data = style_data.dropna()

# Check if we have enough data
if len(style_data) == 0:
    print("No valid data found after preprocessing")
    exit()

# Calculate average MRP across all platforms for each style
style_data['Avg_MRP'] = style_data[price_columns].mean(axis=1)

# Calculate percentage deviations from average MRP
deviation_data = {}
for col in price_columns:
    platform_name = col.replace(' MRP', '').replace(' ', '_')
    deviation_data[platform_name] = ((style_data[col] - style_data['Avg_MRP']) / style_data['Avg_MRP'] * 100)

# Calculate cost-to-price ratio (TP/MRP)
style_data['Cost_Price_Ratio'] = style_data['TP 1'] / style_data['Avg_MRP']

# Calculate pricing variance for each style
price_variance = style_data[price_columns].std(axis=1)
variance_threshold = style_data['Avg_MRP'] * 0.1  # 10% of mean
style_data['High_Variance'] = price_variance > variance_threshold

# Select top styles for visualization (limit to available data)
num_styles = min(12, len(style_data))
top_styles = style_data.nlargest(num_styles, 'Avg_MRP').reset_index(drop=True)

# Create the composite visualization
fig, ax1 = plt.subplots(figsize=(16, 10))

# Prepare data for diverging bar chart
platforms = list(deviation_data.keys())
x_pos = np.arange(len(top_styles))
bar_width = 0.08

# Color scheme for platforms
platform_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

# Create diverging bars for each platform
for i, platform in enumerate(platforms):
    deviations = []
    for j in range(len(top_styles)):
        style_id = top_styles.iloc[j]['Style Id']
        # Find the deviation for this style using boolean indexing
        style_mask = style_data['Style Id'] == style_id
        if style_mask.any():
            style_row = style_data[style_mask].iloc[0]
            # Get the deviation value directly from the series
            deviation_series = deviation_data[platform]
            style_index = style_data[style_mask].index[0]
            deviation = deviation_series.loc[style_index]
            deviations.append(deviation)
        else:
            deviations.append(0)  # Default value if not found
    
    # Create bars
    alpha_val = 0.8 if i % 2 == 0 else 0.6
    
    bars = ax1.barh(x_pos + i * bar_width - (len(platforms) * bar_width / 2), 
                    deviations, bar_width, 
                    color=platform_colors[i % len(platform_colors)], alpha=alpha_val, 
                    label=platform.replace('_', ' '))

# Customize the bar chart
ax1.set_xlabel('Percentage Deviation from Average MRP (%)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Product Styles', fontweight='bold', fontsize=12)
ax1.set_title('E-commerce Platform Pricing Deviations and Markup Analysis', 
              fontweight='bold', fontsize=16, pad=20)

# Set y-axis labels
style_labels = []
for i in range(len(top_styles)):
    style = top_styles.iloc[i]['Style Id']
    if len(str(style)) > 12:
        style_labels.append(f"{str(style)[:12]}...")
    else:
        style_labels.append(str(style))

ax1.set_yticks(x_pos)
ax1.set_yticklabels(style_labels)

# Add vertical line at zero
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Create second y-axis for cost-to-price ratio line plot
ax2 = ax1.twinx()

# Prepare cost-to-price ratio data
cost_ratios = []
high_variance_mask = []

for j in range(len(top_styles)):
    cost_ratios.append(top_styles.iloc[j]['Cost_Price_Ratio'])
    high_variance_mask.append(top_styles.iloc[j]['High_Variance'])

# Create line plot with different colors for high/low variance
high_var_x = []
high_var_y = []
low_var_x = []
low_var_y = []

for i in range(len(cost_ratios)):
    if pd.notna(cost_ratios[i]):  # Check for valid values
        if high_variance_mask[i]:
            high_var_x.append(cost_ratios[i])
            high_var_y.append(x_pos[i])
            ax2.plot(cost_ratios[i], x_pos[i], marker='o', color='#ff6b6b', 
                     markersize=8, markeredgecolor='white', markeredgewidth=1)
        else:
            low_var_x.append(cost_ratios[i])
            low_var_y.append(x_pos[i])
            ax2.plot(cost_ratios[i], x_pos[i], marker='s', color='#4ecdc4', 
                     markersize=6, markeredgecolor='white', markeredgewidth=1)

# Connect points with lines
if high_var_x:
    ax2.plot(high_var_x, high_var_y, color='#ff6b6b', alpha=0.6, linewidth=2, 
             label='High Variance Styles (>10%)')

if low_var_x:
    ax2.plot(low_var_x, low_var_y, color='#4ecdc4', alpha=0.6, linewidth=2,
             label='Low Variance Styles (≤10%)')

ax2.set_xlabel('Cost-to-Price Ratio (TP/MRP)', fontweight='bold', fontsize=12)
if cost_ratios and any(pd.notna(cost_ratios)):
    valid_ratios = [r for r in cost_ratios if pd.notna(r)]
    if valid_ratios:
        ax2.set_xlim(0, max(valid_ratios) * 1.1)

# Add legends
legend1 = ax1.legend(title='E-commerce Platforms', loc='upper left', bbox_to_anchor=(1.15, 1), 
                     fontsize=9, title_fontsize=10)
legend1.get_title().set_fontweight('bold')

legend2 = ax2.legend(title='Pricing Variance', loc='upper left', bbox_to_anchor=(1.15, 0.7), 
                     fontsize=9, title_fontsize=10)
legend2.get_title().set_fontweight('bold')

# Add grid for better readability
ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Calculate statistics for insights
high_variance_count = sum(high_variance_mask)
valid_cost_ratios = [r for r in cost_ratios if pd.notna(r)]
avg_cost_ratio = np.mean(valid_cost_ratios) if valid_cost_ratios else 0

# Get deviation ranges
all_deviations = []
for platform in platforms:
    deviation_series = deviation_data[platform]
    for j in range(len(top_styles)):
        style_id = top_styles.iloc[j]['Style Id']
        style_mask = style_data['Style Id'] == style_id
        if style_mask.any():
            style_index = style_data[style_mask].index[0]
            deviation = deviation_series.loc[style_index]
            if pd.notna(deviation):
                all_deviations.append(deviation)

min_deviation = min(all_deviations) if all_deviations else 0
max_deviation = max(all_deviations) if all_deviations else 0

# Add text box with key insights
textstr = f'Key Insights:\n• {high_variance_count} styles show high pricing variance\n• Avg cost-to-price ratio: {avg_cost_ratio:.3f}\n• Deviation range: {min_deviation:.1f}% to {max_deviation:.1f}%'
props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(right=0.75)

# Save the plot
plt.savefig('ecommerce_pricing_deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()