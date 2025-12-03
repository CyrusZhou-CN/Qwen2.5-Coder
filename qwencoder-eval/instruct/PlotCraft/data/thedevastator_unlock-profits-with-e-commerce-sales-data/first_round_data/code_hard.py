import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
amazon_df = pd.read_csv('Amazon Sale Report.csv')
pricing_df = pd.read_csv('May-2022.csv')
pl_df = pd.read_csv('P  L March 2021.csv')
intl_df = pd.read_csv('International sale Report.csv')

# Data preprocessing
# Clean and convert numeric columns
def clean_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean pricing data
pricing_cols = ['Ajio MRP', 'Amazon MRP', 'Flipkart MRP', 'Myntra MRP', 'Paytm MRP', 'TP']
pricing_df = clean_numeric(pricing_df, pricing_cols)

# Clean P&L data
pl_cols = ['TP 1', 'TP 2', 'Ajio MRP', 'Amazon MRP', 'Flipkart MRP', 'Myntra MRP', 'Paytm MRP']
pl_df = clean_numeric(pl_df, pl_cols)

# Clean Amazon data
amazon_df = clean_numeric(amazon_df, ['Amount', 'Qty'])
amazon_df = amazon_df[amazon_df['Status'] != 'Cancelled'].copy()

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
deviation_colors = ['#D32F2F', '#1976D2']  # Red for negative, Blue for positive

# Row 1: Platform Pricing Analysis

# Subplot 1: Diverging bar chart - MRP deviations from average
ax1 = plt.subplot(3, 3, 1)
platforms = ['Ajio MRP', 'Amazon MRP', 'Flipkart MRP', 'Myntra MRP', 'Paytm MRP']
platform_means = []
platform_stds = []

for platform in platforms:
    values = pricing_df[platform].dropna()
    if len(values) > 0:
        platform_means.append(values.mean())
        platform_stds.append(values.std())
    else:
        platform_means.append(0)
        platform_stds.append(0)

overall_mean = np.mean(platform_means)
deviations = [mean - overall_mean for mean in platform_means]

# Create diverging bars
bars = ax1.barh(range(len(platforms)), deviations, 
                color=[deviation_colors[0] if x < 0 else deviation_colors[1] for x in deviations])

# Add error bars
ax1.errorbar(deviations, range(len(platforms)), xerr=platform_stds, 
             fmt='none', color='black', capsize=5, alpha=0.7)

ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
ax1.set_yticks(range(len(platforms)))
ax1.set_yticklabels([p.replace(' MRP', '') for p in platforms])
ax1.set_xlabel('MRP Deviation from Average (₹)')
ax1.set_title('Platform MRP Deviations with Variance', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Subplot 2: Violin plot with scatter points
ax2 = plt.subplot(3, 3, 2)
platform_data = []
platform_labels = []

for i, platform in enumerate(platforms):
    values = pricing_df[platform].dropna()
    if len(values) > 10:  # Only include platforms with sufficient data
        platform_data.append(values)
        platform_labels.append(platform.replace(' MRP', ''))

if len(platform_data) > 0:
    # Create violin plot
    parts = ax2.violinplot(platform_data, positions=range(len(platform_data)), 
                           showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    # Add scatter points for outliers
    for i, data in enumerate(platform_data):
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        outliers = data[(data < q25 - 1.5*iqr) | (data > q75 + 1.5*iqr)]
        if len(outliers) > 0:
            ax2.scatter([i]*len(outliers), outliers, alpha=0.6, s=20, color='red')

    ax2.set_xticks(range(len(platform_labels)))
    ax2.set_xticklabels(platform_labels, rotation=45)
    ax2.set_ylabel('MRP (₹)')
    ax2.set_title('MRP Distribution Across Platforms', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Insufficient data for violin plot', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('MRP Distribution Across Platforms', fontweight='bold', fontsize=12)

# Subplot 3: Dumbbell plot with secondary axis
ax3 = plt.subplot(3, 3, 3)
ax3_twin = ax3.twinx()

# Sample data for TP1 vs TP2 comparison
tp_data = pl_df[['TP 1', 'TP 2', 'Category']].dropna()
tp_data = tp_data.head(10)  # Limit for visibility

if len(tp_data) > 0:
    y_pos = range(len(tp_data))
    tp1_vals = pd.to_numeric(tp_data['TP 1'], errors='coerce')
    tp2_vals = pd.to_numeric(tp_data['TP 2'], errors='coerce')

    # Create dumbbell plot
    for i, (tp1, tp2) in enumerate(zip(tp1_vals, tp2_vals)):
        if not (np.isnan(tp1) or np.isnan(tp2)):
            ax3.plot([tp1, tp2], [i, i], 'o-', color=colors[0], linewidth=2, markersize=6)

    ax3.scatter(tp1_vals, y_pos, color=colors[1], s=60, label='TP1', zorder=5)
    ax3.scatter(tp2_vals, y_pos, color=colors[2], s=60, label='TP2', zorder=5)

    # Add profit margin line on secondary axis
    profit_margins = ((tp1_vals - tp2_vals) / tp1_vals * 100).fillna(0)
    target_margin = 25  # Assumed target margin
    margin_deviations = profit_margins - target_margin

    ax3_twin.plot(range(len(margin_deviations)), margin_deviations, 
                  color=colors[3], linewidth=2, marker='s', label='Margin Deviation')
    ax3_twin.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f'SKU {i+1}' for i in range(len(tp_data))])
    ax3.set_xlabel('Cost (₹)')
    ax3.set_title('TP1 vs TP2 Cost Comparison', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left')
    ax3_twin.set_ylabel('Margin Deviation (%)')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No TP data available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('TP1 vs TP2 Cost Comparison', fontweight='bold', fontsize=12)

# Row 2: Sales Performance Deviations

# Subplot 4: Diverging lollipop chart with area chart
ax4 = plt.subplot(3, 3, 4)

# Calculate quantity deviations by category
category_sales = amazon_df.groupby('Category')['Qty'].sum().head(8)
if len(category_sales) > 0:
    expected_sales = category_sales.mean()
    qty_deviations = category_sales - expected_sales

    # Create lollipop chart
    for i, (cat, dev) in enumerate(qty_deviations.items()):
        color = deviation_colors[0] if dev < 0 else deviation_colors[1]
        ax4.plot([0, dev], [i, i], color=color, linewidth=2)
        ax4.scatter(dev, i, color=color, s=80, zorder=5)

    # Add background area chart (cumulative trend)
    cumulative = np.cumsum(category_sales.values)
    ax4_bg = ax4.twinx()
    ax4_bg.fill_between(range(len(cumulative)), cumulative, alpha=0.2, color=colors[0])

    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_yticks(range(len(qty_deviations)))
    ax4.set_yticklabels(qty_deviations.index, fontsize=10)
    ax4.set_xlabel('Quantity Deviation from Expected')
    ax4.set_title('Sales Quantity Deviations by Category', fontweight='bold', fontsize=12)
    ax4_bg.set_ylabel('Cumulative Sales')
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No sales data available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Sales Quantity Deviations by Category', fontweight='bold', fontsize=12)

# Subplot 5: Slope chart with histogram
ax5 = plt.subplot(3, 3, 5)

# Sample Amazon sales data
amazon_sample = amazon_df[['Amount', 'Category']].dropna().head(20)
if len(amazon_sample) > 0:
    amazon_sample['MRP_estimate'] = amazon_sample['Amount'] * 1.3  # Estimated MRP

    # Create slope chart
    for i in range(len(amazon_sample)):
        ax5.plot([0, 1], [amazon_sample.iloc[i]['Amount'], amazon_sample.iloc[i]['MRP_estimate']], 
                 color=colors[i % len(colors)], alpha=0.6, linewidth=1)

    ax5.set_xlim(-0.1, 1.1)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['Sale Amount', 'Est. MRP'])
    ax5.set_ylabel('Amount (₹)')
    ax5.set_title('Sale Amount to MRP Comparison', fontweight='bold', fontsize=12)

    # Add histogram on the right
    ax5_hist = ax5.twinx()
    price_ratios = amazon_sample['Amount'] / amazon_sample['MRP_estimate']
    ax5_hist.hist(price_ratios, bins=10, alpha=0.3, color=colors[2], orientation='horizontal')
    ax5_hist.set_xlabel('Frequency')
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'No Amazon sales data', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Sale Amount to MRP Comparison', fontweight='bold', fontsize=12)

# Subplot 6: Radar chart
ax6 = plt.subplot(3, 3, 6, projection='polar')

# Prepare radar chart data
categories = amazon_df['Category'].value_counts().head(5).index
if len(categories) > 0:
    metrics = ['Sales Volume', 'Avg Price', 'Quantity']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for i, cat in enumerate(categories):
        cat_data = amazon_df[amazon_df['Category'] == cat]
        if len(cat_data) > 0:
            values = [
                cat_data['Qty'].sum() / 100,  # Normalized sales volume
                cat_data['Amount'].mean() / 100,  # Normalized average price
                len(cat_data) / 100  # Normalized quantity count
            ]
            values += values[:1]  # Complete the circle
            
            ax6.plot(angles, values, 'o-', linewidth=2, label=cat, color=colors[i])
            ax6.fill(angles, values, alpha=0.1, color=colors[i])

    # Add reference lines
    reference_values = [5, 5, 5, 5]  # Target benchmarks
    ax6.plot(angles, reference_values, 'r--', linewidth=2, alpha=0.7, label='Target')

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)
    ax6.set_title('Multi-dimensional Performance by Category', fontweight='bold', fontsize=12, pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)
else:
    ax6.text(0.5, 0.5, 'No category data', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Multi-dimensional Performance by Category', fontweight='bold', fontsize=12)

# Row 3: Cost Structure and Profitability Deviations

# Subplot 7: Stacked diverging bar chart
ax7 = plt.subplot(3, 3, 7)

# Sample cost component data
cost_categories = pl_df['Category'].value_counts().head(6).index
if len(cost_categories) > 0:
    tp1_deviations = []
    tp2_deviations = []

    for cat in cost_categories:
        cat_data = pl_df[pl_df['Category'] == cat]
        tp1_mean = pd.to_numeric(cat_data['TP 1'], errors='coerce').mean()
        tp2_mean = pd.to_numeric(cat_data['TP 2'], errors='coerce').mean()
        
        # Assume industry standards
        tp1_standard = 500
        tp2_standard = 400
        
        tp1_deviations.append(tp1_mean - tp1_standard if not np.isnan(tp1_mean) else 0)
        tp2_deviations.append(tp2_mean - tp2_standard if not np.isnan(tp2_mean) else 0)

    y_pos = range(len(cost_categories))
    ax7.barh(y_pos, tp1_deviations, height=0.4, label='TP1 Deviation', 
             color=colors[0], alpha=0.8)
    ax7.barh([y + 0.4 for y in y_pos], tp2_deviations, height=0.4, label='TP2 Deviation', 
             color=colors[1], alpha=0.8)

    ax7.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax7.set_yticks([y + 0.2 for y in y_pos])
    ax7.set_yticklabels(cost_categories)
    ax7.set_xlabel('Cost Deviation from Standard (₹)')
    ax7.set_title('Cost Component Deviations', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'No cost data available', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Cost Component Deviations', fontweight='bold', fontsize=12)

# Subplot 8: Bubble plot
ax8 = plt.subplot(3, 3, 8)

# Prepare bubble plot data
bubble_data = amazon_df[['Amount', 'Qty', 'Fulfilment', 'Category']].dropna().head(50)
if len(bubble_data) > 0:
    bubble_data['MRP_deviation'] = np.random.normal(0, 100, len(bubble_data))  # Simulated MRP deviation
    bubble_data['Sales_deviation'] = bubble_data['Qty'] - bubble_data['Qty'].mean()
    bubble_data['Profit'] = bubble_data['Amount'] * 0.2  # Assumed profit margin

    # Create bubble plot
    fulfillment_types = bubble_data['Fulfilment'].unique()
    for i, fulfillment in enumerate(fulfillment_types):
        data = bubble_data[bubble_data['Fulfilment'] == fulfillment]
        ax8.scatter(data['MRP_deviation'], data['Sales_deviation'], 
                   s=data['Profit']*2, alpha=0.6, color=colors[i % len(colors)], 
                   label=fulfillment)

    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('MRP Deviation from Cost Price (₹)')
    ax8.set_ylabel('Sales Volume Deviation from Average')
    ax8.set_title('Profitability Analysis by Fulfillment', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'No fulfillment data', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Profitability Analysis by Fulfillment', fontweight='bold', fontsize=12)

# Subplot 9: Combination chart with box plot - FIXED VERSION
ax9 = plt.subplot(3, 3, 9)
ax9_box = ax9.twinx()

# Sample profit margin data by SKU
sku_sample = amazon_df.groupby('SKU')['Amount'].agg(['mean', 'count']).head(10)
if len(sku_sample) > 0:
    sku_sample['profit_margin'] = np.random.normal(20, 5, len(sku_sample))  # Simulated margins
    target_margin = 25
    margin_deviations = sku_sample['profit_margin'] - target_margin

    # Create diverging bars
    bars = ax9.bar(range(len(margin_deviations)), margin_deviations,
                   color=[deviation_colors[0] if x < 0 else deviation_colors[1] for x in margin_deviations],
                   alpha=0.7)

    # Add box plot for margin distribution by sales channel - FIXED
    channel_margins = []
    channels = amazon_df['Sales Channel '].unique()[:3]  # Limit to 3 channels
    valid_channels = []
    
    for channel in channels:
        channel_data = amazon_df[amazon_df['Sales Channel '] == channel]['Amount']
        if len(channel_data) > 0:
            # Simulate profit margins
            margins = np.random.normal(22, 4, min(len(channel_data), 100))
            channel_margins.append(margins)
            valid_channels.append(channel)

    if len(channel_margins) > 0:
        # Create positions that match the number of channel_margins
        positions = np.linspace(2, 8, len(channel_margins))
        
        # Create box plot with matching positions
        box_plot = ax9_box.boxplot(channel_margins, positions=positions, widths=1.0, 
                                  patch_artist=True)

        # Apply colors to box patches
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.5)

    ax9.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax9.set_xticks(range(len(margin_deviations)))
    ax9.set_xticklabels([f'SKU{i+1}' for i in range(len(margin_deviations))], rotation=45)
    ax9.set_ylabel('Profit Margin Deviation (%)')
    ax9.set_title('Profit Margin Analysis by SKU & Channel', fontweight='bold', fontsize=12)
    ax9_box.set_ylabel('Margin Distribution by Channel')
    ax9.grid(True, alpha=0.3)
else:
    ax9.text(0.5, 0.5, 'No SKU data available', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Profit Margin Analysis by SKU & Channel', fontweight='bold', fontsize=12)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('comprehensive_ecommerce_analysis.png', dpi=300, bbox_inches='tight')
plt.show()