import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load data
amazon_df = pd.read_csv('Amazon Sale Report.csv')
may_df = pd.read_csv('May-2022.csv')
march_df = pd.read_csv('P  L March 2021.csv')

# Data preprocessing
def clean_numeric_column(df, col):
    """Clean and convert column to numeric"""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df

# Clean pricing columns for both datasets
price_columns = ['MRP Old', 'Final MRP Old', 'Ajio MRP', 'Amazon MRP', 'Amazon FBA MRP', 
                'Flipkart MRP', 'Limeroad MRP', 'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP']

for col in price_columns:
    may_df = clean_numeric_column(may_df, col)
    march_df = clean_numeric_column(march_df, col)

# Clean TP columns
march_df = clean_numeric_column(march_df, 'TP 1')
march_df = clean_numeric_column(march_df, 'TP 2')

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color scheme
positive_color = '#2E8B57'  # Sea green
negative_color = '#DC143C'  # Crimson
neutral_color = '#4682B4'   # Steel blue

# Subplot 1: Diverging bar chart - Platform MRP deviation from average
ax1 = plt.subplot(3, 2, 1)

# Calculate average MRP across platforms for May data
platform_cols = ['Ajio MRP', 'Amazon MRP', 'Flipkart MRP', 'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP']
may_clean = may_df.dropna(subset=platform_cols)

platform_means = {}
platform_stds = {}
overall_mean = 0

for col in platform_cols:
    platform_means[col] = may_clean[col].mean()
    platform_stds[col] = may_clean[col].std()
    overall_mean += platform_means[col]

overall_mean = overall_mean / len(platform_cols)

# Calculate deviations
deviations = {col: platform_means[col] - overall_mean for col in platform_cols}
platform_names = [col.replace(' MRP', '') for col in platform_cols]

# Create diverging bar chart
y_pos = np.arange(len(platform_names))
colors = [positive_color if dev > 0 else negative_color for dev in deviations.values()]

bars = ax1.barh(y_pos, list(deviations.values()), color=colors, alpha=0.7)

# Add error bars
ax1.errorbar(list(deviations.values()), y_pos, 
            xerr=[platform_stds[col] for col in platform_cols],
            fmt='none', color='black', capsize=5, capthick=2)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(platform_names, fontweight='bold')
ax1.set_xlabel('Deviation from Average MRP (₹)', fontweight='bold')
ax1.set_title('Platform MRP Deviations from Average\nwith Standard Deviation Error Bars', 
              fontweight='bold', fontsize=14, pad=20)
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax1.grid(True, alpha=0.3)

# Subplot 2: Dumbbell plot with secondary axis
ax2 = plt.subplot(3, 2, 2)

# Get top categories and calculate TP1 vs TP2 comparison
top_categories = march_df['Category'].value_counts().head(6).index
category_data = march_df[march_df['Category'].isin(top_categories)].groupby('Category').agg({
    'TP 1': 'mean',
    'TP 2': 'mean',
    'Amazon MRP': 'mean'
}).reset_index()

# Calculate profit margins
category_data['margin_tp1'] = ((category_data['Amazon MRP'] - category_data['TP 1']) / category_data['Amazon MRP'] * 100)
category_data['margin_tp2'] = ((category_data['Amazon MRP'] - category_data['TP 2']) / category_data['Amazon MRP'] * 100)
overall_margin = category_data['margin_tp1'].mean()

y_pos = np.arange(len(category_data))

# Create dumbbell plot
for i, row in category_data.iterrows():
    ax2.plot([row['TP 1'], row['TP 2']], [i, i], 'o-', color=neutral_color, linewidth=3, markersize=8)

ax2.scatter(category_data['TP 1'], y_pos, color=positive_color, s=100, label='TP1', zorder=5)
ax2.scatter(category_data['TP 2'], y_pos, color=negative_color, s=100, label='TP2', zorder=5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(category_data['Category'], fontweight='bold')
ax2.set_xlabel('Cost (₹)', fontweight='bold')
ax2.set_title('TP1 vs TP2 Cost Comparison by Category\nwith Profit Margin Deviations', 
              fontweight='bold', fontsize=14, pad=20)

# Secondary y-axis for profit margin deviation
ax2_twin = ax2.twinx()
margin_deviations = category_data['margin_tp1'] - overall_margin
ax2_twin.plot(range(len(margin_deviations)), margin_deviations, 
              color='orange', linewidth=3, marker='s', markersize=8, label='Margin Deviation')
ax2_twin.set_ylabel('Profit Margin Deviation (%)', fontweight='bold', color='orange')
ax2_twin.tick_params(axis='y', labelcolor='orange')

ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Subplot 3: Slope chart for top 10 style IDs
ax3 = plt.subplot(3, 2, 3)

# Get top 10 style IDs by frequency
top_styles = may_df['Style Id'].value_counts().head(10).index
style_data = may_df[may_df['Style Id'].isin(top_styles)].groupby('Style Id').agg({
    'MRP Old': 'mean',
    'Final MRP Old': 'mean'
}).reset_index()

# Calculate price adjustment magnitude
style_data['adjustment'] = abs(style_data['Final MRP Old'] - style_data['MRP Old'])

y_pos = np.arange(len(style_data))

# Create slope chart
for i, row in style_data.iterrows():
    color = positive_color if row['Final MRP Old'] > row['MRP Old'] else negative_color
    ax3.plot([row['MRP Old'], row['Final MRP Old']], [i, i], 
             color=color, linewidth=2, alpha=0.7)

# Add scatter points for magnitude
scatter = ax3.scatter(style_data['MRP Old'], y_pos, 
                     s=style_data['adjustment']/10, color='blue', alpha=0.6, label='Old MRP')
ax3.scatter(style_data['Final MRP Old'], y_pos, 
           s=style_data['adjustment']/10, color='red', alpha=0.6, label='Final MRP')

ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"Style {i+1}" for i in range(len(style_data))], fontweight='bold')
ax3.set_xlabel('MRP (₹)', fontweight='bold')
ax3.set_title('MRP Changes for Top 10 Style IDs\nwith Price Adjustment Magnitude', 
              fontweight='bold', fontsize=14, pad=20)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Diverging lollipop chart - Platform deviation from Amazon baseline
ax4 = plt.subplot(3, 2, (4, 5))

# Calculate deviations from Amazon MRP
amazon_baseline = may_clean['Amazon MRP'].mean()
platform_deviations = {}

for col in platform_cols:
    if col != 'Amazon MRP':
        platform_deviations[col] = ((may_clean[col].mean() - amazon_baseline) / amazon_baseline * 100)

platform_names_clean = [col.replace(' MRP', '') for col in platform_deviations.keys()]
deviations_pct = list(platform_deviations.values())

# Create lollipop chart
y_pos = np.arange(len(platform_names_clean))
colors = [positive_color if dev > 0 else negative_color for dev in deviations_pct]

# Stems
for i, (pos, dev) in enumerate(zip(y_pos, deviations_pct)):
    ax4.plot([0, dev], [pos, pos], color=colors[i], linewidth=3, alpha=0.7)

# Lollipops
ax4.scatter(deviations_pct, y_pos, color=colors, s=150, zorder=5)

# Reference lines
ax4.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='+10%')
ax4.axvline(x=-10, color='orange', linestyle='--', alpha=0.7, label='-10%')
ax4.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='+20%')
ax4.axvline(x=-20, color='red', linestyle='--', alpha=0.7, label='-20%')
ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)

ax4.set_yticks(y_pos)
ax4.set_yticklabels(platform_names_clean, fontweight='bold')
ax4.set_xlabel('Deviation from Amazon Baseline (%)', fontweight='bold')
ax4.set_title('Platform MRP Deviations from Amazon Baseline\nwith ±10% and ±20% Thresholds', 
              fontweight='bold', fontsize=14, pad=20)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Subplot 5: Radar chart for top 5 categories
ax5 = plt.subplot(3, 2, 6, projection='polar')

# Get top 5 categories
top_5_categories = march_df['Category'].value_counts().head(5).index
radar_data = march_df[march_df['Category'].isin(top_5_categories)].groupby('Category').agg({
    'TP 1': 'mean',
    'TP 2': 'mean',
    'Amazon MRP': 'mean',
    'Flipkart MRP': 'mean',
    'Myntra MRP': 'mean'
}).reset_index()

# Normalize data (0-1 scale)
metrics = ['TP 1', 'TP 2', 'Amazon MRP', 'Flipkart MRP', 'Myntra MRP']
for metric in metrics:
    max_val = radar_data[metric].max()
    min_val = radar_data[metric].min()
    radar_data[f'{metric}_norm'] = (radar_data[metric] - min_val) / (max_val - min_val)

# Set up radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, category in enumerate(top_5_categories):
    values = [radar_data[radar_data['Category'] == category][f'{metric}_norm'].iloc[0] for metric in metrics]
    values += values[:1]  # Complete the circle
    
    ax5.plot(angles, values, 'o-', linewidth=2, label=category, color=colors_radar[i])
    ax5.fill(angles, values, alpha=0.25, color=colors_radar[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(metrics, fontweight='bold')
ax5.set_ylim(0, 1)
ax5.set_title('Normalized Pricing Metrics Comparison\nTop 5 Categories', 
              fontweight='bold', fontsize=14, pad=30)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax5.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.suptitle('E-commerce Platform Pricing Analysis: Deviations and Profit Margins', 
             fontsize=18, fontweight='bold', y=0.98)

plt.show()