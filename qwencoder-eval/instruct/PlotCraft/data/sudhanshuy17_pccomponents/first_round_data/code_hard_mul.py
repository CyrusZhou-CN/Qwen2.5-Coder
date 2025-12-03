import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Load CPU data
intel_df = pd.read_csv('intel_cpus.csv')
amd_df = pd.read_csv('amd_cpus.csv')

# Clean price data function
def clean_price(price_str):
    if isinstance(price_str, str):
        return float(price_str.replace('₹', '').replace(',', ''))
    return price_str

# Clean CPU data
intel_df['price_clean'] = intel_df['MRP'].apply(clean_price)
amd_df['price_clean'] = amd_df['MRP'].apply(clean_price)

# Create synthetic data for other components to demonstrate the visualization
np.random.seed(42)

# Generate synthetic component data
components_data = {
    'CPU': pd.concat([intel_df[['CPU', 'price_clean']], amd_df[['CPU', 'price_clean']].rename(columns={'CPU': 'CPU'})]),
    'GPU': pd.DataFrame({
        'name': [f'GPU_{i}' for i in range(50)],
        'price_clean': np.random.lognormal(9, 0.8, 50)
    }),
    'MotherBoard': pd.DataFrame({
        'name': [f'MB_{i}' for i in range(40)],
        'price_clean': np.random.lognormal(7.5, 0.6, 40)
    }),
    'PowerSupply': pd.DataFrame({
        'name': [f'PSU_{i}' for i in range(35)],
        'price_clean': np.random.lognormal(7, 0.7, 35)
    }),
    'RAM': pd.DataFrame({
        'name': [f'RAM_{i}' for i in range(60)],
        'price_clean': np.random.lognormal(7.8, 0.5, 60)
    }),
    'StorageSSD': pd.DataFrame({
        'name': [f'SSD_{i}' for i in range(45)],
        'price_clean': np.random.lognormal(7.2, 0.6, 45)
    }),
    'Cabinets': pd.DataFrame({
        'name': [f'Case_{i}' for i in range(30)],
        'price_clean': np.random.lognormal(6.5, 0.8, 30)
    })
}

# Prepare data for analysis
component_stats = {}
for comp, data in components_data.items():
    component_stats[comp] = {
        'count': len(data),
        'avg_price': data['price_clean'].mean(),
        'total_value': data['price_clean'].sum(),
        'prices': data['price_clean'].values
    }

# Set up the figure with white background
fig = plt.figure(figsize=(20, 12), facecolor='white')
fig.suptitle('PC Component Market Composition and Pricing Analysis', fontsize=20, fontweight='bold', y=0.95)

# Color palette for components
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
component_colors = dict(zip(component_stats.keys(), colors))

# Subplot 1: Stacked bar chart with line overlay
ax1 = plt.subplot(2, 3, 1)
categories = list(component_stats.keys())
counts = [component_stats[cat]['count'] for cat in categories]
avg_prices = [component_stats[cat]['avg_price'] for cat in categories]

# Create stacked bar chart
bars = ax1.bar(categories, counts, color=[component_colors[cat] for cat in categories], alpha=0.8)

# Add line plot for average prices
ax1_twin = ax1.twinx()
line = ax1_twin.plot(categories, avg_prices, color='red', marker='o', linewidth=3, markersize=8, label='Avg Price')

ax1.set_title('Component Count Distribution with Average Pricing', fontweight='bold', fontsize=12, pad=20)
ax1.set_ylabel('Component Count', fontweight='bold')
ax1_twin.set_ylabel('Average Price (₹)', fontweight='bold', color='red')
ax1.tick_params(axis='x', rotation=45)
ax1_twin.tick_params(axis='y', labelcolor='red')

# Subplot 2: Treemap with pie chart inset
ax2 = plt.subplot(2, 3, 2)

# Create treemap using rectangles
total_value = sum([component_stats[cat]['total_value'] for cat in categories])
values = [component_stats[cat]['total_value'] for cat in categories]
sizes = [val/total_value for val in values]

# Simple treemap layout
x, y = 0, 0
for i, (cat, size) in enumerate(zip(categories, sizes)):
    width = size * 0.8
    height = 0.8 / len(categories)
    rect = Rectangle((x, y), width, height, facecolor=component_colors[cat], alpha=0.7, edgecolor='white', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x + width/2, y + height/2, f'{cat}\n₹{values[i]/1000:.0f}K', 
             ha='center', va='center', fontweight='bold', fontsize=8)
    y += height

# Inset pie chart
ax2_inset = fig.add_axes([0.42, 0.65, 0.15, 0.15])
ax2_inset.pie(counts, colors=[component_colors[cat] for cat in categories], 
              startangle=90, wedgeprops=dict(width=0.5))
ax2_inset.set_title('Count %', fontsize=8, fontweight='bold')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Market Value Treemap with Count Distribution', fontweight='bold', fontsize=12, pad=20)
ax2.set_xticks([])
ax2.set_yticks([])

# Subplot 3: Waffle chart for CPU brands with box plots
ax3 = plt.subplot(2, 3, 3)

# CPU brand analysis
intel_count = len(intel_df)
amd_count = len(amd_df)
total_cpu = intel_count + amd_count

# Create waffle chart
waffle_size = 10
intel_squares = int((intel_count / total_cpu) * waffle_size * waffle_size)
amd_squares = waffle_size * waffle_size - intel_squares

# Draw waffle squares
for i in range(waffle_size):
    for j in range(waffle_size):
        square_num = i * waffle_size + j
        color = '#0071C5' if square_num < intel_squares else '#ED1C24'
        rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        ax3.add_patch(rect)

ax3.set_xlim(0, waffle_size)
ax3.set_ylim(0, waffle_size)
ax3.set_aspect('equal')
ax3.set_title('CPU Brand Distribution (Intel vs AMD)', fontweight='bold', fontsize=12, pad=20)
ax3.set_xticks([])
ax3.set_yticks([])

# Add legend
intel_patch = mpatches.Patch(color='#0071C5', label=f'Intel ({intel_count})')
amd_patch = mpatches.Patch(color='#ED1C24', label=f'AMD ({amd_count})')
ax3.legend(handles=[intel_patch, amd_patch], loc='upper right', bbox_to_anchor=(1, 1))

# Subplot 4: Stacked area chart with outliers
ax4 = plt.subplot(2, 3, 4)

# Define price ranges
def categorize_price(price):
    if price < 5000:
        return 'Budget'
    elif price <= 20000:
        return 'Mid-range'
    else:
        return 'Premium'

# Calculate price range distributions
price_ranges = {'Budget': [], 'Mid-range': [], 'Premium': []}
outliers_x, outliers_y = [], []

for i, cat in enumerate(categories):
    prices = component_stats[cat]['prices']
    budget = sum(1 for p in prices if p < 5000)
    mid_range = sum(1 for p in prices if 5000 <= p <= 20000)
    premium = sum(1 for p in prices if p > 20000)
    
    price_ranges['Budget'].append(budget)
    price_ranges['Mid-range'].append(mid_range)
    price_ranges['Premium'].append(premium)
    
    # Add outliers (top 5% of prices)
    high_prices = sorted(prices, reverse=True)[:max(1, len(prices)//20)]
    for price in high_prices:
        if price > 30000:  # High-value threshold
            outliers_x.append(i)
            outliers_y.append(price)

# Create stacked area chart
x_pos = range(len(categories))
ax4.stackplot(x_pos, price_ranges['Budget'], price_ranges['Mid-range'], price_ranges['Premium'],
              labels=['Budget (<₹5K)', 'Mid-range (₹5K-₹20K)', 'Premium (>₹20K)'],
              colors=['#90EE90', '#FFD700', '#FF6347'], alpha=0.8)

# Add outlier scatter points
if outliers_x:
    ax4.scatter(outliers_x, outliers_y, color='red', s=50, alpha=0.7, zorder=5, label='High-value outliers')

ax4.set_title('Price Range Distribution with High-Value Outliers', fontweight='bold', fontsize=12, pad=20)
ax4.set_xlabel('Component Categories', fontweight='bold')
ax4.set_ylabel('Count / Price (₹)', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(categories, rotation=45)
ax4.legend(loc='upper left')

# Subplot 5: Nested donut chart with radar overlay
ax5 = plt.subplot(2, 3, 5)

# Outer ring - component categories
outer_sizes = [component_stats[cat]['count'] for cat in categories]
outer_colors = [component_colors[cat] for cat in categories]

# Inner ring - price segments within each category
inner_labels = []
inner_sizes = []
inner_colors = []

for cat in categories:
    prices = component_stats[cat]['prices']
    budget_count = sum(1 for p in prices if p < 5000)
    mid_count = sum(1 for p in prices if 5000 <= p <= 20000)
    premium_count = sum(1 for p in prices if p > 20000)
    
    inner_sizes.extend([budget_count, mid_count, premium_count])
    inner_colors.extend(['lightgreen', 'gold', 'lightcoral'])

# Create nested donut
wedges1, texts1 = ax5.pie(outer_sizes, colors=outer_colors, radius=1, 
                          wedgeprops=dict(width=0.3, edgecolor='white'))
wedges2, texts2 = ax5.pie(inner_sizes, colors=inner_colors, radius=0.7,
                          wedgeprops=dict(width=0.3, edgecolor='white'))

ax5.set_title('Component Hierarchy: Categories & Price Segments', fontweight='bold', fontsize=12, pad=20)

# Add center text
ax5.text(0, 0, 'PC Market\nComposition', ha='center', va='center', fontweight='bold', fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()