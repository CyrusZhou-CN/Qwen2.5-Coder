import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from math import pi

# Load and combine data
intel_df = pd.read_csv('intel_cpus.csv')
amd_df = pd.read_csv('amd_cpus.csv')

# Clean price data - remove currency symbols and convert to numeric
def clean_price(price_str):
    if isinstance(price_str, str):
        # Remove currency symbols and commas
        cleaned = price_str.replace('₹', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except:
            return np.nan
    return price_str

intel_df['price_clean'] = intel_df['MRP'].apply(clean_price)
amd_df['price_clean'] = amd_df['MRP'].apply(clean_price)

# Add component type
intel_df['component_type'] = 'Intel CPU'
amd_df['component_type'] = 'AMD CPU'
intel_df['brand'] = 'Intel'
amd_df['brand'] = 'AMD'

# Combine datasets
combined_df = pd.concat([intel_df, amd_df], ignore_index=True)
combined_df = combined_df.dropna(subset=['price_clean'])

# Create price brackets
def categorize_price(price):
    if price < 5000:
        return 'Budget'
    elif price < 20000:
        return 'Mid-range'
    else:
        return 'Premium'

combined_df['price_bracket'] = combined_df['price_clean'].apply(categorize_price)

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Define consistent color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']
brand_colors = {'Intel': '#0071C5', 'AMD': '#ED1C24'}

# Row 1, Subplot 1: Stacked bar + line overlay
ax1 = plt.subplot(3, 3, 1, facecolor='white')
component_stats = combined_df.groupby('component_type').agg({
    'price_clean': ['count', 'mean']
}).round(2)
component_stats.columns = ['count', 'avg_price']

# Stacked bar by price bracket
price_bracket_counts = combined_df.groupby(['component_type', 'price_bracket']).size().unstack(fill_value=0)
price_bracket_counts.plot(kind='bar', stacked=True, ax=ax1, color=['#FFB3BA', '#BAFFC9', '#BAE1FF'], alpha=0.8)

# Overlay line plot for average prices
ax1_twin = ax1.twinx()
ax1_twin.plot(range(len(component_stats)), component_stats['avg_price'], 
              color='red', marker='o', linewidth=3, markersize=8, label='Avg Price')
ax1_twin.set_ylabel('Average Price (₹)', fontweight='bold', color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')

ax1.set_title('Component Distribution by Price Brackets with Average Pricing', fontweight='bold', fontsize=12)
ax1.set_xlabel('Component Type', fontweight='bold')
ax1.set_ylabel('Product Count', fontweight='bold')
ax1.legend(title='Price Bracket', loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.tick_params(axis='x', rotation=45)

# Row 1, Subplot 2: Pie chart with inner donut
ax2 = plt.subplot(3, 3, 2, facecolor='white')
component_counts = combined_df['component_type'].value_counts()

# Outer pie chart - component types
wedges1, texts1, autotexts1 = ax2.pie(component_counts.values, labels=component_counts.index, 
                                       autopct='%1.1f%%', colors=colors[:len(component_counts)],
                                       radius=1, wedgeprops=dict(width=0.3))

# Inner donut - price brackets
price_bracket_counts_total = combined_df['price_bracket'].value_counts()
wedges2, texts2, autotexts2 = ax2.pie(price_bracket_counts_total.values, 
                                       labels=price_bracket_counts_total.index,
                                       autopct='%1.1f%%', colors=['#FFB3BA', '#BAFFC9', '#BAE1FF'],
                                       radius=0.7, wedgeprops=dict(width=0.4))

ax2.set_title('Market Share: Component Types & Price Distribution', fontweight='bold', fontsize=12)

# Row 1, Subplot 3: Treemap simulation using rectangles
ax3 = plt.subplot(3, 3, 3, facecolor='white')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Create rectangles for treemap
intel_count = len(intel_df)
amd_count = len(amd_df)
total_count = intel_count + amd_count

intel_area = (intel_count / total_count) * 100
amd_area = (amd_count / total_count) * 100

# Intel rectangle
intel_rect = Rectangle((0, 0), 10, intel_area/10, 
                      facecolor=brand_colors['Intel'], alpha=0.7, edgecolor='white', linewidth=2)
ax3.add_patch(intel_rect)
ax3.text(5, intel_area/20, f'Intel\n{intel_count} products\n₹{intel_df["price_clean"].mean():.0f} avg', 
         ha='center', va='center', fontweight='bold', color='white')

# AMD rectangle
amd_rect = Rectangle((0, intel_area/10), 10, amd_area/10, 
                    facecolor=brand_colors['AMD'], alpha=0.7, edgecolor='white', linewidth=2)
ax3.add_patch(amd_rect)
ax3.text(5, intel_area/10 + amd_area/20, f'AMD\n{amd_count} products\n₹{amd_df["price_clean"].mean():.0f} avg', 
         ha='center', va='center', fontweight='bold', color='white')

ax3.set_title('Component Market Treemap\n(Size: Count, Color: Brand)', fontweight='bold', fontsize=12)
ax3.set_xticks([])
ax3.set_yticks([])

# Row 2, Subplot 4: Grouped bar + scatter overlay
ax4 = plt.subplot(3, 3, 4, facecolor='white')
brand_price_counts = combined_df.groupby(['brand', 'price_bracket']).size().unstack(fill_value=0)
brand_price_counts.plot(kind='bar', ax=ax4, color=['#FFB3BA', '#BAFFC9', '#BAE1FF'], alpha=0.8)

# Overlay scatter points
for i, brand in enumerate(['Intel', 'AMD']):
    brand_data = combined_df[combined_df['brand'] == brand]
    x_positions = np.random.normal(i, 0.1, len(brand_data))
    ax4.scatter(x_positions, brand_data['price_clean'], alpha=0.6, s=30, color=brand_colors[brand])

ax4.set_title('Brand Competition: CPU Counts by Price Brackets', fontweight='bold', fontsize=12)
ax4.set_xlabel('Brand', fontweight='bold')
ax4.set_ylabel('Product Count', fontweight='bold')
ax4.legend(title='Price Bracket')
ax4.tick_params(axis='x', rotation=0)

# Row 2, Subplot 5: Stacked area chart
ax5 = plt.subplot(3, 3, 5, facecolor='white')
price_ranges = np.arange(0, combined_df['price_clean'].max() + 5000, 2000)
intel_hist, _ = np.histogram(intel_df['price_clean'], bins=price_ranges)
amd_hist, _ = np.histogram(amd_df['price_clean'], bins=price_ranges)

intel_cumsum = np.cumsum(intel_hist)
amd_cumsum = np.cumsum(amd_hist)

x_centers = (price_ranges[:-1] + price_ranges[1:]) / 2
ax5.fill_between(x_centers, 0, intel_cumsum, alpha=0.7, color=brand_colors['Intel'], label='Intel')
ax5.fill_between(x_centers, intel_cumsum, intel_cumsum + amd_cumsum, alpha=0.7, color=brand_colors['AMD'], label='AMD')

# Add median lines
ax5.axvline(intel_df['price_clean'].median(), color='blue', linestyle='--', linewidth=2, label='Intel Median')
ax5.axvline(amd_df['price_clean'].median(), color='red', linestyle='--', linewidth=2, label='AMD Median')

ax5.set_title('Cumulative Price Distribution with Median Lines', fontweight='bold', fontsize=12)
ax5.set_xlabel('Price (₹)', fontweight='bold')
ax5.set_ylabel('Cumulative Count', fontweight='bold')
ax5.legend()

# Row 2, Subplot 6: Bubble chart
ax6 = plt.subplot(3, 3, 6, facecolor='white')
bubble_data = combined_df.groupby('component_type').agg({
    'price_clean': ['mean', 'std', 'count']
}).round(2)
bubble_data.columns = ['avg_price', 'price_variance', 'count']

x_pos = range(len(bubble_data))
scatter = ax6.scatter(x_pos, bubble_data['avg_price'], 
                     s=bubble_data['count']*2, 
                     c=bubble_data['price_variance'], 
                     cmap='viridis', alpha=0.7, edgecolors='black')

ax6.set_xticks(x_pos)
ax6.set_xticklabels(bubble_data.index, rotation=45)
ax6.set_title('Component Analysis Bubble Chart\n(Size: Count, Color: Price Variance)', fontweight='bold', fontsize=12)
ax6.set_xlabel('Component Type', fontweight='bold')
ax6.set_ylabel('Average Price (₹)', fontweight='bold')
plt.colorbar(scatter, ax=ax6, label='Price Variance')

# Row 3, Subplot 7: Box plot + violin overlay (FIXED - removed alpha from violinplot)
ax7 = plt.subplot(3, 3, 7, facecolor='white')
price_data = [intel_df['price_clean'].dropna(), amd_df['price_clean'].dropna()]
labels = ['Intel', 'AMD']

# Box plot
bp = ax7.boxplot(price_data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], [brand_colors['Intel'], brand_colors['AMD']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Violin plot overlay (removed alpha parameter and set it on individual parts)
parts = ax7.violinplot(price_data, positions=[1, 2], showmeans=True)
for pc, color in zip(parts['bodies'], [brand_colors['Intel'], brand_colors['AMD']]):
    pc.set_facecolor(color)
    pc.set_alpha(0.3)

ax7.set_title('Price Distribution: Box Plot with Violin Overlay', fontweight='bold', fontsize=12)
ax7.set_xlabel('Brand', fontweight='bold')
ax7.set_ylabel('Price (₹)', fontweight='bold')

# Row 3, Subplot 8: Horizontal stacked bar with annotations
ax8 = plt.subplot(3, 3, 8, facecolor='white')
price_composition = combined_df.groupby(['component_type', 'price_bracket']).size().unstack(fill_value=0)
price_composition_pct = price_composition.div(price_composition.sum(axis=1), axis=0) * 100

price_composition_pct.plot(kind='barh', stacked=True, ax=ax8, 
                          color=['#FFB3BA', '#BAFFC9', '#BAE1FF'], alpha=0.8)

# Add percentage annotations
for i, (idx, row) in enumerate(price_composition_pct.iterrows()):
    cumsum = 0
    for j, (col, val) in enumerate(row.items()):
        if val > 5:  # Only annotate if percentage > 5%
            ax8.text(cumsum + val/2, i, f'{val:.1f}%', 
                    ha='center', va='center', fontweight='bold')
        cumsum += val

ax8.set_title('Price Tier Composition by Component Type', fontweight='bold', fontsize=12)
ax8.set_xlabel('Percentage (%)', fontweight='bold')
ax8.set_ylabel('Component Type', fontweight='bold')
ax8.legend(title='Price Bracket', bbox_to_anchor=(1.05, 1), loc='upper left')

# Row 3, Subplot 9: Radar chart
ax9 = plt.subplot(3, 3, 9, facecolor='white', projection='polar')

# Prepare radar chart data
categories = ['Count', 'Avg Price', 'Price Range', 'Market Share']
intel_values = [
    len(intel_df) / max(len(intel_df), len(amd_df)),
    intel_df['price_clean'].mean() / max(intel_df['price_clean'].mean(), amd_df['price_clean'].mean()),
    (intel_df['price_clean'].max() - intel_df['price_clean'].min()) / 
    max((intel_df['price_clean'].max() - intel_df['price_clean'].min()),
        (amd_df['price_clean'].max() - amd_df['price_clean'].min())),
    len(intel_df) / (len(intel_df) + len(amd_df))
]

amd_values = [
    len(amd_df) / max(len(intel_df), len(amd_df)),
    amd_df['price_clean'].mean() / max(intel_df['price_clean'].mean(), amd_df['price_clean'].mean()),
    (amd_df['price_clean'].max() - amd_df['price_clean'].min()) / 
    max((intel_df['price_clean'].max() - intel_df['price_clean'].min()),
        (amd_df['price_clean'].max() - amd_df['price_clean'].min())),
    len(amd_df) / (len(intel_df) + len(amd_df))
]

# Angles for radar chart
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]  # Complete the circle

intel_values += intel_values[:1]
amd_values += amd_values[:1]

# Plot radar chart
ax9.plot(angles, intel_values, 'o-', linewidth=2, label='Intel', color=brand_colors['Intel'])
ax9.fill(angles, intel_values, alpha=0.25, color=brand_colors['Intel'])
ax9.plot(angles, amd_values, 'o-', linewidth=2, label='AMD', color=brand_colors['AMD'])
ax9.fill(angles, amd_values, alpha=0.25, color=brand_colors['AMD'])

ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(categories)
ax9.set_ylim(0, 1)
ax9.set_title('Normalized Brand Comparison Radar Chart', fontweight='bold', fontsize=12, pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('pc_components_analysis.png', dpi=300, bbox_inches='tight')
plt.show()