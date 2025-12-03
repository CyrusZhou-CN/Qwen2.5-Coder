import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('uae_properties.csv')

# Convert addedOn to datetime
df['addedOn'] = pd.to_datetime(df['addedOn'])
df['month'] = df['addedOn'].dt.to_period('M')
df['days_since_listing'] = (datetime.now() - df['addedOn']).dt.days

# Remove outliers for better visualization (keep only reasonable price range)
df_clean = df[(df['price'] > 0) & (df['price'] <= df['price'].quantile(0.95))].copy()

# Create price categories
df_clean['price_category'] = pd.cut(df_clean['price'], 
                                   bins=[0, 2000000, 4000000, float('inf')], 
                                   labels=['Budget (<2M)', 'Mid-range (2-4M)', 'Luxury (>4M)'])

# Get top 6 locations
top_locations = df_clean['displayAddress'].value_counts().head(6).index

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.patch.set_facecolor('white')

# 1. Top-left: Price distribution histogram with KDE and quartiles
ax1 = axes[0, 0]
prices = df_clean['price']
ax1.hist(prices, bins=25, alpha=0.7, color='skyblue', density=True, edgecolor='white')

# Simple KDE approximation using numpy
hist, bin_edges = np.histogram(prices, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# Smooth the histogram for KDE-like curve
from scipy.ndimage import gaussian_filter1d
smoothed = gaussian_filter1d(hist, sigma=2)
ax1.plot(bin_centers, smoothed, color='darkblue', linewidth=2, label='Smoothed Density')

# Add quartile lines
quartiles = prices.quantile([0.25, 0.5, 0.75])
colors = ['red', 'orange', 'green']
for i, (q, color) in enumerate(zip(quartiles, colors)):
    ax1.axvline(q, color=color, linestyle='--', linewidth=2, 
                label=f'Q{i+1}: {q/1000000:.1f}M')

ax1.set_title('Price Distribution with Quartiles', fontweight='bold', fontsize=10)
ax1.set_xlabel('Price (AED)')
ax1.set_ylabel('Density')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Top-center: Bathroom count distribution with percentage labels
ax2 = axes[0, 1]
bathroom_counts = df_clean['bathrooms'].value_counts().sort_index()
bars = ax2.bar(bathroom_counts.index, bathroom_counts.values, color='lightcoral', edgecolor='white')

# Add percentage labels
total = len(df_clean)
for bar, count in zip(bars, bathroom_counts.values):
    height = bar.get_height()
    percentage = (count / total) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax2.set_title('Bathroom Count Distribution', fontweight='bold', fontsize=10)
ax2.set_xlabel('Number of Bathrooms')
ax2.set_ylabel('Count')
ax2.grid(True, alpha=0.3)

# 3. Top-right: Bedroom count pie chart
ax3 = axes[0, 2]
bedroom_counts = df_clean['bedrooms'].value_counts().head(6)  # Limit to top 6 for clarity
colors = plt.cm.Set3(np.linspace(0, 1, len(bedroom_counts)))
wedges, texts, autotexts = ax3.pie(bedroom_counts.values, labels=bedroom_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax3.set_title('Bedroom Count Distribution', fontweight='bold', fontsize=10)

# 4. Middle-left: Price vs location violin plot
ax4 = axes[1, 0]
location_data = df_clean[df_clean['displayAddress'].isin(top_locations)]
# Use box plot instead of violin plot for faster execution
box_data = [location_data[location_data['displayAddress'] == loc]['price'].values 
            for loc in top_locations]
ax4.boxplot(box_data, labels=[loc.split(',')[0][:15] + '...' if len(loc) > 15 else loc.split(',')[0] 
                              for loc in top_locations])
ax4.set_title('Price Distribution by Top Locations', fontweight='bold', fontsize=10)
ax4.set_xlabel('Location')
ax4.set_ylabel('Price (AED)')
ax4.tick_params(axis='x', rotation=45, labelsize=8)

# 5. Middle-center: Monthly listing pattern
ax5 = axes[1, 1]
monthly_counts = df_clean.groupby('month').size()
monthly_indices = range(len(monthly_counts))

bars = ax5.bar(monthly_indices, monthly_counts.values, 
               color='lightgreen', alpha=0.7, edgecolor='white')

# Add simple trend line
if len(monthly_counts) > 1:
    z = np.polyfit(monthly_indices, monthly_counts.values, 1)
    p = np.poly1d(z)
    ax5.plot(monthly_indices, p(monthly_indices), "r--", linewidth=2, label='Trend')

ax5.set_title('Monthly Listing Additions', fontweight='bold', fontsize=10)
ax5.set_xlabel('Month')
ax5.set_ylabel('Number of Listings')
ax5.set_xticks(monthly_indices[::2])  # Show every other month
ax5.set_xticklabels([str(month) for month in monthly_counts.index[::2]], rotation=45, fontsize=8)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Middle-right: Price range by bedroom count stacked bar
ax6 = axes[1, 2]
price_bedroom_cross = pd.crosstab(df_clean['bedrooms'], df_clean['price_category'])
price_bedroom_cross.plot(kind='bar', stacked=True, ax=ax6, 
                        color=['lightblue', 'orange', 'red'], edgecolor='white')
ax6.set_title('Price Categories by Bedroom Count', fontweight='bold', fontsize=10)
ax6.set_xlabel('Number of Bedrooms')
ax6.set_ylabel('Count')
ax6.legend(title='Price Category', fontsize=8)
ax6.tick_params(axis='x', rotation=0)

# 7. Bottom-left: Box plot with strip plot
ax7 = axes[2, 0]
bedroom_price_data = df_clean[df_clean['bedrooms'].isin([0, 1, 2, 3, 4, 5])]
bedroom_groups = bedroom_price_data.groupby('bedrooms')['price'].apply(list)
ax7.boxplot([group for group in bedroom_groups], 
            labels=bedroom_groups.index, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))

# Add some sample points instead of all points for performance
for i, (bedroom, prices) in enumerate(bedroom_groups.items()):
    if len(prices) > 20:
        sample_prices = np.random.choice(prices, 20, replace=False)
    else:
        sample_prices = prices
    x_pos = np.random.normal(i+1, 0.04, len(sample_prices))
    ax7.scatter(x_pos, sample_prices, alpha=0.6, color='red', s=10)

ax7.set_title('Price Distribution by Bedroom Count', fontweight='bold', fontsize=10)
ax7.set_xlabel('Number of Bedrooms')
ax7.set_ylabel('Price (AED)')

# 8. Bottom-center: Ridge plot for bedroom-bathroom combinations
ax8 = axes[2, 1]
df_clean['bed_bath_combo'] = df_clean['bedrooms'].astype(str) + 'BR-' + df_clean['bathrooms'].astype(str) + 'BA'
top_combos = df_clean['bed_bath_combo'].value_counts().head(5).index

y_pos = 0
colors = plt.cm.viridis(np.linspace(0, 1, len(top_combos)))
for i, combo in enumerate(top_combos):
    combo_data = df_clean[df_clean['bed_bath_combo'] == combo]['price']
    if len(combo_data) > 5:
        # Simple histogram-based density
        hist, bin_edges = np.histogram(combo_data, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Normalize and scale
        hist_norm = hist / hist.max() * 0.8
        ax8.fill_between(bin_centers, y_pos, y_pos + hist_norm, alpha=0.7, color=colors[i])
        ax8.text(combo_data.median(), y_pos + 0.4, combo, fontsize=8, ha='center')
        y_pos += 1

ax8.set_title('Price Distributions by BR-BA Combinations', fontweight='bold', fontsize=10)
ax8.set_xlabel('Price (AED)')
ax8.set_ylabel('Combination')
ax8.set_yticks([])

# 9. Bottom-right: Scatter plot instead of hexbin for performance
ax9 = axes[2, 2]
valid_data = df_clean.dropna(subset=['days_since_listing', 'price'])
# Sample data if too large
if len(valid_data) > 1000:
    valid_data = valid_data.sample(1000)

scatter = ax9.scatter(valid_data['days_since_listing'], valid_data['price'], 
                     alpha=0.6, c=valid_data['price'], cmap='Blues', s=20)
ax9.set_title('Price vs Days Since Listing', fontweight='bold', fontsize=10)
ax9.set_xlabel('Days Since Listing')
ax9.set_ylabel('Price (AED)')

# Add colorbar
plt.colorbar(scatter, ax=ax9, shrink=0.8, label='Price')

# Adjust layout
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.35, wspace=0.35)
plt.savefig('uae_real_estate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()