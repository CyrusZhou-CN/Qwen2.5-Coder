import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

# Load data
df = pd.read_csv('uae_properties.csv')

# Data preprocessing - more efficient filtering
# Remove rows with missing values in key columns
df_clean = df.dropna(subset=['price', 'bedrooms', 'bathrooms']).copy()

# Remove outliers more efficiently
df_clean = df_clean[
    (df_clean['bedrooms'] > 0) & 
    (df_clean['bedrooms'] <= 6) &  # Limit to reasonable bedroom counts
    (df_clean['price'] > 100000) & 
    (df_clean['price'] < 15000000)  # Remove extreme outliers
]

# Calculate price per bedroom
df_clean['price_per_bedroom'] = df_clean['price'] / df_clean['bedrooms']

# Limit data size for performance if needed
if len(df_clean) > 400:
    df_clean = df_clean.sample(n=400, random_state=42)

# Create figure with simpler grid layout
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], 
              hspace=0.3, wspace=0.3)

# Main scatter plot
ax_main = fig.add_subplot(gs[1, 0])

# Create scatter plot with optimized parameters
# Normalize size values to reasonable range
size_values = (df_clean['price_per_bedroom'] - df_clean['price_per_bedroom'].min()) / \
              (df_clean['price_per_bedroom'].max() - df_clean['price_per_bedroom'].min()) * 100 + 20

scatter = ax_main.scatter(df_clean['bedrooms'], df_clean['price'], 
                         c=df_clean['bathrooms'], s=size_values,
                         alpha=0.6, cmap='viridis', edgecolors='white', linewidth=0.3)

# Add simple trend line
bedrooms_unique = np.sort(df_clean['bedrooms'].unique())
price_means = [df_clean[df_clean['bedrooms'] == br]['price'].mean() for br in bedrooms_unique]
ax_main.plot(bedrooms_unique, price_means, "r-", alpha=0.8, linewidth=2, 
             label='Average price trend', marker='o', markersize=4)

# Calculate correlation coefficient
correlation = np.corrcoef(df_clean['bedrooms'], df_clean['price'])[0, 1]

# Styling for main plot
ax_main.set_xlabel('Number of Bedrooms', fontsize=12, fontweight='bold')
ax_main.set_ylabel('Price (AED)', fontsize=12, fontweight='bold')
ax_main.set_title('UAE Property Prices vs Bedrooms\n(Color: Bathrooms, Size: Price per Bedroom)', 
                  fontsize=13, fontweight='bold')
ax_main.grid(True, alpha=0.3)

# Add correlation text
ax_main.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax_main.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Color bar for bathrooms
cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
cbar.set_label('Number of Bathrooms', fontsize=10)

# Top box plot (simplified)
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
bedroom_counts = sorted(df_clean['bedrooms'].unique())

# Create box plot data more efficiently
box_data = []
positions = []
for br in bedroom_counts:
    br_data = df_clean[df_clean['bedrooms'] == br]['price'].values
    if len(br_data) >= 3:  # Only include if we have enough data points
        box_data.append(br_data)
        positions.append(br)

if box_data:
    bp = ax_top.boxplot(box_data, positions=positions, widths=0.5, 
                        patch_artist=True, showfliers=False)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax_top.set_ylabel('Price (AED)', fontsize=10)
ax_top.set_title('Price Distribution by Bedroom Count', fontsize=11, fontweight='bold')
ax_top.grid(True, alpha=0.3)
plt.setp(ax_top.get_xticklabels(), visible=False)

# Right histogram
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
ax_right.hist(df_clean['price'], bins=20, orientation='horizontal', alpha=0.7, 
              color='skyblue', edgecolor='white')
ax_right.set_xlabel('Count', fontsize=10)
ax_right.set_title('Price\nDistribution', fontsize=11, fontweight='bold')
ax_right.grid(True, alpha=0.3)
plt.setp(ax_right.get_yticklabels(), visible=False)

# Add simple legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                             markersize=6, alpha=0.6, label='Small: Low price/bedroom'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                             markersize=12, alpha=0.6, label='Large: High price/bedroom')]
ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Format price axis labels to millions
def price_formatter(x, p):
    return f'{x/1e6:.1f}M'

ax_main.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
ax_top.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))

# Set bedroom ticks
ax_main.set_xticks(bedroom_counts)

# Add summary statistics
stats_text = f'Properties: {len(df_clean)}\nAvg Price: {df_clean["price"].mean()/1e6:.1f}M AED'
ax_main.text(0.95, 0.05, stats_text, transform=ax_main.transAxes, 
             fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('uae_property_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()