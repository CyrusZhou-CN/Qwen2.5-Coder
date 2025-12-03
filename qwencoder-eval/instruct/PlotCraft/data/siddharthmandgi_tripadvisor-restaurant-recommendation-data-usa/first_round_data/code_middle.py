import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('TripAdvisor_RestauarantRecommendation.csv')

# Data preprocessing
# Extract numerical rating from Reviews column (e.g., "4.5 of 5 bubbles" -> 4.5)
df['Rating'] = df['Reviews'].str.extract(r'(\d+\.?\d*)').astype(float)

# Extract numerical values from No of Reviews column and handle NaN values
df['Num_Reviews_Raw'] = df['No of Reviews'].str.extract(r'(\d+)')
# Convert to numeric, handling NaN values
df['Num_Reviews'] = pd.to_numeric(df['Num_Reviews_Raw'], errors='coerce')

# Clean and standardize price ranges - escape dollar signs to prevent mathtext parsing
df['Price_Range'] = df['Price_Range'].fillna('Unknown')
# Replace dollar signs with text to avoid mathtext parsing issues
price_range_mapping = {
    '$': 'Budget',
    '$$': 'Moderate', 
    '$$ - $$$': 'Mid-Range',
    '$$$$': 'Expensive',
    'Unknown': 'Unknown'
}

# Apply mapping, keeping original if not in mapping
df['Price_Range_Clean'] = df['Price_Range'].map(price_range_mapping).fillna(df['Price_Range'])

# Remove rows with missing ratings or review counts
df = df.dropna(subset=['Rating', 'Num_Reviews'])

# Ensure we have data to work with
if len(df) == 0:
    print("No valid data found after cleaning")
    exit()

# Get top 15 restaurants by rating (with tie-breaking by number of reviews)
top_15 = df.nlargest(15, ['Rating', 'Num_Reviews'])

# Create color mapping for price ranges
price_ranges = df['Price_Range_Clean'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(price_ranges)))
color_map = dict(zip(price_ranges, colors))

# Create figure with 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
fig.patch.set_facecolor('white')

# Top subplot: Horizontal bar chart of top 15 restaurants
restaurant_names = [name[:30] + '...' if len(name) > 30 else name for name in top_15['Name']]
bar_colors = [color_map[price] for price in top_15['Price_Range_Clean']]

bars = ax1.barh(range(len(top_15)), top_15['Rating'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(restaurant_names, fontsize=10)
ax1.set_xlabel('Average Star Rating', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Restaurants by Average Star Rating', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlim(0, 5)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Add rating values on bars
for i, (bar, rating) in enumerate(zip(bars, top_15['Rating'])):
    ax1.text(rating + 0.05, i, f'{rating:.1f}', va='center', fontsize=9, fontweight='bold')

# Create legend for price ranges - use cleaned names
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[price], alpha=0.8, edgecolor='black') 
                  for price in price_ranges]
ax1.legend(legend_elements, price_ranges, loc='lower right', title='Price Range', 
          title_fontsize=10, fontsize=9)

# Bottom subplot: Scatter plot with trend lines
# Create ranking positions for point sizes (inverse ranking for size)
df_with_rank = df.copy()
df_with_rank['Rank'] = df_with_rank['Rating'].rank(method='dense', ascending=False)
max_rank = df_with_rank['Rank'].max()
df_with_rank['Point_Size'] = 200 - (df_with_rank['Rank'] - 1) * (150 / max_rank)

# Create scatter plot
for price_range in price_ranges:
    subset = df_with_rank[df_with_rank['Price_Range_Clean'] == price_range]
    if len(subset) > 0:
        ax2.scatter(subset['Num_Reviews'], subset['Rating'], 
                   s=subset['Point_Size'], alpha=0.6, 
                   color=color_map[price_range], label=price_range,
                   edgecolors='black', linewidth=0.5)
        
        # Add trend line if there are enough points
        if len(subset) > 2:
            try:
                # Remove any remaining NaN values for trend line calculation
                clean_subset = subset.dropna(subset=['Num_Reviews', 'Rating'])
                if len(clean_subset) > 2:
                    z = np.polyfit(clean_subset['Num_Reviews'], clean_subset['Rating'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(clean_subset['Num_Reviews'].min(), 
                                        clean_subset['Num_Reviews'].max(), 100)
                    ax2.plot(x_trend, p(x_trend), color=color_map[price_range], 
                            linestyle='--', alpha=0.8, linewidth=2)
            except:
                pass

ax2.set_xlabel('Number of Reviews', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Star Rating', fontsize=12, fontweight='bold')
ax2.set_title('Restaurant Ratings vs Review Volume by Price Range\n(Point size indicates ranking position)', 
             fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)
ax2.legend(title='Price Range', title_fontsize=10, fontsize=9, loc='upper right')

# Set y-axis limits for better visualization
ax2.set_ylim(0, 5.2)

# Adjust layout to prevent overlap
plt.subplots_adjust(hspace=0.3)
plt.savefig('restaurant_ranking_analysis.png', dpi=300, bbox_inches='tight')
plt.show()