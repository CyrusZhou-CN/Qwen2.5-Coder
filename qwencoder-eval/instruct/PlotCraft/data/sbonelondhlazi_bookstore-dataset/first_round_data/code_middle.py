import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('books_scraped.csv')

# Data preprocessing
# Convert star ratings from text to numeric
rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
df['Star_rating_numeric'] = df['Star_rating'].map(rating_map)

# Remove any rows with missing values in key columns
df_clean = df.dropna(subset=['Price', 'Star_rating_numeric', 'Book_category'])

# Create figure with subplots
fig = plt.figure(figsize=(12, 10))

# Create a 2x2 grid layout
gs = plt.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], 
                  hspace=0.3, wspace=0.3)

# Main scatter plot
ax_main = plt.subplot(gs[1, 0])

# Get unique categories and create a simple color palette
categories = df_clean['Book_category'].unique()
# Use a limited set of distinct colors to avoid performance issues
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = base_colors[:len(categories)]
color_map = dict(zip(categories, colors))

# Create scatter plot with category colors (simplified)
for i, category in enumerate(categories):
    category_data = df_clean[df_clean['Book_category'] == category]
    ax_main.scatter(category_data['Price'], category_data['Star_rating_numeric'], 
                   c=colors[i % len(colors)], label=category, alpha=0.6, s=20)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['Price'], df_clean['Star_rating_numeric'])
line_x = np.array([df_clean['Price'].min(), df_clean['Price'].max()])
line_y = slope * line_x + intercept
ax_main.plot(line_x, line_y, 'red', linewidth=2, alpha=0.8, 
            label=f'Regression Line (r={r_value:.3f})')

# Style main plot
ax_main.set_xlabel('Price ($)', fontweight='bold', fontsize=12)
ax_main.set_ylabel('Star Rating', fontweight='bold', fontsize=12)
ax_main.set_title('Book Prices vs Star Ratings by Category', fontweight='bold', fontsize=14)
ax_main.grid(True, alpha=0.3)
ax_main.set_yticks([1, 2, 3, 4, 5])
ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Top histogram (price distribution)
ax_top = plt.subplot(gs[0, 0])
ax_top.hist(df_clean['Price'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
ax_top.set_ylabel('Frequency', fontweight='bold', fontsize=10)
ax_top.set_title('Price Distribution', fontweight='bold', fontsize=11)
ax_top.grid(True, alpha=0.3)

# Right histogram (rating distribution)
ax_right = plt.subplot(gs[1, 1])
ax_right.hist(df_clean['Star_rating_numeric'], bins=5, alpha=0.7, color='lightcoral', 
              edgecolor='black', orientation='horizontal')
ax_right.set_xlabel('Frequency', fontweight='bold', fontsize=10)
ax_right.set_title('Rating\nDistribution', fontweight='bold', fontsize=11)
ax_right.grid(True, alpha=0.3)
ax_right.set_yticks([1, 2, 3, 4, 5])

# Add correlation statistics as text box
textstr = f'Correlation: r = {r_value:.3f}\nP-value: {p_value:.3e}\nSample size: {len(df_clean)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('book_price_rating_correlation.png', dpi=300, bbox_inches='tight')
plt.show()