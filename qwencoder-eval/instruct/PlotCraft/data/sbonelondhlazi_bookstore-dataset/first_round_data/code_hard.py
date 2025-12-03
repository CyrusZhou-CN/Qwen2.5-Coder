import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib.patches import Rectangle
import networkx as nx
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('books_scraped.csv')

# Convert star ratings to numeric
rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
df['Star_rating_num'] = df['Star_rating'].map(rating_map)

# Clean and prepare data
df = df.dropna(subset=['Price', 'Quantity', 'Star_rating_num'])
df = df[df['Price'] > 0]  # Remove invalid prices

# Create figure with white background
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Comprehensive Book Market Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

# Color palette
colors = plt.cm.Set3(np.linspace(0, 1, len(df['Book_category'].unique())))
category_colors = dict(zip(df['Book_category'].unique(), colors))

# 1. Top-left: Scatter plot with marginal histograms
ax1 = plt.subplot(3, 3, 1)
scatter = ax1.scatter(df['Price'], df['Quantity'], c=df['Star_rating_num'], 
                     cmap='viridis', alpha=0.6, s=30)
ax1.set_xlabel('Price ($)', fontweight='bold')
ax1.set_ylabel('Quantity', fontweight='bold')
ax1.set_title('Price vs Quantity by Star Rating', fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Star Rating', fontweight='bold')

# 2. Top-center: Stacked bar chart with average price line
ax2 = plt.subplot(3, 3, 2)
category_counts = df['Book_category'].value_counts()
category_avg_price = df.groupby('Book_category')['Price'].mean()

# Stacked bar by rating
rating_counts = df.groupby(['Book_category', 'Star_rating']).size().unstack(fill_value=0)
rating_counts.plot(kind='bar', stacked=True, ax=ax2, alpha=0.7)

# Overlay average price line
ax2_twin = ax2.twinx()
category_avg_price.plot(kind='line', ax=ax2_twin, color='red', marker='o', linewidth=2)
ax2_twin.set_ylabel('Average Price ($)', color='red', fontweight='bold')

ax2.set_xlabel('Category', fontweight='bold')
ax2.set_ylabel('Book Count', fontweight='bold')
ax2.set_title('Category Distribution with Average Price', fontweight='bold', pad=20)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Star Rating', bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. Top-right: Box plot with violin overlay (FIXED)
ax3 = plt.subplot(3, 3, 3)
top_categories = df['Book_category'].value_counts().head(8).index
df_top = df[df['Book_category'].isin(top_categories)]

# Violin plot - fix alpha issue by setting it on the returned parts
parts = ax3.violinplot([df_top[df_top['Book_category'] == cat]['Price'].values 
                       for cat in top_categories], positions=range(len(top_categories)))
for pc in parts['bodies']:
    pc.set_alpha(0.3)
    pc.set_facecolor('lightblue')

# Box plot overlay
box_data = [df_top[df_top['Book_category'] == cat]['Price'].values for cat in top_categories]
ax3.boxplot(box_data, positions=range(len(top_categories)))

ax3.set_xlabel('Category', fontweight='bold')
ax3.set_ylabel('Price ($)', fontweight='bold')
ax3.set_title('Price Distribution by Category', fontweight='bold', pad=20)
ax3.set_xticks(range(len(top_categories)))
ax3.set_xticklabels([cat[:10] for cat in top_categories], rotation=45)
ax3.grid(True, alpha=0.3)

# 4. Middle-left: Heatmap of category vs star rating
ax4 = plt.subplot(3, 3, 4)
heatmap_data = df.groupby(['Book_category', 'Star_rating']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'shrink': 0.8})
ax4.set_xlabel('Star Rating', fontweight='bold')
ax4.set_ylabel('Category', fontweight='bold')
ax4.set_title('Category vs Star Rating Counts', fontweight='bold', pad=20)

# 5. Middle-center: Radar chart
ax5 = plt.subplot(3, 3, 5, projection='polar')
top5_categories = df['Book_category'].value_counts().head(5).index

# Prepare metrics
metrics = []
for cat in top5_categories:
    cat_data = df[df['Book_category'] == cat]
    metrics.append([
        cat_data['Price'].mean(),
        cat_data['Quantity'].mean(),
        len(cat_data),
        cat_data['Star_rating_num'].mean()
    ])

metrics = np.array(metrics)
# Normalize metrics to 0-1 scale
metrics_norm = (metrics - metrics.min(axis=0)) / (metrics.max(axis=0) - metrics.min(axis=0) + 1e-8)

# Radar chart
angles = np.linspace(0, 2 * pi, 4, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors_radar = ['red', 'blue', 'green', 'orange', 'purple']
for i, cat in enumerate(top5_categories):
    values = metrics_norm[i].tolist()
    values += values[:1]  # Complete the circle
    ax5.plot(angles, values, 'o-', linewidth=2, label=cat[:10], color=colors_radar[i])
    ax5.fill(angles, values, alpha=0.25, color=colors_radar[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(['Avg Price', 'Avg Quantity', 'Book Count', 'Avg Rating'], fontweight='bold')
ax5.set_title('Top 5 Categories Comparison', fontweight='bold', pad=30)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 6. Middle-right: Treemap simulation using rectangles
ax6 = plt.subplot(3, 3, 6)
category_stats = df.groupby('Book_category').agg({
    'Title': 'count',
    'Price': 'mean'
}).rename(columns={'Title': 'count'})

# Sort by count for better visualization
category_stats = category_stats.sort_values('count', ascending=False).head(10)

# Create treemap-like visualization
x, y = 0, 0
max_width = 10
current_row_height = 0
row_width = 0

for i, (cat, stats) in enumerate(category_stats.iterrows()):
    width = max(stats['count'] / 10, 0.5)  # Scale width with minimum
    height = max(stats['Price'] / 20, 0.5)  # Scale height with minimum
    
    if row_width + width > max_width:
        y += current_row_height + 0.1
        x = 0
        row_width = 0
        current_row_height = 0
    
    # Color intensity based on price
    price_range = category_stats['Price'].max() - category_stats['Price'].min()
    if price_range > 0:
        color_intensity = (stats['Price'] - category_stats['Price'].min()) / price_range
    else:
        color_intensity = 0.5
    
    rect = Rectangle((x, y), width, height, facecolor=plt.cm.viridis(color_intensity), 
                    edgecolor='white', linewidth=1)
    ax6.add_patch(rect)
    
    # Add text if rectangle is large enough
    if width > 1 and height > 1:
        ax6.text(x + width/2, y + height/2, cat[:8], ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    x += width
    row_width += width
    current_row_height = max(current_row_height, height)

ax6.set_xlim(0, max_width)
ax6.set_ylim(0, max(y + current_row_height + 1, 5))
ax6.set_title('Category Treemap (Size: Count, Color: Avg Price)', fontweight='bold', pad=20)
ax6.set_xticks([])
ax6.set_yticks([])

# 7. Bottom-left: Parallel coordinates plot
ax7 = plt.subplot(3, 3, 7)
sample_data = df.sample(min(200, len(df)))  # Sample for clarity

# Normalize data for parallel coordinates
features = ['Price', 'Quantity', 'Star_rating_num']
normalized_data = sample_data[features].copy()
for feature in features:
    feature_range = normalized_data[feature].max() - normalized_data[feature].min()
    if feature_range > 0:
        normalized_data[feature] = (normalized_data[feature] - normalized_data[feature].min()) / feature_range
    else:
        normalized_data[feature] = 0.5

# Plot parallel coordinates
unique_categories = sample_data['Book_category'].unique()
colors_parallel = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
cat_color_map = dict(zip(unique_categories, colors_parallel))

for idx, row in normalized_data.iterrows():
    cat = sample_data.loc[idx, 'Book_category']
    color = cat_color_map.get(cat, 'gray')
    ax7.plot(range(len(features)), row.values, alpha=0.3, color=color, linewidth=0.5)

ax7.set_xticks(range(len(features)))
ax7.set_xticklabels(features, fontweight='bold')
ax7.set_ylabel('Normalized Values', fontweight='bold')
ax7.set_title('Parallel Coordinates Plot', fontweight='bold', pad=20)
ax7.grid(True, alpha=0.3)

# 8. Bottom-center: Dendrogram
ax8 = plt.subplot(3, 3, 8)
category_features = df.groupby('Book_category').agg({
    'Price': 'mean',
    'Quantity': 'mean',
    'Star_rating_num': 'mean'
}).head(10)  # Limit to top 10 for readability

# Perform hierarchical clustering
if len(category_features) > 1:
    linkage_matrix = linkage(category_features.values, method='ward')
    dendrogram(linkage_matrix, labels=[cat[:8] for cat in category_features.index], 
              ax=ax8, orientation='top')
    ax8.set_title('Category Clustering Dendrogram', fontweight='bold', pad=20)
    ax8.tick_params(axis='x', rotation=90)
else:
    ax8.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center', 
            transform=ax8.transAxes, fontsize=12)
    ax8.set_title('Category Clustering Dendrogram', fontweight='bold', pad=20)

# 9. Bottom-right: Network graph
ax9 = plt.subplot(3, 3, 9)
G = nx.Graph()

# Create network based on similar price ranges
categories = df['Book_category'].unique()[:10]  # Limit for readability
category_stats_net = df.groupby('Book_category').agg({
    'Price': 'mean',
    'Star_rating_num': 'mean'
})

for i, cat1 in enumerate(categories):
    G.add_node(cat1)
    for cat2 in categories[i+1:]:
        if cat1 in category_stats_net.index and cat2 in category_stats_net.index:
            price1 = category_stats_net.loc[cat1, 'Price']
            price2 = category_stats_net.loc[cat2, 'Price']
            rating1 = category_stats_net.loc[cat1, 'Star_rating_num']
            rating2 = category_stats_net.loc[cat2, 'Star_rating_num']
            
            # Connect if similar price and rating
            if abs(price1 - price2) < 15 and abs(rating1 - rating2) < 1.5:
                G.add_edge(cat1, cat2, weight=1/(abs(price1-price2)+1))

# Draw network
if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, ax=ax9, with_labels=True, node_color='lightblue', 
            node_size=300, font_size=6, font_weight='bold')
    ax9.set_title('Category Relationship Network', fontweight='bold', pad=20)
else:
    ax9.text(0.5, 0.5, 'No network connections found', ha='center', va='center', 
            transform=ax9.transAxes, fontsize=12)
    ax9.set_title('Category Relationship Network', fontweight='bold', pad=20)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
plt.savefig('comprehensive_book_analysis.png', dpi=300, bbox_inches='tight')
plt.show()