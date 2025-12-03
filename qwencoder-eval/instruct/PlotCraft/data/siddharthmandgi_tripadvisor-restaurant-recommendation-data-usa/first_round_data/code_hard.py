import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.patches import Ellipse
import squarify
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to avoid mathtext parsing issues
plt.rcParams['text.usetex'] = False

# Load and preprocess data
df = pd.read_csv('TripAdvisor_RestauarantRecommendation.csv')

# Clean and extract numeric ratings
def extract_rating(rating_str):
    if pd.isna(rating_str):
        return np.nan
    try:
        return float(str(rating_str).split()[0])
    except:
        return np.nan

# Clean and extract review counts
def extract_review_count(review_str):
    if pd.isna(review_str):
        return 0
    try:
        return int(str(review_str).split()[0].replace(',', ''))
    except:
        return 0

# Extract state from location
def extract_state(location_str):
    if pd.isna(location_str):
        return 'Unknown'
    try:
        parts = str(location_str).split(',')
        if len(parts) >= 2:
            state_part = parts[-1].strip()
            return state_part.split()[0] if state_part else 'Unknown'
        return 'Unknown'
    except:
        return 'Unknown'

# Extract city from location
def extract_city(location_str):
    if pd.isna(location_str):
        return 'Unknown'
    try:
        parts = str(location_str).split(',')
        return parts[0].strip() if parts else 'Unknown'
    except:
        return 'Unknown'

# Apply preprocessing
df['Rating'] = df['Reviews'].apply(extract_rating)
df['Review_Count'] = df['No of Reviews'].apply(extract_review_count)
df['State'] = df['Location'].apply(extract_state)
df['City'] = df['Location'].apply(extract_city)

# Clean price range and convert to numeric - avoid $ symbols in labels
def clean_price_range(price_str):
    if pd.isna(price_str):
        return 'Budget'
    price_str = str(price_str).strip()
    if price_str == '$':
        return 'Budget'
    elif price_str in ['$ - $$', '$$ - $$$']:
        return 'Mid-range'
    elif price_str == '$$$$':
        return 'Expensive'
    else:
        return 'Mid-range'

df['Price_Category'] = df['Price_Range'].apply(clean_price_range)
price_map = {'Budget': 1, 'Mid-range': 2.5, 'Expensive': 4}
df['Price_Numeric'] = df['Price_Category'].map(price_map).fillna(2.5)

# Extract primary cuisine type
df['Primary_Cuisine'] = df['Type'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')

# Remove rows with missing critical data
df_clean = df.dropna(subset=['Rating', 'Review_Count']).copy()
df_clean = df_clean[df_clean['Review_Count'] > 0].copy()

# Limit to top states for better visualization
top_states = df_clean['State'].value_counts().head(6).index
df_viz = df_clean[df_clean['State'].isin(top_states)].copy()

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Subplot 1: Scatter plot with density contours
ax1 = plt.subplot(3, 3, 1)
scatter = ax1.scatter(df_viz['Review_Count'], df_viz['Rating'], 
                     c=df_viz['Price_Numeric'], cmap='viridis', alpha=0.6, s=30)
ax1.set_xlabel('Review Count', fontweight='bold')
ax1.set_ylabel('Star Rating', fontweight='bold')
ax1.set_title('Review Count vs Rating by Price Range', fontweight='bold', fontsize=11)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Price Level')
ax1.grid(True, alpha=0.3)

# Subplot 2: Grouped violin plots with box plots
ax2 = plt.subplot(3, 3, 2)
price_categories = ['Budget', 'Mid-range', 'Expensive']
df_price = df_viz[df_viz['Price_Category'].isin(price_categories)]
violin_data = [df_price[df_price['Price_Category'] == pc]['Review_Count'].values 
               for pc in price_categories if len(df_price[df_price['Price_Category'] == pc]) > 0]
valid_categories = [pc for pc in price_categories if len(df_price[df_price['Price_Category'] == pc]) > 0]

if len(violin_data) > 0:
    parts = ax2.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    ax2.boxplot(violin_data, positions=range(len(violin_data)), widths=0.3)
    ax2.set_xticks(range(len(valid_categories)))
    ax2.set_xticklabels(valid_categories, rotation=45)

ax2.set_ylabel('Review Count', fontweight='bold')
ax2.set_title('Review Count Distribution by Price Range', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3)

# Subplot 3: Stacked bar chart with line plot
ax3 = plt.subplot(3, 3, 3)
ax3_twin = ax3.twinx()

# Top cuisines by state
top_cuisines = df_viz['Primary_Cuisine'].value_counts().head(4).index
cuisine_by_state = df_viz.groupby(['State', 'Primary_Cuisine']).size().unstack(fill_value=0)
cuisine_by_state = cuisine_by_state.reindex(columns=top_cuisines, fill_value=0)
cuisine_by_state.plot(kind='bar', stacked=True, ax=ax3, colormap='Set3')

# Average ratings per state
avg_ratings = df_viz.groupby('State')['Rating'].mean()
ax3_twin.plot(range(len(avg_ratings)), avg_ratings.values, 'ro-', linewidth=2, markersize=6)
ax3_twin.set_ylabel('Average Rating', fontweight='bold', color='red')

ax3.set_xlabel('State', fontweight='bold')
ax3.set_ylabel('Restaurant Count', fontweight='bold')
ax3.set_title('Cuisine Distribution by State', fontweight='bold', fontsize=11)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.tick_params(axis='x', rotation=45)

# Subplot 4: Hierarchical clustering dendrogram
ax4 = plt.subplot(3, 3, 4)
# Prepare data for clustering
cluster_data = df_viz[['Rating', 'Review_Count', 'Price_Numeric']].copy()
cluster_data = cluster_data.sample(min(50, len(cluster_data)))  # Limit for performance
cluster_data_scaled = StandardScaler().fit_transform(cluster_data)

# Compute linkage
linkage_matrix = linkage(cluster_data_scaled, method='ward')
dendrogram(linkage_matrix, ax=ax4, leaf_rotation=90, leaf_font_size=6)
ax4.set_title('Restaurant Clustering Dendrogram', fontweight='bold', fontsize=11)
ax4.set_xlabel('Restaurant Index', fontweight='bold')
ax4.set_ylabel('Distance', fontweight='bold')

# Subplot 5: Network graph
ax5 = plt.subplot(3, 3, 5)
G = nx.Graph()
# Create network based on cuisine similarity and price range
sample_df = df_viz.sample(min(30, len(df_viz)))  # Limit for performance
for i, row1 in sample_df.iterrows():
    for j, row2 in sample_df.iterrows():
        if i < j and row1['Primary_Cuisine'] == row2['Primary_Cuisine'] and row1['Price_Category'] == row2['Price_Category']:
            G.add_edge(i, j)

if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1, iterations=50)
    node_sizes = [max(20, sample_df.loc[node, 'Review_Count'] / 20) for node in G.nodes()]
    nx.draw(G, pos, ax=ax5, node_size=node_sizes, node_color='lightcoral', 
            edge_color='gray', alpha=0.7, with_labels=False)
ax5.set_title('Restaurant Network by Cuisine & Price', fontweight='bold', fontsize=11)

# Subplot 6: Parallel coordinates plot
ax6 = plt.subplot(3, 3, 6)
sample_parallel = df_viz.sample(min(50, len(df_viz)))
features = ['Rating', 'Review_Count', 'Price_Numeric']
normalized_data = StandardScaler().fit_transform(sample_parallel[features])

for i in range(len(normalized_data)):
    ax6.plot(range(len(features)), normalized_data[i], alpha=0.3, color='blue')

ax6.set_xticks(range(len(features)))
ax6.set_xticklabels(['Rating', 'Reviews', 'Price'], fontweight='bold')
ax6.set_ylabel('Normalized Values', fontweight='bold')
ax6.set_title('Restaurant Multi-dimensional Profiles', fontweight='bold', fontsize=11)
ax6.grid(True, alpha=0.3)

# Subplot 7: Treemap
ax7 = plt.subplot(3, 3, 7)
# Prepare treemap data
treemap_data = df_viz.groupby(['State', 'City']).agg({
    'Review_Count': 'sum',
    'Rating': 'mean'
}).reset_index()
treemap_data = treemap_data.head(15)  # Limit for visibility

if len(treemap_data) > 0:
    sizes = treemap_data['Review_Count'].values
    colors = treemap_data['Rating'].values
    labels = [f"{row['State']}\n{row['City'][:8]}" for _, row in treemap_data.iterrows()]
    
    squarify.plot(sizes=sizes, label=labels, color=plt.cm.RdYlBu(colors/5), 
                  alpha=0.8, ax=ax7, text_kwargs={'fontsize': 7})

ax7.set_title('Restaurant Distribution by Location', fontweight='bold', fontsize=11)
ax7.axis('off')

# Subplot 8: Radar chart
ax8 = plt.subplot(3, 3, 8, projection='polar')
# Top 5 most reviewed restaurants
top_restaurants = df_viz.nlargest(5, 'Review_Count')
metrics = ['Rating', 'Review_Count', 'Price_Numeric']

if len(top_restaurants) > 0:
    # Normalize metrics for radar chart
    normalized_metrics = StandardScaler().fit_transform(top_restaurants[metrics])
    normalized_metrics = (normalized_metrics - normalized_metrics.min()) / (normalized_metrics.max() - normalized_metrics.min() + 1e-8)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_restaurants)))
    for i, (_, restaurant) in enumerate(top_restaurants.iterrows()):
        values = normalized_metrics[i].tolist()
        values += values[:1]  # Complete the circle
        ax8.plot(angles, values, 'o-', linewidth=2, label=restaurant['Name'][:10], color=colors[i])
        ax8.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(['Rating', 'Reviews', 'Price'], fontweight='bold')

ax8.set_title('Top 5 Restaurants Comparison', fontweight='bold', fontsize=11, pad=20)
ax8.legend(bbox_to_anchor=(1.3, 1.0), fontsize=7)

# Subplot 9: PCA cluster scatter plot
ax9 = plt.subplot(3, 3, 9)
# Prepare data for PCA
pca_data = df_viz[['Rating', 'Review_Count', 'Price_Numeric']].copy()
pca_data_scaled = StandardScaler().fit_transform(pca_data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data_scaled)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_data_scaled)

# Plot clusters
colors = ['red', 'blue', 'green']
for i in range(3):
    mask = clusters == i
    if np.any(mask):
        ax9.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=colors[i], alpha=0.6, s=30, label=f'Cluster {i+1}')

# Plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax9.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroids')

ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
ax9.set_title('Restaurant Clusters (PCA)', fontweight='bold', fontsize=11)
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

# Final layout adjustment
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.4)
plt.savefig('restaurant_analysis.png', dpi=300, bbox_inches='tight')
plt.show()