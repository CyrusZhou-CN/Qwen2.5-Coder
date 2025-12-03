import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('udemy_courses.csv')

# Data preprocessing
# Clean and prepare numerical columns
df_clean = df.dropna(subset=['num_subscribers', 'rating', 'num_reviews', 'instructional_level'])
df_clean = df_clean[df_clean['rating'] > 0]  # Remove courses with no rating
df_clean = df_clean[df_clean['num_subscribers'] > 0]  # Remove courses with no subscribers

# Sample data for performance (use top courses by subscribers)
df_sample = df_clean.nlargest(2000, 'num_subscribers').copy()

# Prepare features for clustering
features = ['num_subscribers', 'rating', 'num_reviews']
X = df_sample[features].copy()

# Log transform skewed features
X['num_subscribers'] = np.log1p(X['num_subscribers'])
X['num_reviews'] = np.log1p(X['num_reviews'])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create figure with white background
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes
colors_level = {'Beginner Level': '#2E86AB', 'Intermediate Level': '#A23B72', 
                'Advanced Level': '#F18F01', 'All Levels': '#C73E1D', 'Expert Level': '#592E83'}

# Top-left: Scatter plot with K-means clustering overlay
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('white')

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_sample['cluster'] = clusters

# Create scatter plot
for level in df_sample['instructional_level'].unique():
    if level in colors_level:
        mask = df_sample['instructional_level'] == level
        sizes = (df_sample.loc[mask, 'num_reviews'] / df_sample['num_reviews'].max() * 200 + 20)
        ax1.scatter(df_sample.loc[mask, 'num_subscribers'], df_sample.loc[mask, 'rating'], 
                   c=colors_level[level], s=sizes, alpha=0.6, label=level, edgecolors='white', linewidth=0.5)

# Add cluster boundaries using contour
xx, yy = np.meshgrid(np.linspace(df_sample['num_subscribers'].min(), df_sample['num_subscribers'].max(), 100),
                     np.linspace(df_sample['rating'].min(), df_sample['rating'].max(), 100))
grid_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, df_sample['num_reviews'].median())]
grid_scaled = scaler.transform(grid_points)
Z = kmeans.predict(grid_scaled)
Z = Z.reshape(xx.shape)

ax1.contour(xx, yy, Z, levels=2, colors='black', linestyles='--', alpha=0.5, linewidths=1)
ax1.set_xlabel('Number of Subscribers (log scale)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Rating', fontsize=12, fontweight='bold')
ax1.set_title('Course Clustering: Subscribers vs Rating\nwith K-means Boundaries (k=3)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Top-right: Hierarchical clustering dendrogram with correlation heatmap
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('white')

# Sample smaller subset for dendrogram
df_dendro = df_sample.sample(n=50, random_state=42)
X_dendro = df_dendro[features].copy()
X_dendro['num_subscribers'] = np.log1p(X_dendro['num_subscribers'])
X_dendro['num_reviews'] = np.log1p(X_dendro['num_reviews'])
X_dendro_scaled = StandardScaler().fit_transform(X_dendro)

# Create dendrogram
linkage_matrix = linkage(X_dendro_scaled, method='ward')
dendro = dendrogram(linkage_matrix, ax=ax2, orientation='top', 
                   color_threshold=0.7*max(linkage_matrix[:,2]), 
                   above_threshold_color='gray')

ax2.set_title('Hierarchical Clustering Dendrogram\n(Sample of 50 courses)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Course Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('Distance', fontsize=12, fontweight='bold')

# Add correlation heatmap as inset
ax2_inset = fig.add_axes([0.65, 0.65, 0.15, 0.15])
corr_matrix = df_sample[features].corr()
im = ax2_inset.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax2_inset.set_xticks(range(len(features)))
ax2_inset.set_yticks(range(len(features)))
ax2_inset.set_xticklabels(['Subscribers', 'Rating', 'Reviews'], rotation=45, fontsize=8)
ax2_inset.set_yticklabels(['Subscribers', 'Rating', 'Reviews'], fontsize=8)
ax2_inset.set_title('Correlation Matrix', fontsize=10, fontweight='bold')

# Add correlation values
for i in range(len(features)):
    for j in range(len(features)):
        ax2_inset.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                      ha='center', va='center', fontsize=8, fontweight='bold')

# Bottom-left: Parallel coordinates with bar chart overlay
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('white')

# Normalize features for parallel coordinates (0-1 scale)
df_parallel = df_sample.sample(n=200, random_state=42).copy()
features_norm = df_parallel[features].copy()
for col in features:
    features_norm[col] = (features_norm[col] - features_norm[col].min()) / (features_norm[col].max() - features_norm[col].min())

# Plot parallel coordinates
x_pos = np.arange(len(features))
for idx, row in features_norm.iterrows():
    level = df_parallel.loc[idx, 'instructional_level']
    if level in colors_level:
        ax3.plot(x_pos, row[features], color=colors_level[level], alpha=0.3, linewidth=1)

# Add mean lines for each level
for level in colors_level.keys():
    if level in df_parallel['instructional_level'].values:
        mask = df_parallel['instructional_level'] == level
        if mask.sum() > 0:
            mean_values = features_norm[mask].mean()
            ax3.plot(x_pos, mean_values, color=colors_level[level], linewidth=3, 
                    label=f'{level} (mean)', alpha=0.8)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Subscribers', 'Rating', 'Reviews'], fontsize=12, fontweight='bold')
ax3.set_ylabel('Normalized Values (0-1)', fontsize=12, fontweight='bold')
ax3.set_title('Parallel Coordinates Plot\nwith Instructional Level Means', 
              fontsize=14, fontweight='bold', pad=20)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Add secondary y-axis for distribution
ax3_twin = ax3.twinx()
level_counts = df_sample['instructional_level'].value_counts()
level_counts_filtered = level_counts[level_counts.index.isin(colors_level.keys())]
bars = ax3_twin.bar(range(len(level_counts_filtered)), level_counts_filtered.values, 
                   alpha=0.3, width=0.3, color='gray')
ax3_twin.set_ylabel('Course Count by Level', fontsize=12, fontweight='bold')
ax3_twin.set_ylim(0, max(level_counts_filtered.values) * 1.2)

# Bottom-right: Network visualization with radial bar chart
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('white')

# Create instructor network based on shared instructional levels
instructor_data = df_sample.groupby('instructor_names').agg({
    'num_subscribers': 'sum',
    'rating': 'mean',
    'instructional_level': lambda x: list(x.unique())
}).reset_index()

# Filter top instructors
top_instructors = instructor_data.nlargest(15, 'num_subscribers')

# Create network graph
G = nx.Graph()
for idx, instructor in top_instructors.iterrows():
    G.add_node(instructor['instructor_names'], 
               subscribers=instructor['num_subscribers'],
               rating=instructor['rating'])

# Add edges for instructors sharing instructional levels
for i, inst1 in top_instructors.iterrows():
    for j, inst2 in top_instructors.iterrows():
        if i < j:  # Avoid duplicate edges
            shared_levels = set(inst1['instructional_level']) & set(inst2['instructional_level'])
            if shared_levels:
                G.add_edge(inst1['instructor_names'], inst2['instructor_names'], 
                          weight=len(shared_levels))

# Position nodes in a circular layout
pos = nx.circular_layout(G)

# Draw network
node_sizes = [G.nodes[node]['subscribers'] / 10000 for node in G.nodes()]
node_colors = [G.nodes[node]['rating'] for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                      cmap='viridis', alpha=0.7, ax=ax4)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax4)

# Add radial bar chart around the perimeter
top_10_instructors = top_instructors.head(10)
angles = np.linspace(0, 2*np.pi, len(top_10_instructors), endpoint=False)
ratings = top_10_instructors['rating'].values
max_rating = ratings.max()

for i, (angle, rating) in enumerate(zip(angles, ratings)):
    # Calculate position for radial bars
    radius_start = 1.3
    radius_end = radius_start + (rating / max_rating) * 0.4
    
    x_start = radius_start * np.cos(angle)
    y_start = radius_start * np.sin(angle)
    x_end = radius_end * np.cos(angle)
    y_end = radius_end * np.sin(angle)
    
    ax4.plot([x_start, x_end], [y_start, y_end], 
             color='red', linewidth=8, alpha=0.7)
    
    # Add instructor name
    text_x = (radius_end + 0.1) * np.cos(angle)
    text_y = (radius_end + 0.1) * np.sin(angle)
    instructor_name = top_10_instructors.iloc[i]['instructor_names']
    if len(instructor_name) > 20:
        instructor_name = instructor_name[:20] + '...'
    
    ax4.text(text_x, text_y, instructor_name, 
             rotation=np.degrees(angle) if angle < np.pi else np.degrees(angle) + 180,
             ha='left' if angle < np.pi else 'right',
             va='center', fontsize=8, fontweight='bold')

ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.set_aspect('equal')
ax4.set_title('Instructor Network & Top 10 by Average Rating\n(Node size = Total subscribers, Edges = Shared levels)', 
              fontsize=14, fontweight='bold', pad=20)
ax4.axis('off')

# Add colorbar for network nodes
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax4, shrink=0.6, pad=0.1)
cbar.set_label('Average Rating', fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)

plt.show()