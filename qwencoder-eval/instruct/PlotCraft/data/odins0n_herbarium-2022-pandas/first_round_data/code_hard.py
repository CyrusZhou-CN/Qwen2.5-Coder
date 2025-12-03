import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('train.csv')

# Data preprocessing
# Convert image_id to numeric for analysis (extract numeric part)
df['image_id_numeric'] = df['image_id'].str.extract('(\d+)').astype(int)

# Sample data for computational efficiency (use 10000 samples)
df_sample = df.sample(n=min(10000, len(df)), random_state=42)

# Create the 2x2 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Subplot 1: Top-left - Combined bar chart with cumulative percentage and error bars
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('white')

# Category distribution
category_counts = df['category'].value_counts().sort_index()
categories = category_counts.index[:20]  # Top 20 categories for visibility
counts = category_counts.values[:20]

# Calculate cumulative percentage
cumulative_pct = np.cumsum(counts) / np.sum(counts) * 100

# Calculate standard deviation of genus_id within each category
std_devs = []
for cat in categories:
    genus_std = df[df['category'] == cat]['genus_id'].std()
    std_devs.append(genus_std if not np.isnan(genus_std) else 0)

# Bar chart
bars = ax1.bar(range(len(categories)), counts, alpha=0.7, color='steelblue', 
               yerr=std_devs, capsize=3, error_kw={'color': 'red', 'alpha': 0.6})

# Overlaid line plot for cumulative percentage
ax1_twin = ax1.twinx()
line = ax1_twin.plot(range(len(categories)), cumulative_pct, 'ro-', 
                     linewidth=2, markersize=4, color='darkred', alpha=0.8)

ax1.set_xlabel('Category ID', fontweight='bold')
ax1.set_ylabel('Sample Count', fontweight='bold', color='steelblue')
ax1_twin.set_ylabel('Cumulative Percentage (%)', fontweight='bold', color='darkred')
ax1.set_title('Category Distribution with Cumulative Percentage\nand Genus ID Standard Deviation', 
              fontweight='bold', fontsize=12, pad=20)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1_twin.tick_params(axis='y', labelcolor='darkred')
ax1.grid(True, alpha=0.3)

# Subplot 2: Top-right - Hierarchical clustering with scatter plot overlay
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('white')

# Prepare data for clustering (sample for performance)
cluster_sample = df_sample.sample(n=min(500, len(df_sample)), random_state=42)
cluster_data = cluster_sample[['genus_id', 'category']].values

# Standardize the data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Perform hierarchical clustering
linkage_matrix = linkage(cluster_data_scaled, method='ward')

# Create dendrogram
dendro = dendrogram(linkage_matrix, ax=ax2, leaf_rotation=90, leaf_font_size=8,
                   color_threshold=0.7*max(linkage_matrix[:,2]))

# Overlay scatter plot
ax2_twin = ax2.twinx()
scatter_sample = df_sample.sample(n=min(200, len(df_sample)), random_state=42)
scatter = ax2_twin.scatter(scatter_sample['image_id_numeric'], scatter_sample['category'], 
                          c=scatter_sample['genus_id'], cmap='viridis', alpha=0.6, s=20)

ax2.set_title('Hierarchical Clustering Dendrogram\nwith Image ID vs Category Scatter', 
              fontweight='bold', fontsize=12, pad=20)
ax2.set_xlabel('Sample Index', fontweight='bold')
ax2.set_ylabel('Distance', fontweight='bold')
ax2_twin.set_ylabel('Category', fontweight='bold')

# Add colorbar for scatter plot
cbar = plt.colorbar(scatter, ax=ax2_twin, shrink=0.8)
cbar.set_label('Genus ID', fontweight='bold')

# Subplot 3: Bottom-left - Violin plot with strip plot and box plot overlay
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('white')

# Select top categories for violin plot
top_categories = df['category'].value_counts().head(8).index
violin_data = []
violin_labels = []

for cat in top_categories:
    cat_data = df[df['category'] == cat]['image_id_numeric'].values
    if len(cat_data) > 10:  # Only include categories with sufficient data
        violin_data.append(cat_data[:1000])  # Limit for performance
        violin_labels.append(f'Cat {cat}')

# Create violin plot
violin_parts = ax3.violinplot(violin_data, positions=range(len(violin_data)), 
                             showmeans=True, showmedians=True)

# Customize violin plot colors
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

# Add strip plot overlay
for i, data in enumerate(violin_data):
    sample_data = np.random.choice(data, min(50, len(data)), replace=False)
    y_pos = np.random.normal(i, 0.04, len(sample_data))
    ax3.scatter(sample_data, y_pos, alpha=0.4, s=8, color='darkblue')

# Add box plot statistics
box_data = [np.percentile(data, [25, 50, 75]) for data in violin_data]
for i, (q25, median, q75) in enumerate(box_data):
    ax3.plot([q25, q75], [i, i], 'k-', linewidth=2)
    ax3.plot([median], [i], 'ro', markersize=4)

ax3.set_xlabel('Image ID (Numeric)', fontweight='bold')
ax3.set_ylabel('Category', fontweight='bold')
ax3.set_title('Distribution of Image IDs by Category\n(Violin + Strip + Box Plot)', 
              fontweight='bold', fontsize=12, pad=20)
ax3.set_yticks(range(len(violin_labels)))
ax3.set_yticklabels(violin_labels)
ax3.grid(True, alpha=0.3)

# Subplot 4: Bottom-right - Network visualization with parallel coordinates
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('white')

# Create network data
network_sample = df_sample.sample(n=min(1000, len(df_sample)), random_state=42)

# Count connections between categories and genus_ids
connections = Counter()
for _, row in network_sample.iterrows():
    connections[(f"C{row['category']}", f"G{row['genus_id']}")] += 1

# Create network graph
G = nx.Graph()
for (cat, genus), weight in connections.most_common(50):  # Top 50 connections
    G.add_edge(cat, genus, weight=weight)

# Position nodes
pos = nx.spring_layout(G, k=1, iterations=50)

# Draw network
node_colors = ['lightcoral' if node.startswith('C') else 'lightblue' for node in G.nodes()]
node_sizes = [300 if node.startswith('C') else 200 for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                      alpha=0.8, ax=ax4)
nx.draw_networkx_labels(G, pos, font_size=6, ax=ax4)

# Draw edges with thickness based on weight
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights) if weights else 1
edge_widths = [w/max_weight * 3 for w in weights]

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                      edge_color='gray', ax=ax4)

# Add parallel coordinates overlay (simplified)
parallel_sample = network_sample.sample(n=min(100, len(network_sample)), random_state=42)
normalized_data = parallel_sample[['image_id_numeric', 'genus_id', 'category']].copy()

# Normalize data for parallel coordinates
for col in normalized_data.columns:
    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                          (normalized_data[col].max() - normalized_data[col].min())

# Draw parallel coordinates lines
x_positions = [0.1, 0.5, 0.9]
for _, row in normalized_data.head(20).iterrows():  # Show only 20 lines for clarity
    y_values = [row['image_id_numeric'], row['genus_id'], row['category']]
    ax4.plot(x_positions, y_values, alpha=0.3, linewidth=0.5, color='purple')

ax4.set_title('Network Analysis: Category-Genus Connections\nwith Parallel Coordinates Overlay', 
              fontweight='bold', fontsize=12, pad=20)
ax4.set_xlim(-0.2, 1.1)
ax4.set_ylim(-0.1, 1.1)
ax4.set_xticks([0.1, 0.5, 0.9])
ax4.set_xticklabels(['Image ID', 'Genus ID', 'Category'], fontweight='bold')

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()