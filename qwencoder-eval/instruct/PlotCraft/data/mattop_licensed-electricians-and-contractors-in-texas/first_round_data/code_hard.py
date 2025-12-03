import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('texas-electricians.csv')

# Data preprocessing
df = df.dropna(subset=['license', 'county'])
license_counts = df['license'].value_counts()
county_license = df.groupby(['county', 'license']).size().reset_index(name='count')

# Create figure with white background
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Complex Analysis of Licensed Electricians in Texas: Clustering and Hierarchical Relationships', 
             fontsize=20, fontweight='bold', y=0.95)

# Subplot 1: Treemap with text annotations
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('white')

# Calculate treemap data
total_licenses = len(df)
license_data = []
for license_type, count in license_counts.head(8).items():
    percentage = (count / total_licenses) * 100
    license_data.append((license_type, count, percentage))

# Create simple treemap using rectangles
colors = plt.cm.Set3(np.linspace(0, 1, len(license_data)))
y_pos = 0
for i, (license_type, count, percentage) in enumerate(license_data):
    height = percentage / 100
    rect = Rectangle((0, y_pos), 1, height, facecolor=colors[i], 
                    edgecolor='white', linewidth=2, alpha=0.8)
    ax1.add_patch(rect)
    
    # Add text annotations
    ax1.text(0.5, y_pos + height/2, f'{license_type}\n{count:,}\n({percentage:.1f}%)', 
            ha='center', va='center', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    y_pos += height

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('License Type Composition Treemap', fontsize=14, fontweight='bold', pad=20)
ax1.axis('off')

# Subplot 2: Dendrogram with horizontal bar chart
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('white')

# Prepare data for clustering
top_licenses = license_counts.head(10)
county_matrix = []
top_counties = df['county'].value_counts().head(50).index

for license_type in top_licenses.index:
    license_counties = df[df['license'] == license_type]['county'].value_counts()
    county_vector = []
    for county in top_counties:
        county_vector.append(license_counties.get(county, 0))
    county_matrix.append(county_vector)

county_matrix = np.array(county_matrix)

# Perform hierarchical clustering
linkage_matrix = linkage(county_matrix, method='ward')

# Create dendrogram
dendro = dendrogram(linkage_matrix, labels=top_licenses.index, 
                   orientation='left', ax=ax2, leaf_font_size=10)

# Add horizontal bar chart overlay
ax2_twin = ax2.twinx()
y_positions = np.arange(len(top_licenses))
bars = ax2_twin.barh(y_positions * 10 + 5, top_licenses.values, 
                    height=8, alpha=0.6, color='lightblue', edgecolor='navy')

ax2.set_title('Hierarchical Clustering of License Types', fontsize=14, fontweight='bold', pad=20)
ax2_twin.set_ylabel('License Count', fontsize=12)
ax2_twin.grid(True, alpha=0.3)

# Subplot 3: Network graph with scatter plot overlay
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('white')

# Create network data
G = nx.Graph()
top_counties_net = df['county'].value_counts().head(15).index
license_county_pairs = df[df['county'].isin(top_counties_net)].groupby(['license', 'county']).size()

# Add nodes and edges
for license_type in license_counts.head(8).index:
    G.add_node(license_type, node_type='license', size=license_counts[license_type])

for county in top_counties_net[:10]:
    G.add_node(county, node_type='county', size=df[df['county'] == county].shape[0])

# Add edges based on co-occurrence
for (license_type, county), count in license_county_pairs.items():
    if license_type in license_counts.head(8).index and county in top_counties_net[:10]:
        if count > 100:  # Threshold for significant relationships
            G.add_edge(license_type, county, weight=count)

# Create layout
pos = nx.spring_layout(G, k=3, iterations=50)

# Draw network
license_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'license']
county_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'county']

# Draw license nodes
license_sizes = [G.nodes[n]['size']/50 for n in license_nodes]
nx.draw_networkx_nodes(G, pos, nodelist=license_nodes, node_color='lightcoral', 
                      node_size=license_sizes, alpha=0.8, ax=ax3)

# Draw county nodes
county_sizes = [G.nodes[n]['size']/100 for n in county_nodes]
nx.draw_networkx_nodes(G, pos, nodelist=county_nodes, node_color='lightblue', 
                      node_size=county_sizes, alpha=0.8, ax=ax3)

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax3)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax3)

# Scatter plot overlay
x_coords = [pos[node][0] for node in license_nodes]
y_coords = [pos[node][1] for node in license_nodes]
sizes = [license_counts[node]/10 for node in license_nodes]
ax3.scatter(x_coords, y_coords, s=sizes, alpha=0.3, c='red', marker='o')

ax3.set_title('Network Analysis: License-County Relationships', fontsize=14, fontweight='bold', pad=20)
ax3.axis('off')

# Subplot 4: Parallel coordinates with violin plot
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('white')

# Prepare data for parallel coordinates
top_licenses_pc = license_counts.head(6).index
top_counties_pc = df['county'].value_counts().head(8).index

# Create parallel coordinates data
pc_data = []
for license_type in top_licenses_pc:
    license_df = df[df['license'] == license_type]
    county_dist = []
    for county in top_counties_pc:
        count = license_df[license_df['county'] == county].shape[0]
        county_dist.append(count)
    pc_data.append(county_dist)

pc_data = np.array(pc_data)

# Normalize data for parallel coordinates
pc_data_norm = (pc_data - pc_data.min(axis=1, keepdims=True)) / (pc_data.max(axis=1, keepdims=True) - pc_data.min(axis=1, keepdims=True) + 1e-8)

# Plot parallel coordinates
x_positions = np.arange(len(top_counties_pc))
colors_pc = plt.cm.tab10(np.linspace(0, 1, len(top_licenses_pc)))

for i, (license_type, color) in enumerate(zip(top_licenses_pc, colors_pc)):
    ax4.plot(x_positions, pc_data_norm[i], color=color, alpha=0.7, 
            linewidth=2, label=license_type[:15] + '...' if len(license_type) > 15 else license_type)

# Add violin plot overlay (fixed - removed alpha parameter)
violin_data = [pc_data_norm[:, i] for i in range(len(top_counties_pc))]
violin_parts = ax4.violinplot(violin_data, positions=x_positions, widths=0.8)

# Set alpha for violin plot bodies manually
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightgray')
    pc.set_alpha(0.3)

ax4.set_xticks(x_positions)
ax4.set_xticklabels([county[:8] + '...' if len(county) > 8 else county for county in top_counties_pc], 
                   rotation=45, ha='right')
ax4.set_ylabel('Normalized License Distribution', fontsize=12)
ax4.set_title('Parallel Coordinates: License Types vs County Distribution', fontsize=14, fontweight='bold', pad=20)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# Final layout adjustment
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
plt.savefig('texas_electricians_analysis.png', dpi=300, bbox_inches='tight')
plt.show()