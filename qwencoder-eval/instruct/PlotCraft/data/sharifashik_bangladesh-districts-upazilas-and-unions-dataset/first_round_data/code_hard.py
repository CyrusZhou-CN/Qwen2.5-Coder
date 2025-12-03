import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from matplotlib.patches import Rectangle
import networkx as nx
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load data
union_df = pd.read_csv('union.csv')
upozila_df = pd.read_csv('upozila.csv')
district_df = pd.read_csv('district.csv')

# Data preprocessing
# Merge datasets to get complete hierarchy
merged_df = union_df.merge(upozila_df, on=['district_id', 'upozila_id'])
merged_df = merged_df.merge(district_df, on='district_id')

# Calculate administrative metrics
district_stats = merged_df.groupby(['district_id', 'জেলা']).agg({
    'upozila_id': 'nunique',
    'ইউনিয়ন': 'count'
}).rename(columns={'upozila_id': 'upazila_count', 'ইউনিয়ন': 'union_count'}).reset_index()

upazila_stats = merged_df.groupby(['district_id', 'জেলা', 'upozila_id', 'উপজেলা']).agg({
    'ইউনিয়ন': 'count'
}).rename(columns={'ইউনিয়ন': 'union_count'}).reset_index()

# Create comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# 1. Top-left: Treemap showing district sizes by number of upazilas
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Simple treemap implementation
district_sorted = district_stats.sort_values('upazila_count', ascending=False)
colors = plt.cm.Set3(np.linspace(0, 1, len(district_sorted)))

# Calculate treemap layout (simplified)
total_area = sum(district_sorted['upazila_count'])
x, y = 0, 0
width, height = 1, 1

for i, (_, row) in enumerate(district_sorted.head(20).iterrows()):
    area_ratio = row['upazila_count'] / total_area * 4  # Scale for visibility
    rect_width = np.sqrt(area_ratio) * 0.8
    rect_height = area_ratio / rect_width if rect_width > 0 else 0.1
    
    if x + rect_width > 1:
        x = 0
        y += 0.2
    
    rect = Rectangle((x, y), rect_width, rect_height, 
                    facecolor=colors[i], edgecolor='white', linewidth=1)
    ax1.add_patch(rect)
    
    # Add text label
    if rect_width > 0.15 and rect_height > 0.08:
        ax1.text(x + rect_width/2, y + rect_height/2, 
                f"{row['জেলা']}\n{row['upazila_count']}", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    x += rect_width + 0.02

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('District Sizes by Number of Upazilas', fontweight='bold', fontsize=12)
ax1.set_xticks([])
ax1.set_yticks([])

# 2. Top-center: Hierarchical dendrogram
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Create distance matrix based on upazila distribution
district_matrix = district_stats[['upazila_count', 'union_count']].values
linkage_matrix = linkage(district_matrix, method='ward')

dendrogram(linkage_matrix, ax=ax2, orientation='top', 
          labels=district_stats['জেলা'].values, leaf_rotation=90)
ax2.set_title('District Clustering by Administrative Structure', fontweight='bold', fontsize=12)
ax2.tick_params(axis='x', labelsize=8)

# 3. Top-right: Network graph
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Create network graph (simplified for top districts)
G = nx.Graph()
top_districts = district_stats.nlargest(10, 'upazila_count')

for _, district in top_districts.iterrows():
    district_upazilas = upazila_stats[upazila_stats['district_id'] == district['district_id']]
    
    # Add district node
    G.add_node(f"D_{district['জেলা']}", node_type='district', 
              size=district['union_count'])
    
    # Add upazila nodes and edges
    for _, upazila in district_upazilas.head(3).iterrows():  # Limit for visibility
        upazila_name = f"U_{upazila['উপজেলা'][:8]}"
        G.add_node(upazila_name, node_type='upazila', 
                  size=upazila['union_count'])
        G.add_edge(f"D_{district['জেলা']}", upazila_name)

pos = nx.spring_layout(G, k=2, iterations=50)
district_nodes = [n for n in G.nodes() if n.startswith('D_')]
upazila_nodes = [n for n in G.nodes() if n.startswith('U_')]

# Draw network
nx.draw_networkx_nodes(G, pos, nodelist=district_nodes, 
                      node_color='lightcoral', node_size=300, ax=ax3)
nx.draw_networkx_nodes(G, pos, nodelist=upazila_nodes, 
                      node_color='lightblue', node_size=150, ax=ax3)
nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax3)
nx.draw_networkx_labels(G, pos, font_size=6, ax=ax3)

ax3.set_title('District-Upazila Network Relationships', fontweight='bold', fontsize=12)
ax3.axis('off')

# 4. Middle-left: Parallel coordinates plot
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# Normalize data for parallel coordinates
top_10_districts = district_stats.nlargest(10, 'union_count')
normalized_data = top_10_districts[['upazila_count', 'union_count']].copy()
normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())

colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_districts)))

for i, (_, row) in enumerate(normalized_data.iterrows()):
    ax4.plot([0, 1], [row['upazila_count'], row['union_count']], 
            color=colors[i], alpha=0.7, linewidth=2)

ax4.set_xlim(-0.1, 1.1)
ax4.set_ylim(-0.1, 1.1)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Upazilas', 'Unions'])
ax4.set_title('Administrative Flow: Districts to Unions', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# 5. Middle-center: Radar chart
ax5 = plt.subplot(3, 3, 5, projection='polar')
ax5.set_facecolor('white')

# Top 8 districts for radar chart
top_8_districts = district_stats.nlargest(8, 'union_count')
categories = top_8_districts['জেলা'].tolist()
N = len(categories)

# Normalize values
upazila_values = top_8_districts['upazila_count'].values
union_values = top_8_districts['union_count'].values

upazila_norm = upazila_values / upazila_values.max()
union_norm = union_values / union_values.max()

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

upazila_norm = np.concatenate((upazila_norm, [upazila_norm[0]]))
union_norm = np.concatenate((union_norm, [union_norm[0]]))

ax5.plot(angles, upazila_norm, 'o-', linewidth=2, label='Upazilas', color='red')
ax5.fill(angles, upazila_norm, alpha=0.25, color='red')
ax5.plot(angles, union_norm, 'o-', linewidth=2, label='Unions', color='blue')
ax5.fill(angles, union_norm, alpha=0.25, color='blue')

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=8)
ax5.set_title('Top 8 Districts: Administrative Complexity', fontweight='bold', fontsize=12, pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 6. Middle-right: Clustered bubble chart
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Create size categories
district_stats['size_category'] = pd.cut(district_stats['upazila_count'], 
                                       bins=3, labels=['Small', 'Medium', 'Large'])

colors_dict = {'Small': 'lightgreen', 'Medium': 'orange', 'Large': 'red'}
for category in ['Small', 'Medium', 'Large']:
    data = district_stats[district_stats['size_category'] == category]
    if len(data) > 0:
        ax6.scatter(data['upazila_count'], data['union_count'], 
                   s=data['union_count']*2, alpha=0.6, 
                   c=colors_dict[category], label=category)

ax6.set_xlabel('Number of Upazilas', fontweight='bold')
ax6.set_ylabel('Number of Unions', fontweight='bold')
ax6.set_title('Districts by Administrative Size Categories', fontweight='bold', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Bottom-left: Stacked bar chart with line plot
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Top 15 districts for visibility
top_15 = district_stats.nlargest(15, 'union_count')
x_pos = np.arange(len(top_15))

bars = ax7.bar(x_pos, top_15['upazila_count'], color='skyblue', 
               label='Upazilas', alpha=0.8)

# Overlay line plot for cumulative union percentages
ax7_twin = ax7.twinx()
cumulative_unions = top_15['union_count'].cumsum() / top_15['union_count'].sum() * 100
line = ax7_twin.plot(x_pos, cumulative_unions, color='red', marker='o', 
                     linewidth=2, label='Cumulative Union %')

ax7.set_xlabel('Districts', fontweight='bold')
ax7.set_ylabel('Number of Upazilas', fontweight='bold')
ax7_twin.set_ylabel('Cumulative Union %', fontweight='bold')
ax7.set_title('Upazila Distribution with Cumulative Union Percentages', fontweight='bold', fontsize=12)
ax7.set_xticks(x_pos)
ax7.set_xticklabels(top_15['জেলা'], rotation=45, ha='right', fontsize=8)

# Combine legends
lines1, labels1 = ax7.get_legend_handles_labels()
lines2, labels2 = ax7_twin.get_legend_handles_labels()
ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 8. Bottom-center: Heatmap correlation matrix
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Create correlation matrix
corr_data = district_stats[['upazila_count', 'union_count']].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))

sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, ax=ax8, cbar_kws={'shrink': 0.8})
ax8.set_title('Administrative Metrics Correlation Matrix', fontweight='bold', fontsize=12)

# 9. Bottom-right: Multi-level donut chart
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Create nested donut chart
# Inner ring: Districts by size
size_counts = district_stats['size_category'].value_counts()
colors1 = ['lightcoral', 'gold', 'lightgreen']

# Outer ring: Top districts
top_5_districts = district_stats.nlargest(5, 'union_count')
colors2 = plt.cm.Set3(np.linspace(0, 1, len(top_5_districts)))

# Inner donut - Fixed unpacking issue
pie_result1 = ax9.pie(size_counts.values, labels=size_counts.index,
                      colors=colors1, radius=0.6, 
                      wedgeprops=dict(width=0.3, edgecolor='white'),
                      autopct='%1.1f%%', pctdistance=0.85)

# Outer donut - Fixed unpacking issue  
pie_result2 = ax9.pie(top_5_districts['union_count'], 
                      labels=top_5_districts['জেলা'],
                      colors=colors2, radius=1.0,
                      wedgeprops=dict(width=0.3, edgecolor='white'),
                      autopct='%1.0f', pctdistance=1.15)

ax9.set_title('Hierarchical Composition: Districts to Unions', fontweight='bold', fontsize=12)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Save the plot
plt.savefig('bangladesh_administrative_analysis.png', dpi=300, bbox_inches='tight')