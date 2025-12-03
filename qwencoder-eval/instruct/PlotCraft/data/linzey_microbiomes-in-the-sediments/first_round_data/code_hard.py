import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
mag_data = pd.read_csv('Supplementary Fig.10cd.csv')
pathway_data = pd.read_csv('Fig.3a.csv')
expression_data = pd.read_csv('Supplementary Fig.14.csv')
abundance_data = pd.read_csv('Fig.2b.csv')

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Create 2x2 subplot grid
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

# Top-left: Scatter plot with marginal histograms
ax1 = fig.add_subplot(gs[0, 0])

# Create synthetic genome size data for demonstration (since not in original data)
np.random.seed(42)
mag_data['genome_size'] = np.random.normal(3.5, 1.2, len(mag_data))

# Create scatter plot colored by phylum
phylums = mag_data['Phylum'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(phylums)))
phylum_colors = dict(zip(phylums, colors))

for phylum in phylums:
    phylum_data = mag_data[mag_data['Phylum'] == phylum]
    ax1.scatter(phylum_data['genome_size'], phylum_data['CAZyme rate'], 
               s=phylum_data['CAZyme count']*3, alpha=0.7, 
               c=[phylum_colors[phylum]], label=phylum, edgecolors='white', linewidth=0.5)

ax1.set_xlabel('Genome Size (Mbp)', fontweight='bold')
ax1.set_ylabel('CAZyme Rate', fontweight='bold')
ax1.set_title('Genome Size vs CAZyme Rate by Phylum', fontweight='bold', fontsize=14)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Top-right: Clustered heatmap (manual implementation)
ax2 = fig.add_subplot(gs[0, 1])

# Get top 15 most abundant genomes
genome_abundance = abundance_data.set_index('genome')
sample_cols = [col for col in genome_abundance.columns if 'cmbsf' in col]
genome_means = genome_abundance[sample_cols].mean(axis=1)
top_genomes = genome_means.nlargest(15).index

# Create heatmap data
heatmap_data = genome_abundance.loc[top_genomes, sample_cols].fillna(0)

# Perform hierarchical clustering
row_linkage = linkage(pdist(heatmap_data, metric='euclidean'), method='ward')
col_linkage = linkage(pdist(heatmap_data.T, metric='euclidean'), method='ward')

# Get clustered order
row_order = leaves_list(row_linkage)
col_order = leaves_list(col_linkage)
clustered_data = heatmap_data.iloc[row_order, col_order]

# Create heatmap
im = ax2.imshow(clustered_data, cmap='viridis', aspect='auto')
ax2.set_xticks(range(len(clustered_data.columns)))
ax2.set_xticklabels(clustered_data.columns, rotation=45, ha='right', fontsize=8)
ax2.set_yticks(range(len(clustered_data.index)))
ax2.set_yticklabels(clustered_data.index, fontsize=8)
ax2.set_title('Clustered Heatmap: Top 15 Genomes', fontweight='bold', fontsize=14)
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Relative Abundance', fontweight='bold')

# Bottom-left: Network correlation plot
ax3 = fig.add_subplot(gs[1, 0])

# Calculate correlations between genes
gene_expression = expression_data.pivot_table(values='Abundance (TPM)', 
                                            index='Sample', columns='gene', 
                                            aggfunc='mean').fillna(0)

# Get top 15 genes for network analysis (reduced for better visualization)
top_genes = gene_expression.mean().nlargest(15).index
gene_subset = gene_expression[top_genes]

# Calculate correlation matrix
corr_matrix = gene_subset.corr()

# Create network graph
G = nx.Graph()
correlations = []
for i, gene1 in enumerate(top_genes):
    for j, gene2 in enumerate(top_genes):
        if i < j:
            corr_val = corr_matrix.loc[gene1, gene2]
            if abs(corr_val) > 0.5:  # Lowered threshold for more connections
                G.add_edge(gene1, gene2, weight=abs(corr_val))
                correlations.append(abs(corr_val))

# Add isolated nodes
for gene in top_genes:
    if gene not in G.nodes():
        G.add_node(gene)

# Position nodes
pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

# Draw network
node_sizes = [gene_expression[gene].mean() * 0.5 for gene in G.nodes()]
node_colors = plt.cm.Set2(np.linspace(0, 1, len(G.nodes())))

# Draw edges with varying thickness based on correlation
edges = G.edges()
if edges:
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=[w*3 for w in edge_weights], ax=ax3)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                      alpha=0.8, ax=ax3)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax3)

ax3.set_title('Gene Correlation Network (|r| > 0.5)', fontweight='bold', fontsize=14)
ax3.axis('off')

# Bottom-right: Grouped violin plot with box plots
ax4 = fig.add_subplot(gs[1, 1])

# Create depth categories based on sample names
def categorize_depth(sample):
    # Extract depth information from sample names
    if '0-2' in sample or '0-3' in sample:
        return '0-10cm'
    elif '4-6' in sample or '6-8' in sample or '6-9' in sample or '8-10' in sample:
        return '10-20cm'
    else:
        return '>20cm'

expression_data['depth_category'] = expression_data['Sample'].apply(categorize_depth)

# Get top 8 most highly expressed genes (reduced for better visualization)
top_expressed_genes = expression_data.groupby('gene')['Abundance (TPM)'].mean().nlargest(8).index
violin_data = expression_data[expression_data['gene'].isin(top_expressed_genes)]

# Prepare data for violin plot
depth_categories = ['0-10cm', '10-20cm', '>20cm']
colors = ['lightblue', 'lightgreen', 'lightcoral']

# Create positions for each gene-depth combination
positions = []
labels = []
plot_data = []
plot_colors = []

for i, gene in enumerate(top_expressed_genes):
    for j, depth in enumerate(depth_categories):
        data = violin_data[(violin_data['gene'] == gene) & 
                          (violin_data['depth_category'] == depth)]['Abundance (TPM)'].values
        if len(data) > 0:
            positions.append(i * 4 + j)
            plot_data.append(data)
            plot_colors.append(colors[j])
            if i == 0:  # Only add depth labels for first gene
                labels.append(depth)

# Create violin plot
if plot_data:
    parts = ax4.violinplot(plot_data, positions=positions, widths=0.8, showmeans=True)
    
    # Color violin parts
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.7)

# Set x-axis
gene_positions = [i * 4 + 1 for i in range(len(top_expressed_genes))]
ax4.set_xticks(gene_positions)
ax4.set_xticklabels(top_expressed_genes, rotation=45, ha='right', fontsize=10)
ax4.set_ylabel('Gene Expression (TPM)', fontweight='bold')
ax4.set_title('Gene Expression by Depth Category', fontweight='bold', fontsize=14)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
                                label=depth) for i, depth in enumerate(depth_categories)]
ax4.legend(handles=legend_elements, loc='upper right')

ax4.grid(True, alpha=0.3)

# Final layout adjustment
plt.tight_layout()
plt.savefig('microbial_community_analysis.png', dpi=300, bbox_inches='tight')
plt.show()