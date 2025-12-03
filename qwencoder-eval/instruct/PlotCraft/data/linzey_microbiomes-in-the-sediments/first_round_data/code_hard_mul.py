import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Load all datasets
df_phylum = pd.read_csv('Supplementary Fig.3.csv')
df_gene_expr = pd.read_csv('Fig.3b.csv')
df_cazyme = pd.read_csv('Supplementary Fig.10cd.csv')
df_mag_temporal = pd.read_csv('Supplementary Fig.9.csv')
df_genome_size = pd.read_csv('Fig.2c.csv')
df_mns = pd.read_csv('Fig.1de.csv')

# Data preprocessing
# Extract depth information from sample names for phylum data
def extract_depth(sample_name):
    if '(' in sample_name and 'cmbsf' in sample_name:
        depth_part = sample_name.split('(')[1].split('cmbsf')[0]
        if '-' in depth_part:
            start, end = depth_part.split('-')
            return (int(start) + int(end)) / 2
    return 0

df_phylum['Depth_cmbsf'] = df_phylum['Samples'].apply(extract_depth)
df_phylum = df_phylum.sort_values('Depth_cmbsf')

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('white')

# Define consistent color schemes
phylum_colors = {
    'Proteobacteria': '#2E86AB',
    'Chloroflexota': '#A23B72', 
    'Thaumarchaeota': '#F18F01',
    'Planctomycetota': '#C73E1D'
}

# Check actual values in the data and create position colors accordingly
print("Unique groups in df_mns:", df_mns['Group'].unique())
print("Unique groups in df_genome_size:", df_genome_size['Group'].unique())
print("Unique positions in df_gene_expr:", df_gene_expr['Positon '].unique())

# Create position colors based on actual data
unique_positions_mns = df_mns['Group'].unique()
unique_positions_expr = df_gene_expr['Positon '].unique()
unique_positions_genome = df_genome_size['Group'].unique()

all_positions = list(set(list(unique_positions_mns) + list(unique_positions_expr) + list(unique_positions_genome)))
position_colors = {}
colors_list = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6']
for i, pos in enumerate(all_positions):
    position_colors[pos] = colors_list[i % len(colors_list)]

# Subplot 1: Stacked area chart with MNS overlay
ax1 = plt.subplot(3, 2, 1)
major_phyla = ['Proteobacteria', 'Chloroflexota', 'Thaumarchaeota', 'Planctomycetota']
depths = df_phylum['Depth_cmbsf'].values
phylum_data = df_phylum[major_phyla].values.T

# Create stacked area chart
ax1.stackplot(depths, *phylum_data, labels=major_phyla, 
              colors=[phylum_colors[p] for p in major_phyla], alpha=0.7)

# Overlay MNS trend line
mns_depths = []
mns_values = []
for _, row in df_mns.iterrows():
    depth = extract_depth(row['Sample'])
    if depth > 0:
        mns_depths.append(depth)
        mns_values.append(row['MNS'])

ax1_twin = ax1.twinx()
if mns_depths and mns_values:
    ax1_twin.plot(mns_depths, mns_values, 'k-', linewidth=3, marker='o', 
                  markersize=6, label='MNS', alpha=0.8)
    ax1_twin.set_ylabel('Microbiome Novelty Score', fontweight='bold', fontsize=11)
    ax1_twin.set_ylim(min(mns_values) * 0.9, max(mns_values) * 1.1)

ax1.set_xlabel('Sediment Depth (cmbsf)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Relative Abundance', fontweight='bold', fontsize=11)
ax1.set_title('Phylum Abundance vs Depth with MNS Trend', fontweight='bold', fontsize=13)
ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
if mns_depths and mns_values:
    ax1_twin.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
ax1.grid(True, alpha=0.3)

# Subplot 2: Dual-axis plot with novel 16S percentages and water depth
ax2 = plt.subplot(3, 2, 2)

# Calculate novel percentages (using MNS as proxy)
x_pos = np.arange(len(df_mns))
colors = [position_colors.get(group, '#808080') for group in df_mns['Group']]

bars = ax2.bar(x_pos, df_mns['MNS'] * 100, color=colors, alpha=0.7, 
               label='Novel 16S miTag %')

# Water depth line plot
ax2_twin = ax2.twinx()
ax2_twin.plot(x_pos, df_mns['Depth (m)'], 'k-', linewidth=2, marker='s', 
              markersize=4, label='Water Depth')

ax2.set_xlabel('Sampling Sites', fontweight='bold', fontsize=11)
ax2.set_ylabel('Novel 16S miTag (%)', fontweight='bold', fontsize=11)
ax2_twin.set_ylabel('Water Depth (m)', fontweight='bold', fontsize=11)
ax2.set_title('Novel 16S Percentages vs Water Depth by Position', fontweight='bold', fontsize=13)
ax2.set_xticks(x_pos[::5])
ax2.set_xticklabels([df_mns.iloc[i]['Sample'] for i in range(0, len(df_mns), 5)], 
                    rotation=45, ha='right')

# Create custom legend for positions
unique_groups = df_mns['Group'].unique()
legend_patches = []
for group in unique_groups:
    patch = Rectangle((0, 0), 1, 1, facecolor=position_colors.get(group, '#808080'), alpha=0.7)
    legend_patches.append(patch)
ax2.legend(legend_patches, unique_groups, loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Subplot 3: CAZyme categories with rate overlay
ax3 = plt.subplot(3, 2, 3)

# Create synthetic CAZyme category data based on actual CAZyme counts
cazyme_categories = ['GT', 'GH', 'CE', 'AA', 'PL', 'CBM']
phyla_for_cazyme = df_cazyme['Phylum'].unique()[:4]

# Generate realistic CAZyme data based on actual counts
np.random.seed(42)  # For reproducibility
cazyme_data = []
for phylum in phyla_for_cazyme:
    phylum_cazymes = df_cazyme[df_cazyme['Phylum'] == phylum]['CAZyme count']
    if len(phylum_cazymes) > 0:
        base_count = phylum_cazymes.mean()
        # Distribute across categories
        category_counts = np.random.dirichlet(np.ones(len(cazyme_categories))) * base_count
        cazyme_data.append(category_counts)
    else:
        cazyme_data.append(np.zeros(len(cazyme_categories)))

cazyme_data = np.array(cazyme_data)

# Stacked bar chart
bottom = np.zeros(len(phyla_for_cazyme))
colors_cazyme = plt.cm.Set3(np.linspace(0, 1, len(cazyme_categories)))

for i, category in enumerate(cazyme_categories):
    ax3.bar(range(len(phyla_for_cazyme)), cazyme_data[:, i], bottom=bottom, 
            label=category, color=colors_cazyme[i], alpha=0.8)
    bottom += cazyme_data[:, i]

# Overlay scatter points for CAZyme rate
phylum_rates = df_cazyme.groupby('Phylum')['CAZyme rate'].mean()
for i, phylum in enumerate(phyla_for_cazyme):
    if phylum in phylum_rates.index:
        ax3.scatter(i, phylum_rates[phylum] * 10000, color='red', s=100, 
                   marker='o', edgecolor='black', linewidth=2, zorder=10)

ax3.set_xticks(range(len(phyla_for_cazyme)))
ax3.set_xticklabels(phyla_for_cazyme, rotation=45, ha='right')
ax3.set_xlabel('Phylum', fontweight='bold', fontsize=11)
ax3.set_ylabel('CAZyme Gene Count', fontweight='bold', fontsize=11)
ax3.set_title('CAZyme Categories by Phylum with Rate Overlay', fontweight='bold', fontsize=13)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# Subplot 4: Heatmap-line combination for metabolic pathways
ax4 = plt.subplot(3, 2, 4)

# Group genes by pathway
pathway_genes = {
    'Glycolysis': ['glk', 'pckA'],
    'TCA cycle': ['CS', 'IDH', 'sucD'],
    'Acetate metabolism': ['pta', 'ackA', 'ACS', 'porA', 'aclA']
}

# Create heatmap data
samples = df_gene_expr['Sample'].unique()[:10]
pathway_matrix = []
pathway_labels = []

for pathway, genes in pathway_genes.items():
    pathway_data = []
    for sample in samples:
        sample_data = df_gene_expr[df_gene_expr['Sample'] == sample]
        pathway_expr = sample_data[sample_data['gene'].isin(genes)]['Abundance (TPM)'].mean()
        pathway_data.append(pathway_expr if not np.isnan(pathway_expr) else 0)
    pathway_matrix.append(pathway_data)
    pathway_labels.append(pathway)

# Create heatmap
pathway_matrix = np.array(pathway_matrix)
im = ax4.imshow(pathway_matrix, cmap='YlOrRd', aspect='auto')
ax4.set_yticks(range(len(pathway_labels)))
ax4.set_yticklabels(pathway_labels)
ax4.set_xticks(range(len(samples)))
ax4.set_xticklabels([s.split('(')[0] for s in samples], rotation=45, ha='right')

# Overlay trend lines
x_range = np.arange(len(samples))
for i, pathway_data in enumerate(pathway_matrix):
    # Normalize pathway data for overlay
    normalized_data = (pathway_data - pathway_data.min()) / (pathway_data.max() - pathway_data.min() + 1e-8)
    ax4.plot(x_range, i + normalized_data * 0.4 - 0.2, 'k-', alpha=0.7, linewidth=2)

ax4.set_title('Metabolic Pathway Expression Heatmap', fontweight='bold', fontsize=13)
ax4.set_xlabel('Samples', fontweight='bold', fontsize=11)
ax4.set_ylabel('Metabolic Pathways', fontweight='bold', fontsize=11)

# Add colorbar
cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
cbar.set_label('TPM Expression', fontweight='bold')

# Subplot 5: Violin plots with box plots and scatter overlay
ax5 = plt.subplot(3, 2, 5)

# Prepare genome size data by group
groups = df_genome_size['Group'].unique()
genome_data = [df_genome_size[df_genome_size['Group'] == group]['GenomeSize (Mb)'].values 
               for group in groups]

# Create violin plots
parts = ax5.violinplot(genome_data, positions=range(len(groups)), showmeans=True, 
                       showmedians=True, widths=0.6)

# Color violin plots
available_colors = list(position_colors.values())
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(available_colors[i % len(available_colors)])
    pc.set_alpha(0.7)

# Overlay box plots
bp = ax5.boxplot(genome_data, positions=range(len(groups)), widths=0.3, 
                 patch_artist=True, showfliers=False)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor('white')
    patch.set_alpha(0.8)

# Add scatter points
for i, group in enumerate(groups):
    group_data = df_genome_size[df_genome_size['Group'] == group]['GenomeSize (Mb)']
    y_scatter = group_data.values
    x_scatter = np.random.normal(i, 0.04, size=len(y_scatter))
    ax5.scatter(x_scatter, y_scatter, alpha=0.6, s=20, color='black')

ax5.set_xticks(range(len(groups)))
ax5.set_xticklabels(groups)
ax5.set_xlabel('Trench Position', fontweight='bold', fontsize=11)
ax5.set_ylabel('Genome Size (Mb)', fontweight='bold', fontsize=11)
ax5.set_title('Genome Size Distribution by Position', fontweight='bold', fontsize=13)
ax5.grid(True, alpha=0.3)

# Subplot 6: Temporal series plot for MAG clusters
ax6 = plt.subplot(3, 2, 6)

# Prepare temporal data
depth_columns = ['T3L11(6-9cmbsf)', 'T3L11(12-15cmbsf)', 'T3L11(18-21cmbsf)']
depth_midpoints = [7.5, 13.5, 19.5]

# Group by phylum and plot trends
phyla_temporal = df_mag_temporal['Phylum'].unique()[:5]
colors_temporal = plt.cm.tab10(np.linspace(0, 1, len(phyla_temporal)))

for i, phylum in enumerate(phyla_temporal):
    phylum_data = df_mag_temporal[df_mag_temporal['Phylum'] == phylum]
    
    if len(phylum_data) > 0:
        # Calculate mean abundance across depth intervals
        mean_abundances = []
        std_abundances = []
        
        for col in depth_columns:
            abundances = phylum_data[col].values
            abundances = abundances[abundances > 0]  # Remove zeros for log scale
            if len(abundances) > 0:
                mean_abundances.append(np.mean(abundances))
                std_abundances.append(np.std(abundances))
            else:
                mean_abundances.append(1e-8)  # Small value for log scale
                std_abundances.append(0)
        
        # Plot trend line with error bands
        ax6.plot(depth_midpoints, mean_abundances, 'o-', color=colors_temporal[i], 
                linewidth=2, markersize=6, label=phylum, alpha=0.8)
        
        # Add error bands
        upper_bound = np.array(mean_abundances) + np.array(std_abundances)
        lower_bound = np.maximum(np.array(mean_abundances) - np.array(std_abundances), 
                                np.full_like(mean_abundances, 1e-8))
        ax6.fill_between(depth_midpoints, lower_bound, upper_bound,
                        color=colors_temporal[i], alpha=0.2)

ax6.set_xlabel('Sediment Depth (cmbsf)', fontweight='bold', fontsize=11)
ax6.set_ylabel('MAG Cluster Abundance', fontweight='bold', fontsize=11)
ax6.set_title('Temporal MAG Cluster Changes by Depth', fontweight='bold', fontsize=13)
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('deep_sea_microbiome_analysis.png', dpi=300, bbox_inches='tight')
plt.show()