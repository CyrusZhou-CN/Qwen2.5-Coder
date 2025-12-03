import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df_abundance = pd.read_csv('Supplementary Fig.3.csv')
df_cazyme = pd.read_csv('Supplementary Fig.10b.csv')
df_mns = pd.read_csv('Fig.1de.csv')
df_genes = pd.read_csv('Fig.3c.csv')

# Create figure with 2x2 subplot layout and overall title
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('white')
fig.suptitle('Microbial Community Composition and Metabolic Activity in Deep-Sea Sediments', 
             fontsize=18, fontweight='bold', y=0.95)

# Define enhanced color palette with better contrast
colors_phyla = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
colors_genes = ['#e74c3c', '#3498db', '#2ecc71']
colors_group = {'Slope': '#FF6B35', 'Bottom': '#004E89'}

# 1. Top-left: Stacked area chart for T3L11 series microbial phyla
# Create synthetic depth data for demonstration
selected_samples = df_abundance.iloc[:6].copy()
selected_samples['depth'] = [6, 9, 12, 15, 19, 21]

# Select major phyla (top 6 by abundance)
phyla_cols = [col for col in df_abundance.columns if col != 'Samples']
major_phyla = df_abundance[phyla_cols].mean().nlargest(6).index.tolist()

# Create color mapping for phyla consistency
phyla_color_map = {phylum: colors_phyla[i] for i, phylum in enumerate(major_phyla)}

# Create stacked area chart
depths = selected_samples['depth'].values
bottom = np.zeros(len(depths))

for i, phylum in enumerate(major_phyla):
    values = selected_samples[phylum].values
    ax1.fill_between(depths, bottom, bottom + values, 
                    label=phylum, color=colors_phyla[i], alpha=0.85, edgecolor='white', linewidth=0.5)
    bottom += values

ax1.set_title('Microbial Phyla Abundance vs Sediment Depth (T3L11 Series)', 
              fontweight='bold', fontsize=14, pad=15)
ax1.set_xlabel('Sediment Depth (cmbsf)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Relative Abundance', fontweight='bold', fontsize=12)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True, fancybox=True)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xlim(depths.min()-0.5, depths.max()+0.5)

# 2. Top-right: Line plot for metabolic genes
genes_of_interest = ['glk', 'pckA', 'CS']
gene_data = df_genes[df_genes['Gene'].isin(genes_of_interest)]

depths_genes = [7.5, 13.5, 20]  # Midpoints of depth ranges
gene_columns = ['T3L11(6-9cmbsf)', 'T3L11(12-15cmbsf)', 'T3L11(19-21cmbsf)']

for i, gene in enumerate(genes_of_interest):
    if gene in gene_data['Gene'].values:
        gene_row = gene_data[gene_data['Gene'] == gene].iloc[0]
        values = [gene_row[col] for col in gene_columns]
        ax2.plot(depths_genes, values, marker='o', linewidth=3, markersize=10,
                label=gene, color=colors_genes[i], markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor=colors_genes[i])

ax2.set_title('Metabolic Gene Abundance Trends', fontweight='bold', fontsize=14, pad=15)
ax2.set_xlabel('Sediment Depth (cmbsf)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Gene Abundance (TPM)', fontweight='bold', fontsize=12)
ax2.legend(fontsize=11, frameon=True, fancybox=True)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 3. Bottom-left: Heatmap of CAZyme distribution with consistent coloring
cazyme_matrix = df_cazyme.set_index('Phylum')[['GT', 'GH', 'CE', 'AA', 'PL', 'CBM']]
# Select top 15 phyla by total CAZyme count
top_phyla = cazyme_matrix.sum(axis=1).nlargest(15).index
cazyme_subset = cazyme_matrix.loc[top_phyla]

# Normalize by row for better visualization
cazyme_normalized = cazyme_subset.div(cazyme_subset.sum(axis=1), axis=0)

im = ax3.imshow(cazyme_normalized.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
ax3.set_xticks(range(len(cazyme_normalized.columns)))
ax3.set_xticklabels(cazyme_normalized.columns, fontweight='bold', fontsize=11)
ax3.set_yticks(range(len(cazyme_normalized.index)))

# Color y-axis labels to match phyla colors from top-left plot
y_labels = []
for phylum in cazyme_normalized.index:
    if phylum in phyla_color_map:
        ax3.text(-0.7, cazyme_normalized.index.get_loc(phylum), phylum, 
                color=phyla_color_map[phylum], fontweight='bold', fontsize=10,
                verticalalignment='center', horizontalalignment='right')
    else:
        ax3.text(-0.7, cazyme_normalized.index.get_loc(phylum), phylum, 
                color='black', fontsize=10,
                verticalalignment='center', horizontalalignment='right')

ax3.set_yticklabels([])  # Hide default labels since we're using custom colored ones
ax3.set_title('CAZyme Functional Categories Distribution', fontweight='bold', fontsize=14, pad=15)
ax3.set_xlabel('CAZyme Categories', fontweight='bold', fontsize=12)
ax3.set_ylabel('Microbial Phyla', fontweight='bold', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.15)
cbar.set_label('Relative Proportion', fontweight='bold', fontsize=11)

# 4. Bottom-right: Scatter plot of MNS vs depth with trend lines
# Fix group labels and use consistent colors
available_groups = df_mns['Group'].unique()

for group in available_groups:
    group_data = df_mns[df_mns['Group'] == group]
    # Fix the group label display
    display_label = 'Trench Bottom' if group == 'Bottom' else group
    color = colors_group.get(group, '#4682B4')
    
    ax4.scatter(group_data['Depth (m)'], group_data['MNS'], 
               label=display_label, color=color, s=100, alpha=0.8, 
               edgecolors='white', linewidth=1.5)
    
    # Add trend line if we have enough points
    if len(group_data) > 1:
        z = np.polyfit(group_data['Depth (m)'], group_data['MNS'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(group_data['Depth (m)'].min(), group_data['Depth (m)'].max(), 100)
        ax4.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.8, linewidth=2)

ax4.set_title('Microbiome Novelty vs Trench Depth', fontweight='bold', fontsize=14, pad=15)
ax4.set_xlabel('Depth (m)', fontweight='bold', fontsize=12)
ax4.set_ylabel('Microbiome Novelty Score (MNS)', fontweight='bold', fontsize=12)
ax4.legend(fontsize=11, frameon=True, fancybox=True)
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Adjust layout with increased vertical spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.35, top=0.92)

# Show the plot
plt.show()