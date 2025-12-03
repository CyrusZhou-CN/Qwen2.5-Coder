import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load all datasets
fig1bc = pd.read_csv('Fig.1bc.csv')
supp_fig10cd = pd.read_csv('Supplementary Fig.10cd.csv')
fig2b = pd.read_csv('Fig.2b.csv')
supp_fig9 = pd.read_csv('Supplementary Fig.9.csv')
supp_fig4ab = pd.read_csv('Supplementary Fig.4ab.csv')
fig3c = pd.read_csv('Fig.3c.csv')

# Create figure with 2x2 subplot grid - adjusted spacing
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor('white')

# Subplot 1: Stacked area chart with line plots for T3L11 phyla abundance
t3l11_cols = ['T3L11(6-9cmbsf)', 'T3L11(12-15cmbsf)', 'T3L11(18-21cmbsf)']
depths = [7.5, 13.5, 19.5]

# Get phylum data from supp_fig9 and aggregate by phylum
phylum_data = {}
for phylum in supp_fig9['Phylum'].unique():
    phylum_subset = supp_fig9[supp_fig9['Phylum'] == phylum]
    phylum_totals = []
    for col in t3l11_cols:
        phylum_totals.append(phylum_subset[col].sum())
    phylum_data[phylum] = phylum_totals

# Get top 5 phyla by total abundance
phylum_totals = {k: sum(v) for k, v in phylum_data.items()}
top5_phyla = sorted(phylum_totals.items(), key=lambda x: x[1], reverse=True)[:5]

# Create stacked area chart with better colors
colors = ['#3498DB', '#E74C3C', '#F39C12', '#2ECC71', '#9B59B6']
bottom = np.zeros(3)

for i, (phylum, _) in enumerate(top5_phyla):
    values = np.array(phylum_data[phylum])
    ax1.fill_between(depths, bottom, bottom + values, 
                     alpha=0.8, color=colors[i], label=phylum.replace('bacteria', 'bact.'))
    # Add line plot for each phylum trend
    ax1.plot(depths, bottom + values/2, 
             color='white', linewidth=2.5, marker='o', markersize=6, 
             markeredgecolor=colors[i], markeredgewidth=2)
    bottom += values

ax1.set_xlabel('Sediment Depth (cmbsf)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Relative Abundance', fontweight='bold', fontsize=11)
ax1.set_title('Microbial Community Composition - T3L11\n(Stacked Areas + Trend Lines)', 
              fontweight='bold', fontsize=13)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(6, 21)

# Subplot 2: Horizontal bar chart + scatter plot for genome size vs CAZyme density
# Calculate CAZyme density (rate) and prepare data
cazyme_stats = supp_fig10cd.groupby('Phylum').agg({
    'CAZyme count': ['mean', 'std'],
    'CAZyme rate': 'mean'
}).round(3)

cazyme_stats.columns = ['count_mean', 'count_std', 'rate_mean']
cazyme_stats = cazyme_stats.fillna(0)

# Select top 6 phyla for visualization
top_phyla = cazyme_stats.nlargest(6, 'count_mean')

# Create horizontal bar chart
y_pos = np.arange(len(top_phyla))
bars = ax2.barh(y_pos, top_phyla['count_mean'], xerr=top_phyla['count_std'],
                capsize=4, alpha=0.7, color='#3498DB', edgecolor='black', linewidth=1)

ax2.set_ylabel('Phylum', fontweight='bold', fontsize=11)
ax2.set_xlabel('CAZyme Count', fontweight='bold', fontsize=11, color='#3498DB')
ax2.set_title('CAZyme Analysis by Phylum\n(Bars: Count ± SD, Scatter: Size vs Density)', 
              fontweight='bold', fontsize=13)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([p.replace('bacteria', 'bact.').replace('archaeota', 'arch.') 
                     for p in top_phyla.index], fontsize=10)
ax2.tick_params(axis='x', labelcolor='#3498DB')

# Overlay scatter plot for genome size vs CAZyme density
ax2_twin = ax2.twiny()
# Use CAZyme count as proxy for genome size and CAZyme rate as density
scatter_data = supp_fig10cd[supp_fig10cd['Phylum'].isin(top_phyla.index)]
ax2_twin.scatter(scatter_data['CAZyme rate'], 
                range(len(scatter_data)), 
                alpha=0.6, color='#E74C3C', s=40, 
                edgecolors='black', linewidth=0.5)
ax2_twin.set_xlabel('CAZyme Density (Rate)', fontweight='bold', fontsize=11, color='#E74C3C')
ax2_twin.tick_params(axis='x', labelcolor='#E74C3C')

# Subplot 3: Heatmap + line plot for gene expression and MNS
# Prepare gene expression data - select subset to avoid overlap
gene_expr = fig3c.set_index('Gene')
# Select every 3rd gene to avoid label overlap
selected_genes = gene_expr.index[::3]
gene_expr_subset = gene_expr.loc[selected_genes]
gene_expr_norm = (gene_expr_subset - gene_expr_subset.min()) / (gene_expr_subset.max() - gene_expr_subset.min())

# Create heatmap
im = ax3.imshow(gene_expr_norm.T, cmap='YlOrRd', aspect='auto', alpha=0.9)
ax3.set_yticks(range(len(gene_expr_subset.columns)))
ax3.set_yticklabels(['6-9 cm', '12-15 cm', '19-21 cm'], fontsize=10)
ax3.set_xticks(range(len(selected_genes)))
ax3.set_xticklabels(selected_genes, rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Depth (cmbsf)', fontweight='bold', fontsize=11)
ax3.set_xlabel('Metabolic Genes', fontweight='bold', fontsize=11)

# Add colorbar for heatmap
cbar = plt.colorbar(im, ax=ax3, shrink=0.7, pad=0.02)
cbar.set_label('Normalized TPM', fontweight='bold', fontsize=10)

# Overlay MNS trend line
ax3_twin = ax3.twinx()
mns_data = supp_fig4ab[supp_fig4ab['Sample'].str.contains('T3L11')]
if not mns_data.empty:
    mns_values = mns_data['Microbiome novelty scores (MNS)'].values
    y_positions = [0, 1, 2]  # Corresponding to the three depth levels
    ax3_twin.plot(range(len(selected_genes)), 
                  np.interp(range(len(selected_genes)), [0, len(selected_genes)-1], 
                           [mns_values[1], mns_values[-1]]), 
                  color='#2ECC71', linewidth=4, marker='s', markersize=8, 
                  alpha=0.9, markeredgecolor='white', markeredgewidth=2)
    ax3_twin.set_ylabel('Microbiome Novelty Score', fontweight='bold', fontsize=11, color='#2ECC71')
    ax3_twin.tick_params(axis='y', labelcolor='#2ECC71')
    ax3_twin.legend(['MNS Trend'], bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

ax3.set_title('Gene Expression Heatmap + MNS Trend\n(T3L11 Depth Profile)', 
              fontweight='bold', fontsize=13)

# Subplot 4: Violin plot for trench position + connected scatter for depth changes
# Prepare data for violin plot
fig1bc['Novel_numeric'] = pd.to_numeric(fig1bc['Novel 16s miTags (%)'].str.rstrip('%'), errors='coerce')

# Create violin plot for trench positions
unique_groups = fig1bc['Group'].unique()
violin_data = []
violin_labels = []

for group in unique_groups:
    group_data = fig1bc[fig1bc['Group'] == group]['Novel_numeric'].dropna()
    if len(group_data) > 0:
        violin_data.append(group_data.values)
        violin_labels.append(group)

if len(violin_data) > 0:
    parts = ax4.violinplot(violin_data, positions=range(len(violin_data)), 
                          widths=0.6, showmeans=True, showmedians=True)
    
    # Style violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('#9B59B6')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    # Style other elements
    for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if element in parts:
            parts[element].set_color('black')
            parts[element].set_linewidth(2)

ax4.set_xticks(range(len(violin_labels)))
ax4.set_xticklabels(violin_labels, fontsize=11)
ax4.set_ylabel('Novel 16S miTag Rate (%)', fontweight='bold', fontsize=11, color='#9B59B6')
ax4.tick_params(axis='y', labelcolor='#9B59B6')

# Create separate x-axis for depth-dependent scatter plots
ax4_twin = ax4.twinx()
ax4_depth = ax4.twiny()

# Plot depth-dependent changes in community structure
depth_profile = supp_fig4ab[supp_fig4ab['Sample'].str.contains('T3L11|T1L10')]
if not depth_profile.empty:
    colors_depth = ['#E67E22', '#27AE60']
    for i, sample_prefix in enumerate(['T3L11', 'T1L10']):
        sample_data = depth_profile[depth_profile['Sample'].str.contains(sample_prefix)]
        if not sample_data.empty:
            # Plot against the top x-axis (depth)
            ax4_depth.plot(sample_data['cmbsf'], sample_data['Novel 16S miTag rates'], 
                          marker='o', linewidth=3, markersize=8, 
                          label=f'{sample_prefix} Depth Profile', alpha=0.9,
                          color=colors_depth[i], markeredgecolor='white', markeredgewidth=2)

ax4_twin.set_ylabel('Novel 16S miTag Rate (Depth)', fontweight='bold', fontsize=11, color='#E67E22')
ax4_twin.tick_params(axis='y', labelcolor='#E67E22')
ax4_depth.set_xlabel('Sediment Depth (cmbsf)', fontweight='bold', fontsize=11, color='#27AE60')
ax4_depth.tick_params(axis='x', labelcolor='#27AE60')

if not depth_profile.empty:
    ax4_depth.legend(bbox_to_anchor=(1.02, 0.8), loc='upper left', fontsize=10)

ax4.set_title('Community Structure Analysis\n(Violin: Trench Position, Lines: Depth Changes)', 
              fontweight='bold', fontsize=13)
ax4.set_xlabel('Trench Position', fontweight='bold', fontsize=11)

# Adjust layout with better spacing
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()