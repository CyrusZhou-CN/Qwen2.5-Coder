import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('Supplementary Fig.10cd.csv')

# Identify major phyla (those with at least 10 genomes)
phylum_counts = df['Phylum'].value_counts()
major_phyla = phylum_counts[phylum_counts >= 10].index.tolist()

# Calculate overall KDE for reference
kde_x = np.linspace(df['CAZyme rate'].min(), df['CAZyme rate'].max(), 200)
overall_kde = stats.gaussian_kde(df['CAZyme rate'])

# Set up the figure with 4x4 grid layout for better balance
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.patch.set_facecolor('white')

# Flatten axes array for easier indexing
axes = axes.flatten()

# Use a professional color palette (ColorBrewer Set3)
colors = plt.cm.Set3(np.linspace(0, 1, len(major_phyla)))

# Create histogram bins (using 9 bins as specified)
bins = 9
bin_edges = np.histogram_bin_edges(df['CAZyme rate'], bins=bins)

# Calculate global y-axis limit for standardization
all_densities = []
for phylum in major_phyla:
    phylum_data = df[df['Phylum'] == phylum]['CAZyme rate']
    hist_vals, _ = np.histogram(phylum_data, bins=bin_edges, density=True)
    all_densities.extend(hist_vals)
y_max = max(all_densities) * 1.1

# Create subplot for each major phylum
for i, phylum in enumerate(major_phyla):
    ax = axes[i]
    
    # Get data for this phylum
    phylum_data = df[df['Phylum'] == phylum]['CAZyme rate']
    
    # Create histogram for this phylum
    ax.hist(phylum_data, bins=bin_edges, alpha=0.7, color=colors[i], 
            density=True, edgecolor='white', linewidth=1, 
            label=phylum)
    
    # Add overall KDE as reference (light gray)
    ax.plot(kde_x, overall_kde(kde_x), 'k-', linewidth=2, alpha=0.4, 
            label='Overall distribution')
    
    # Add KDE for this phylum
    if len(phylum_data) > 1:
        phylum_kde = stats.gaussian_kde(phylum_data)
        ax.plot(kde_x, phylum_kde(kde_x), color='darkred', 
                linewidth=2.5, label=f'{phylum} KDE')
    
    # Customize each subplot
    ax.set_title(f'{phylum}\n(n={len(phylum_data)} genomes)', 
                fontweight='bold', fontsize=11, pad=8)
    ax.set_xlabel('CAZyme Rate', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set consistent axis limits for all subplots
    ax.set_xlim(df['CAZyme rate'].min() * 0.95, df['CAZyme rate'].max() * 1.05)
    ax.set_ylim(0, y_max)
    
    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=8)

# Create color legend in the next available subplot
legend_ax = axes[len(major_phyla)]
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis('off')

# Add color legend patches
legend_elements = []
for i, phylum in enumerate(major_phyla):
    from matplotlib.patches import Patch
    legend_elements.append(Patch(facecolor=colors[i], alpha=0.7, label=phylum))

legend_ax.legend(handles=legend_elements, loc='center', fontsize=9, 
                title='Phylum Colors', title_fontsize=10, 
                bbox_to_anchor=(0.5, 0.5), frameon=True, 
                fancybox=True, shadow=True)

# Add summary statistics in another subplot
stats_ax = axes[len(major_phyla) + 1]
stats_ax.set_xlim(0, 1)
stats_ax.set_ylim(0, 1)
stats_ax.axis('off')

stats_text = (f'Dataset Summary:\n\n'
              f'• Total genomes: {len(df)}\n'
              f'• Major phyla (≥10 genomes): {len(major_phyla)}\n'
              f'• Mean CAZyme rate: {df["CAZyme rate"].mean():.4f}\n'
              f'• Std deviation: {df["CAZyme rate"].std():.4f}\n'
              f'• Histogram bins: {bins}\n\n'
              f'Legend:\n'
              f'• Colored bars: Phylum-specific distribution\n'
              f'• Black line: Overall distribution KDE\n'
              f'• Red line: Phylum-specific KDE')

stats_ax.text(0.05, 0.95, stats_text, transform=stats_ax.transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
             fontsize=9)

# Hide remaining empty subplots
for j in range(len(major_phyla) + 2, len(axes)):
    axes[j].set_visible(False)

# Add main title with proper spacing
fig.suptitle('CAZyme Rate Distributions in Deep-Sea Microbial Phyla', 
             fontsize=18, fontweight='bold', y=0.95)

# Ensure proper spacing and no overlap
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, 
                   hspace=0.35, wspace=0.25)
plt.show()