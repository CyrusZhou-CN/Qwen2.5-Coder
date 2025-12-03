import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('wine_quality_classification.csv')

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define colors for quality groups
quality_colors = {'low': '#e74c3c', 'medium': '#f39c12', 'high': '#27ae60'}
quality_order = ['low', 'medium', 'high']

# Row 1, Column 1: Histogram with KDE overlay for fixed_acidity by quality_label
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')
for quality in quality_order:
    data = df[df['quality_label'] == quality]['fixed_acidity']
    ax1.hist(data, alpha=0.6, bins=20, density=True, label=f'{quality.capitalize()}', 
             color=quality_colors[quality], edgecolor='white', linewidth=0.5)
    # Add KDE overlay
    kde_x = np.linspace(data.min(), data.max(), 100)
    kde = stats.gaussian_kde(data)
    ax1.plot(kde_x, kde(kde_x), color=quality_colors[quality], linewidth=2)

ax1.set_title('Fixed Acidity Distribution by Quality\n(Histogram + KDE)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Fixed Acidity')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Row 1, Column 2: Box plot with violin plot overlay for residual_sugar by quality_label
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')
# Violin plot
parts = ax2.violinplot([df[df['quality_label'] == q]['residual_sugar'] for q in quality_order], 
                       positions=range(len(quality_order)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(quality_colors[quality_order[i]])
    pc.set_alpha(0.6)
# Box plot overlay
bp = ax2.boxplot([df[df['quality_label'] == q]['residual_sugar'] for q in quality_order], 
                 positions=range(len(quality_order)), patch_artist=True, widths=0.3)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(quality_colors[quality_order[i]])
    patch.set_alpha(0.8)

ax2.set_title('Residual Sugar Distribution by Quality\n(Violin + Box Plot)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Quality Label')
ax2.set_ylabel('Residual Sugar')
ax2.set_xticks(range(len(quality_order)))
ax2.set_xticklabels([q.capitalize() for q in quality_order])
ax2.grid(True, alpha=0.3)

# Row 1, Column 3: Stacked bar chart with line plot overlay for alcohol content ranges
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')
# Create alcohol ranges
df['alcohol_range'] = pd.cut(df['alcohol'], bins=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
alcohol_quality = pd.crosstab(df['alcohol_range'], df['quality_label'])
alcohol_quality = alcohol_quality.reindex(columns=quality_order)

# Stacked bar chart
bottom = np.zeros(len(alcohol_quality))
for i, quality in enumerate(quality_order):
    ax3.bar(alcohol_quality.index, alcohol_quality[quality], bottom=bottom, 
            label=quality.capitalize(), color=quality_colors[quality], alpha=0.8)
    bottom += alcohol_quality[quality]

# Line plot overlay for mean alcohol content
ax3_twin = ax3.twinx()
mean_alcohol = df.groupby('quality_label')['alcohol'].mean().reindex(quality_order)
ax3_twin.plot(range(len(quality_order)), mean_alcohol.values, 'ko-', linewidth=3, markersize=8, label='Mean Alcohol')

ax3.set_title('Alcohol Content Distribution by Quality\n(Stacked Bar + Line)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Alcohol Range')
ax3.set_ylabel('Count')
ax3_twin.set_ylabel('Mean Alcohol Content')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Row 2, Column 1: Scatter plot with regression lines and marginal histograms
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')
for quality in quality_order:
    data = df[df['quality_label'] == quality]
    ax4.scatter(data['fixed_acidity'], data['density'], alpha=0.6, 
               label=quality.capitalize(), color=quality_colors[quality], s=30)
    # Add regression line
    z = np.polyfit(data['fixed_acidity'], data['density'], 1)
    p = np.poly1d(z)
    x_reg = np.linspace(data['fixed_acidity'].min(), data['fixed_acidity'].max(), 100)
    ax4.plot(x_reg, p(x_reg), color=quality_colors[quality], linewidth=2, linestyle='--')

ax4.set_title('Fixed Acidity vs Density by Quality\n(Scatter + Regression)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Fixed Acidity')
ax4.set_ylabel('Density')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 2, Column 2: Bubble plot
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')
for quality in quality_order:
    data = df[df['quality_label'] == quality]
    ax5.scatter(data['alcohol'], data['residual_sugar'], 
               s=data['density']*1000, alpha=0.6, 
               label=quality.capitalize(), color=quality_colors[quality])

ax5.set_title('Alcohol vs Residual Sugar\n(Bubble size = Density)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Alcohol')
ax5.set_ylabel('Residual Sugar')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Row 2, Column 3: Parallel coordinates plot
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')
# Normalize data for parallel coordinates
numerical_cols = ['fixed_acidity', 'residual_sugar', 'alcohol', 'density']
df_norm = df.copy()
for col in numerical_cols:
    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Sample data for cleaner visualization
sample_size = 100
for quality in quality_order:
    data = df_norm[df_norm['quality_label'] == quality].sample(min(sample_size, len(df_norm[df_norm['quality_label'] == quality])))
    for i in range(len(data)):
        ax6.plot(range(len(numerical_cols)), data.iloc[i][numerical_cols], 
                color=quality_colors[quality], alpha=0.3, linewidth=0.8)

ax6.set_title('Parallel Coordinates by Quality\n(Normalized Features)', fontweight='bold', fontsize=12)
ax6.set_xticks(range(len(numerical_cols)))
ax6.set_xticklabels([col.replace('_', ' ').title() for col in numerical_cols], rotation=45)
ax6.set_ylabel('Normalized Value')
ax6.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=quality_colors[q], lw=2, label=q.capitalize()) for q in quality_order]
ax6.legend(handles=legend_elements)

# Row 3, Column 1: Radar chart
ax7 = plt.subplot(3, 3, 7, projection='polar')
ax7.set_facecolor('white')
# Calculate mean values for each quality group
means = df.groupby('quality_label')[numerical_cols].mean()
angles = np.linspace(0, 2*np.pi, len(numerical_cols), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for quality in quality_order:
    values = means.loc[quality].tolist()
    values += values[:1]  # Complete the circle
    ax7.plot(angles, values, 'o-', linewidth=2, label=quality.capitalize(), 
             color=quality_colors[quality])
    ax7.fill(angles, values, alpha=0.25, color=quality_colors[quality])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels([col.replace('_', ' ').title() for col in numerical_cols])
ax7.set_title('Mean Feature Values by Quality\n(Radar Chart)', fontweight='bold', fontsize=12, pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax7.grid(True, alpha=0.3)

# Row 3, Column 2: PCA cluster plot
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')
# Perform PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_cols])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for quality in quality_order:
    mask = df['quality_label'] == quality
    ax8.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, 
               label=quality.capitalize(), color=quality_colors[quality], s=40)

ax8.set_title(f'PCA Cluster Analysis\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2%})', 
              fontweight='bold', fontsize=12)
ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Row 3, Column 3: Correlation heatmap with dendrograms
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')
# Calculate correlation matrix for each quality group
fig_temp = plt.figure(figsize=(8, 6))
fig_temp.patch.set_facecolor('white')

# Create subplot for combined heatmap
gs = fig_temp.add_gridspec(1, 3, width_ratios=[1, 1, 1])
axes_hm = [fig_temp.add_subplot(gs[0, i]) for i in range(3)]

for i, quality in enumerate(quality_order):
    data_subset = df[df['quality_label'] == quality][numerical_cols]
    corr_matrix = data_subset.corr()
    
    # Create heatmap
    im = axes_hm[i].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes_hm[i].set_xticks(range(len(numerical_cols)))
    axes_hm[i].set_yticks(range(len(numerical_cols)))
    axes_hm[i].set_xticklabels([col.replace('_', '\n') for col in numerical_cols], fontsize=8)
    axes_hm[i].set_yticklabels([col.replace('_', '\n') for col in numerical_cols], fontsize=8)
    axes_hm[i].set_title(f'{quality.capitalize()} Quality\nCorrelation Matrix', fontweight='bold', fontsize=10)
    
    # Add correlation values
    for row in range(len(numerical_cols)):
        for col in range(len(numerical_cols)):
            axes_hm[i].text(col, row, f'{corr_matrix.iloc[row, col]:.2f}', 
                           ha='center', va='center', fontsize=8, 
                           color='white' if abs(corr_matrix.iloc[row, col]) > 0.5 else 'black')

# Add colorbar
cbar = fig_temp.colorbar(im, ax=axes_hm, shrink=0.8, aspect=20)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

plt.tight_layout()

# Copy the heatmap content to the main subplot
ax9.text(0.5, 0.5, 'Correlation Matrices by Quality Group\n(See separate detailed view)', 
         ha='center', va='center', transform=ax9.transAxes, fontsize=12, fontweight='bold')
ax9.set_xticks([])
ax9.set_yticks([])

# Main figure layout adjustment
plt.figure(fig.number)
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

# Show the detailed heatmap figure
plt.figure(fig_temp.number)
plt.show()