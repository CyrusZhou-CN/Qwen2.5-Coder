import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('suicide_dataset.csv')

# Clean and prepare data
df_clean = df.dropna(subset=['Suicide Rate'])

# Check available sex categories
print("Available sex categories:", df_clean['Sex'].unique())

# Create pivot table with proper handling
df_pivot = df_clean.pivot_table(values='Suicide Rate', index='Country', columns='Sex', aggfunc='mean')
df_pivot = df_pivot.dropna()

print("Pivot table columns:", df_pivot.columns.tolist())
print("Pivot table shape:", df_pivot.shape)

# If we don't have enough data, create a simplified version
if df_pivot.empty or len(df_pivot.columns) < 2:
    print("Creating simplified analysis with available data...")
    # Use the raw data for analysis
    country_stats = df_clean.groupby(['Country', 'Sex'])['Suicide Rate'].mean().unstack(fill_value=0)
    df_pivot = country_stats.dropna()

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Define color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
colors_sex = {'Male': '#2E86AB', 'Female': '#A23B72', 'Both sexes': '#F18F01'}

# Get available columns for analysis
available_cols = df_pivot.columns.tolist()
print("Available columns for analysis:", available_cols)

# Subplot 1: Scatter plot with country analysis
ax1 = plt.subplot(3, 3, 1)
if len(available_cols) >= 2:
    col1, col2 = available_cols[0], available_cols[1]
    countries_sample = df_pivot.sample(min(30, len(df_pivot)), random_state=42)
    scatter = ax1.scatter(countries_sample[col1], countries_sample[col2], 
                         c=range(len(countries_sample)), cmap='viridis', alpha=0.7, s=60)
    ax1.plot([0, max(countries_sample[col1].max(), countries_sample[col2].max())], 
             [0, max(countries_sample[col1].max(), countries_sample[col2].max())], 
             'r--', alpha=0.5, linewidth=1)
    ax1.set_xlabel(f'{col1} Suicide Rate', fontweight='bold')
    ax1.set_ylabel(f'{col2} Suicide Rate', fontweight='bold')
    ax1.set_title(f'Country Suicide Rates: {col1} vs {col2}', fontweight='bold', fontsize=12)
else:
    # Fallback: show distribution of suicide rates by country
    country_means = df_clean.groupby('Country')['Suicide Rate'].mean().sort_values(ascending=False)
    top_countries = country_means.head(20)
    ax1.bar(range(len(top_countries)), top_countries.values, color=colors_main[0], alpha=0.7)
    ax1.set_xlabel('Countries', fontweight='bold')
    ax1.set_ylabel('Average Suicide Rate', fontweight='bold')
    ax1.set_title('Top 20 Countries by Suicide Rate', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(0, len(top_countries), 3))
    ax1.set_xticklabels([top_countries.index[i] for i in range(0, len(top_countries), 3)], rotation=45)

ax1.grid(True, alpha=0.3)

# Subplot 2: Hierarchical clustering with heatmap
ax2 = plt.subplot(3, 3, 2)
try:
    # Prepare data for clustering
    cluster_data = df_pivot.fillna(df_pivot.mean())
    if len(cluster_data) > 20:
        cluster_data = cluster_data.sample(20, random_state=42)
    
    if len(cluster_data) > 1:
        # Perform hierarchical clustering
        linkage_matrix = linkage(cluster_data, method='ward')
        dendrogram(linkage_matrix, labels=cluster_data.index, ax=ax2, orientation='top', 
                   leaf_rotation=90, leaf_font_size=8)
        ax2.set_title('Country Clustering by Suicide Rates', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Distance', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Country Clustering (Insufficient Data)', fontweight='bold', fontsize=12)
except Exception as e:
    ax2.text(0.5, 0.5, f'Clustering error: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Country Clustering (Error)', fontweight='bold', fontsize=12)

# Subplot 3: Radar chart comparison
ax3 = plt.subplot(3, 3, 3, projection='polar')
try:
    if len(df_pivot) > 10 and len(available_cols) > 1:
        # Get top 5 and bottom 5 countries by total suicide rate
        total_rates = df_pivot.mean(axis=1).sort_values()
        top5 = total_rates.tail(5).index
        bottom5 = total_rates.head(5).index
        
        angles = np.linspace(0, 2*np.pi, len(df_pivot.columns), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for i, country in enumerate(list(top5)[:3]):  # Show top 3 for clarity
            values = df_pivot.loc[country].values
            values = np.concatenate((values, [values[0]]))
            ax3.plot(angles, values, 'o-', linewidth=2, label=f'{country} (Top)', alpha=0.7)
        
        for i, country in enumerate(list(bottom5)[:3]):  # Show bottom 3 for clarity
            values = df_pivot.loc[country].values
            values = np.concatenate((values, [values[0]]))
            ax3.plot(angles, values, 's--', linewidth=2, label=f'{country} (Bottom)', alpha=0.7)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(df_pivot.columns, fontsize=10)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    else:
        ax3.text(0, 0, 'Insufficient data\nfor radar chart', ha='center', va='center', fontsize=12)
    
    ax3.set_title('Top vs Bottom Countries\nSuicide Rate Comparison', fontweight='bold', fontsize=12, pad=20)
except Exception as e:
    ax3.text(0, 0, f'Radar chart error', ha='center', va='center', fontsize=10)

# Subplot 4: Violin plots with strip plots
ax4 = plt.subplot(3, 3, 4)
sex_data = []
sex_labels = []
for sex in df_clean['Sex'].unique():
    rates = df_clean[df_clean['Sex'] == sex]['Suicide Rate'].dropna()
    if len(rates) > 0:
        sex_data.append(rates)
        sex_labels.append(sex)

if len(sex_data) > 0:
    violin_parts = ax4.violinplot(sex_data, positions=range(len(sex_labels)), showmeans=True, showmedians=True)
    for i, (data, label) in enumerate(zip(sex_data, sex_labels)):
        # Add strip plot
        y_jitter = np.random.normal(i, 0.05, min(len(data), 100))  # Limit points for visibility
        sample_data = data.sample(min(len(data), 100)) if len(data) > 100 else data
        ax4.scatter(y_jitter[:len(sample_data)], sample_data, alpha=0.4, s=20, 
                   color=colors_sex.get(label, 'gray'))
    
    ax4.set_xticks(range(len(sex_labels)))
    ax4.set_xticklabels(sex_labels, fontweight='bold')
    ax4.set_ylabel('Suicide Rate', fontweight='bold')
    ax4.set_title('Suicide Rate Distribution by Sex', fontweight='bold', fontsize=12)
else:
    ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Suicide Rate Distribution (No Data)', fontweight='bold', fontsize=12)

ax4.grid(True, alpha=0.3)

# Subplot 5: Parallel coordinates plot
ax5 = plt.subplot(3, 3, 5)
try:
    if len(df_pivot) > 5 and len(available_cols) > 1:
        sample_countries = df_pivot.sample(min(15, len(df_pivot)), random_state=42)
        normalized_data = (sample_countries - sample_countries.min()) / (sample_countries.max() - sample_countries.min())
        
        for i, (country, row) in enumerate(normalized_data.iterrows()):
            ax5.plot(range(len(row)), row.values, 'o-', alpha=0.6, linewidth=1.5, 
                     color=plt.cm.tab10(i % 10), label=country if i < 5 else "")
        
        ax5.set_xticks(range(len(df_pivot.columns)))
        ax5.set_xticklabels(df_pivot.columns, rotation=45, ha='right')
        ax5.set_ylabel('Normalized Suicide Rate', fontweight='bold')
        if len(normalized_data) <= 5:
            ax5.legend(fontsize=8, loc='upper right')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for\nparallel coordinates', ha='center', va='center', transform=ax5.transAxes)
    
    ax5.set_title('Country Patterns Across Sex Categories', fontweight='bold', fontsize=12)
except Exception as e:
    ax5.text(0.5, 0.5, 'Parallel coordinates\nerror', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Country Patterns (Error)', fontweight='bold', fontsize=12)

ax5.grid(True, alpha=0.3)

# Subplot 6: Network-style scatter plot
ax6 = plt.subplot(3, 3, 6)
try:
    if len(df_pivot) > 5 and len(available_cols) >= 2:
        # Perform clustering
        kmeans = KMeans(n_clusters=min(3, len(df_pivot)), random_state=42)
        clusters = kmeans.fit_predict(df_pivot.fillna(df_pivot.mean()))
        
        col1, col2 = available_cols[0], available_cols[1]
        scatter = ax6.scatter(df_pivot[col1], df_pivot[col2], 
                             c=clusters, cmap='Set1', alpha=0.7, s=60)
        
        # Add cluster centers
        centers = kmeans.cluster_centers_
        ax6.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        ax6.set_xlabel(f'{col1} Suicide Rate', fontweight='bold')
        ax6.set_ylabel(f'{col2} Suicide Rate', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor clustering', ha='center', va='center', transform=ax6.transAxes)
    
    ax6.set_title('Country Clusters by Sex-based Rates', fontweight='bold', fontsize=12)
except Exception as e:
    ax6.text(0.5, 0.5, 'Clustering error', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Country Clusters (Error)', fontweight='bold', fontsize=12)

ax6.grid(True, alpha=0.3)

# Subplot 7: Treemap-style bubble chart
ax7 = plt.subplot(3, 3, 7)
try:
    if len(df_pivot) > 0:
        total_rates = df_pivot.mean(axis=1)
        rate_ranges = pd.cut(total_rates, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        range_counts = rate_ranges.value_counts()
        
        # Create bubble chart
        for i, (range_name, count) in enumerate(range_counts.items()):
            x = i % 3
            y = i // 3
            size = max(count * 50, 100)  # Ensure minimum size
            ax7.scatter(x, y, s=size, alpha=0.6, color=colors_main[i % len(colors_main)])
            ax7.annotate(f'{range_name}\n({count} countries)', (x, y), 
                        ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax7.set_xlim(-0.5, 2.5)
        ax7.set_ylim(-0.5, 1.5)
    else:
        ax7.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax7.transAxes)
    
    ax7.set_title('Country Distribution by Suicide Rate Ranges', fontweight='bold', fontsize=12)
    ax7.set_xticks([])
    ax7.set_yticks([])
except Exception as e:
    ax7.text(0.5, 0.5, 'Bubble chart error', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Rate Distribution (Error)', fontweight='bold', fontsize=12)

# Subplot 8: MDS plot with convex hulls
ax8 = plt.subplot(3, 3, 8)
try:
    if len(df_pivot) > 5 and len(available_cols) > 1:
        # Prepare data for MDS
        mds_data = df_pivot.fillna(df_pivot.mean())
        if len(mds_data) > 30:
            mds_data = mds_data.sample(30, random_state=42)
        
        mds = MDS(n_components=2, random_state=42)
        mds_coords = mds.fit_transform(mds_data)
        
        # Perform clustering for hull boundaries
        kmeans_mds = KMeans(n_clusters=min(3, len(mds_data)), random_state=42)
        mds_clusters = kmeans_mds.fit_predict(mds_data)
        
        # Plot points
        for cluster in np.unique(mds_clusters):
            mask = mds_clusters == cluster
            ax8.scatter(mds_coords[mask, 0], mds_coords[mask, 1], 
                       c=colors_main[cluster], alpha=0.7, s=60, label=f'Cluster {cluster+1}')
        
        ax8.set_xlabel('MDS Dimension 1', fontweight='bold')
        ax8.set_ylabel('MDS Dimension 2', fontweight='bold')
        ax8.legend(fontsize=9)
    else:
        ax8.text(0.5, 0.5, 'Insufficient data\nfor MDS analysis', ha='center', va='center', transform=ax8.transAxes)
    
    ax8.set_title('Country Clusters in Reduced Space', fontweight='bold', fontsize=12)
except Exception as e:
    ax8.text(0.5, 0.5, 'MDS analysis error', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('MDS Analysis (Error)', fontweight='bold', fontsize=12)

ax8.grid(True, alpha=0.3)

# Subplot 9: Correlation heatmap with network overlay
ax9 = plt.subplot(3, 3, 9)
try:
    if len(df_pivot.columns) > 1:
        # Calculate correlation matrix
        corr_matrix = df_pivot.corr()
        
        # Create heatmap
        im = ax9.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax9.set_xticks(range(len(corr_matrix.columns)))
        ax9.set_yticks(range(len(corr_matrix.columns)))
        ax9.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax9.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax9.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontweight='bold')
    else:
        ax9.text(0.5, 0.5, 'Insufficient categories\nfor correlation analysis', ha='center', va='center', transform=ax9.transAxes)
    
    ax9.set_title('Sex Category Correlations Across Countries', fontweight='bold', fontsize=12)
except Exception as e:
    ax9.text(0.5, 0.5, 'Correlation analysis\nerror', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Correlation Analysis (Error)', fontweight='bold', fontsize=12)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add overall title
fig.suptitle('Comprehensive Suicide Rate Analysis: Demographic Patterns and Clustering', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('suicide_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()