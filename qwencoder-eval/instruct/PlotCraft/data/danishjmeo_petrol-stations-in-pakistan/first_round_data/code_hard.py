import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
df = pd.read_csv('petrol-stations-in-pakistan.csv')

# Clean column names and data
df.columns = ['S_No', 'Customer_Name', 'Address', 'City_Location', 'HSD', 'PMG', 'District', 'District1', 'Province', 'Province1']

# Clean and convert price columns
df['HSD'] = pd.to_numeric(df['HSD'], errors='coerce')
df['PMG'] = pd.to_numeric(df['PMG'], errors='coerce')

# Remove rows with missing price data
df_clean = df.dropna(subset=['HSD', 'PMG']).copy()

# Clean province names
df_clean['Province'] = df_clean['Province'].fillna(df_clean['Province1'])
df_clean = df_clean.dropna(subset=['Province'])

# Filter out extreme outliers
q1_hsd, q3_hsd = df_clean['HSD'].quantile([0.25, 0.75])
q1_pmg, q3_pmg = df_clean['PMG'].quantile([0.25, 0.75])
iqr_hsd = q3_hsd - q1_hsd
iqr_pmg = q3_pmg - q1_pmg

df_clean = df_clean[
    (df_clean['HSD'] >= q1_hsd - 1.5 * iqr_hsd) & 
    (df_clean['HSD'] <= q3_hsd + 1.5 * iqr_hsd) &
    (df_clean['PMG'] >= q1_pmg - 1.5 * iqr_pmg) & 
    (df_clean['PMG'] <= q3_pmg + 1.5 * iqr_pmg)
]

# Get top provinces by station count
province_counts = df_clean['Province'].value_counts()
top_provinces = province_counts.head(4).index.tolist()
df_viz = df_clean[df_clean['Province'].isin(top_provinces)].copy()

# Create color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
province_colors = dict(zip(top_provinces, colors))

# Create 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# 1. Top-left: Scatter plot with density contours and trend lines
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

for i, province in enumerate(top_provinces):
    prov_data = df_viz[df_viz['Province'] == province]
    if len(prov_data) > 10:
        # Scatter plot
        ax1.scatter(prov_data['HSD'], prov_data['PMG'], 
                   c=province_colors[province], alpha=0.6, s=30, label=province)
        
        # Trend line
        z = np.polyfit(prov_data['HSD'], prov_data['PMG'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(prov_data['HSD'].min(), prov_data['HSD'].max(), 100)
        ax1.plot(x_trend, p(x_trend), color=province_colors[province], linewidth=2, linestyle='--')

ax1.set_xlabel('HSD Price (Rs/Liter)', fontweight='bold')
ax1.set_ylabel('PMG Price (Rs/Liter)', fontweight='bold')
ax1.set_title('Fuel Price Correlation by Province', fontweight='bold', fontsize=12)
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# 2. Top-center: Grouped violin plots
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Prepare data for violin plot
violin_data = []
violin_labels = []
violin_colors = []

for province in top_provinces:
    prov_data = df_viz[df_viz['Province'] == province]
    if len(prov_data) > 5:
        violin_data.extend([prov_data['HSD'].values, prov_data['PMG'].values])
        violin_labels.extend([f'{province}\nHSD', f'{province}\nPMG'])
        violin_colors.extend([province_colors[province], province_colors[province]])

parts = ax2.violinplot(violin_data, positions=range(len(violin_data)), widths=0.8)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(violin_colors[i])
    pc.set_alpha(0.7)

ax2.set_xticks(range(len(violin_labels)))
ax2.set_xticklabels(violin_labels, rotation=45, ha='right')
ax2.set_ylabel('Price (Rs/Liter)', fontweight='bold')
ax2.set_title('Fuel Price Distributions by Province', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# 3. Top-right: Hierarchical cluster heatmap
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Sample data for clustering (take subset for visualization)
sample_size = min(100, len(df_viz))
df_sample = df_viz.sample(n=sample_size, random_state=42)

# Create price matrix
price_matrix = df_sample[['HSD', 'PMG']].values
scaler = StandardScaler()
price_matrix_scaled = scaler.fit_transform(price_matrix)

# Hierarchical clustering
linkage_matrix = linkage(price_matrix_scaled, method='ward')

# Create heatmap
im = ax3.imshow(price_matrix_scaled, cmap='RdYlBu_r', aspect='auto')
ax3.set_xlabel('Fuel Type', fontweight='bold')
ax3.set_ylabel('Station Index', fontweight='bold')
ax3.set_title('Station Clustering Heatmap', fontweight='bold', fontsize=12)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['HSD', 'PMG'])

# 4. Middle-left: Network-style visualization (simplified as scatter with connections)
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# City-based analysis
city_stats = df_viz.groupby(['City_Location', 'Province']).agg({
    'HSD': 'mean',
    'PMG': 'mean',
    'S_No': 'count'
}).reset_index()
city_stats = city_stats[city_stats['S_No'] >= 3].head(20)  # Top cities with multiple stations

for province in top_provinces:
    prov_cities = city_stats[city_stats['Province'] == province]
    if len(prov_cities) > 0:
        ax4.scatter(prov_cities['HSD'], prov_cities['PMG'], 
                   s=prov_cities['S_No']*20, c=province_colors[province], 
                   alpha=0.7, label=province)

ax4.set_xlabel('Average HSD Price (Rs/Liter)', fontweight='bold')
ax4.set_ylabel('Average PMG Price (Rs/Liter)', fontweight='bold')
ax4.set_title('City-Level Pricing Patterns', fontweight='bold', fontsize=12)
ax4.legend(frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3)

# 5. Middle-center: Parallel coordinates plot (simplified)
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Normalize data for parallel coordinates
features = ['HSD', 'PMG']
df_norm = df_viz[features + ['Province']].copy()
for feature in features:
    df_norm[feature] = (df_norm[feature] - df_norm[feature].min()) / (df_norm[feature].max() - df_norm[feature].min())

# Sample for visualization
sample_data = df_norm.sample(n=min(200, len(df_norm)), random_state=42)

for province in top_provinces:
    prov_data = sample_data[sample_data['Province'] == province]
    if len(prov_data) > 0:
        for idx, row in prov_data.iterrows():
            ax5.plot([0, 1], [row['HSD'], row['PMG']], 
                    color=province_colors[province], alpha=0.3, linewidth=0.5)

# Add mean lines
for province in top_provinces:
    prov_data = df_norm[df_norm['Province'] == province]
    if len(prov_data) > 0:
        mean_vals = [prov_data['HSD'].mean(), prov_data['PMG'].mean()]
        ax5.plot([0, 1], mean_vals, color=province_colors[province], 
                linewidth=3, label=f'{province} (avg)')

ax5.set_xticks([0, 1])
ax5.set_xticklabels(['HSD', 'PMG'])
ax5.set_ylabel('Normalized Price', fontweight='bold')
ax5.set_title('Parallel Coordinates: Fuel Prices', fontweight='bold', fontsize=12)
ax5.legend(frameon=True, fancybox=True, shadow=True)
ax5.grid(True, alpha=0.3)

# 6. Middle-right: Treemap simulation (using nested bars)
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

province_summary = df_viz.groupby('Province').agg({
    'S_No': 'count',
    'HSD': 'mean',
    'PMG': 'mean'
}).reset_index()

y_pos = np.arange(len(province_summary))
bars = ax6.barh(y_pos, province_summary['S_No'], 
                color=[province_colors[p] for p in province_summary['Province']])

# Add price information as text
for i, (idx, row) in enumerate(province_summary.iterrows()):
    ax6.text(row['S_No']/2, i, f'HSD: {row["HSD"]:.1f}\nPMG: {row["PMG"]:.1f}', 
             ha='center', va='center', fontweight='bold', fontsize=8)

ax6.set_yticks(y_pos)
ax6.set_yticklabels(province_summary['Province'])
ax6.set_xlabel('Number of Stations', fontweight='bold')
ax6.set_title('Station Count & Avg Prices by Province', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3, axis='x')

# 7. Bottom-left: Radar chart
ax7 = plt.subplot(3, 3, 7, projection='polar')
ax7.set_facecolor('white')

# Prepare radar chart data
categories = ['HSD Price', 'PMG Price', 'Price Variance', 'Station Density']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for province in top_provinces:
    prov_data = df_viz[df_viz['Province'] == province]
    if len(prov_data) > 0:
        values = [
            prov_data['HSD'].mean() / df_viz['HSD'].max(),
            prov_data['PMG'].mean() / df_viz['PMG'].max(),
            prov_data['HSD'].std() / df_viz['HSD'].std(),
            len(prov_data) / len(df_viz)
        ]
        values += values[:1]  # Complete the circle
        
        ax7.plot(angles, values, 'o-', linewidth=2, 
                label=province, color=province_colors[province])
        ax7.fill(angles, values, alpha=0.25, color=province_colors[province])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(categories)
ax7.set_title('Province Comparison Radar', fontweight='bold', fontsize=12, pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 8. Bottom-center: 2D histogram with marginals
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Create 2D histogram
hist, xedges, yedges = np.histogram2d(df_viz['HSD'], df_viz['PMG'], bins=20)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

im = ax8.imshow(hist.T, origin='lower', extent=extent, cmap='Blues', aspect='auto')
ax8.set_xlabel('HSD Price (Rs/Liter)', fontweight='bold')
ax8.set_ylabel('PMG Price (Rs/Liter)', fontweight='bold')
ax8.set_title('Joint Price Distribution', fontweight='bold', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax8, shrink=0.8)
cbar.set_label('Station Count', fontweight='bold')

# 9. Bottom-right: Cluster validation (Elbow method)
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Prepare data for clustering
X = df_viz[['HSD', 'PMG']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
ax9_twin = ax9.twinx()
line1 = ax9.plot(k_range, inertias, 'bo-', linewidth=2, markersize=6, label='Inertia')
line2 = ax9_twin.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=6, label='Silhouette Score')

ax9.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax9.set_ylabel('Inertia', fontweight='bold', color='blue')
ax9_twin.set_ylabel('Silhouette Score', fontweight='bold', color='red')
ax9.set_title('Cluster Validation Analysis', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax9.get_legend_handles_labels()
lines2, labels2 = ax9_twin.get_legend_handles_labels()
ax9.legend(lines1 + lines2, labels1 + labels2, loc='center right')

# Overall layout adjustment
plt.suptitle('Comprehensive Analysis of Petrol Station Clusters and Regional Patterns in Pakistan', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()