import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Load data
districts_df = pd.read_csv('district_of_seoul.csv')
cameras_df = pd.read_csv('fixed_cctv_for_parking_enforcement.csv')

# Data preprocessing
# Calculate camera counts per district
camera_counts = cameras_df['district'].value_counts().reset_index()
camera_counts.columns = ['name', 'camera_count']

# Merge with district data
merged_df = districts_df.merge(camera_counts, on='name', how='left')
merged_df['camera_count'] = merged_df['camera_count'].fillna(0)
merged_df['camera_density'] = merged_df['camera_count'] / merged_df['area']
merged_df['pop_density'] = merged_df['population'] / merged_df['area']

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Color schemes
district_colors = plt.cm.Set3(np.linspace(0, 1, len(merged_df)))
cluster_colors = plt.cm.viridis

# Top row - District-level analysis
# Subplot 1: Horizontal bar chart with population density scatter
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')
top_districts = merged_df.nlargest(15, 'camera_count')
bars = ax1.barh(range(len(top_districts)), top_districts['camera_count'], 
                color='steelblue', alpha=0.7, edgecolor='white', linewidth=1)
ax1_twin = ax1.twiny()
scatter = ax1_twin.scatter(top_districts['pop_density'], range(len(top_districts)), 
                          c='red', s=80, alpha=0.8, marker='o', edgecolor='white', linewidth=1)
ax1.set_yticks(range(len(top_districts)))
ax1.set_yticklabels(top_districts['name'], fontsize=10)
ax1.set_xlabel('Camera Count', fontweight='bold', fontsize=11)
ax1_twin.set_xlabel('Population Density (per km²)', fontweight='bold', fontsize=11, color='red')
ax1.set_title('Camera Count vs Population Density by District', fontweight='bold', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Subplot 2: Bubble chart
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')
bubble_data = merged_df[merged_df['camera_count'] > 0]
scatter = ax2.scatter(bubble_data['longitude'], bubble_data['latitude'], 
                     s=bubble_data['camera_density']*500, 
                     c=bubble_data['population'], cmap='plasma', 
                     alpha=0.7, edgecolor='white', linewidth=1)
for i, row in bubble_data.iterrows():
    if row['camera_count'] > 50:
        ax2.annotate(row['name'], (row['longitude'], row['latitude']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax2.set_xlabel('Longitude', fontweight='bold', fontsize=11)
ax2.set_ylabel('Latitude', fontweight='bold', fontsize=11)
ax2.set_title('Camera Density Bubbles by District Location', fontweight='bold', fontsize=14, pad=20)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Population', fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Subplot 3: Violin plot with strip plot
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')
top_5_districts = merged_df.nlargest(5, 'camera_count')['name'].tolist()
camera_subset = cameras_df[cameras_df['district'].isin(top_5_districts)]
violin_parts = ax3.violinplot([camera_subset[camera_subset['district'] == d]['latitude'].values 
                              for d in top_5_districts], 
                             positions=range(len(top_5_districts)), widths=0.6, showmeans=True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)
    pc.set_edgecolor('white')
for i, district in enumerate(top_5_districts):
    district_cameras = camera_subset[camera_subset['district'] == district]
    y_jitter = np.random.normal(i, 0.05, len(district_cameras))
    ax3.scatter(y_jitter, district_cameras['latitude'], alpha=0.6, s=20, color='red')
ax3.set_xticks(range(len(top_5_districts)))
ax3.set_xticklabels(top_5_districts, rotation=45, ha='right')
ax3.set_ylabel('Latitude', fontweight='bold', fontsize=11)
ax3.set_title('Latitude Distribution of Cameras (Top 5 Districts)', fontweight='bold', fontsize=14, pad=20)
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Middle row - Geographic clustering analysis
# Subplot 4: Scatter plot with density contours
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')
district_names = cameras_df['district'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(district_names)))
for i, district in enumerate(district_names):
    district_cameras = cameras_df[cameras_df['district'] == district]
    ax4.scatter(district_cameras['longitude'], district_cameras['latitude'], 
               c=[colors[i]], alpha=0.6, s=15, label=district if i < 10 else "")
# Add density contours
x = cameras_df['longitude'].values
y = cameras_df['latitude'].values
ax4.hexbin(x, y, gridsize=20, alpha=0.3, cmap='Blues')
ax4.set_xlabel('Longitude', fontweight='bold', fontsize=11)
ax4.set_ylabel('Latitude', fontweight='bold', fontsize=11)
ax4.set_title('Camera Locations with Density Contours', fontweight='bold', fontsize=14, pad=20)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Subplot 5: 2D histogram with marginals
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')
h = ax5.hist2d(cameras_df['longitude'], cameras_df['latitude'], bins=25, cmap='YlOrRd')
plt.colorbar(h[3], ax=ax5, label='Camera Count')
ax5.set_xlabel('Longitude', fontweight='bold', fontsize=11)
ax5.set_ylabel('Latitude', fontweight='bold', fontsize=11)
ax5.set_title('2D Histogram of Camera Locations', fontweight='bold', fontsize=14, pad=20)

# Subplot 6: Hierarchical clustering
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')
# Sample cameras for clustering (too many for visualization)
sample_cameras = cameras_df.sample(n=min(500, len(cameras_df)), random_state=42)
coords = sample_cameras[['longitude', 'latitude']].values
linkage_matrix = linkage(coords, method='ward')
clusters = fcluster(linkage_matrix, t=8, criterion='maxclust')
scatter = ax6.scatter(sample_cameras['longitude'], sample_cameras['latitude'], 
                     c=clusters, cmap='tab10', s=30, alpha=0.7, edgecolor='white', linewidth=0.5)
ax6.set_xlabel('Longitude', fontweight='bold', fontsize=11)
ax6.set_ylabel('Latitude', fontweight='bold', fontsize=11)
ax6.set_title('Hierarchical Clustering of Camera Locations', fontweight='bold', fontsize=14, pad=20)
plt.colorbar(scatter, ax=ax6, label='Cluster')
ax6.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Bottom row - Comparative district analysis
# Subplot 7: Radar chart
ax7 = plt.subplot(3, 3, 7, projection='polar')
ax7.set_facecolor('white')
top_5 = merged_df.nlargest(5, 'camera_count')
metrics = ['camera_count', 'population', 'area', 'camera_density']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for i, (_, district) in enumerate(top_5.iterrows()):
    values = []
    for metric in metrics:
        # Normalize values to 0-1 scale
        max_val = merged_df[metric].max()
        min_val = merged_df[metric].min()
        normalized = (district[metric] - min_val) / (max_val - min_val)
        values.append(normalized)
    values += values[:1]
    ax7.plot(angles, values, 'o-', linewidth=2, label=district['name'], alpha=0.8)
    ax7.fill(angles, values, alpha=0.1)

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metrics, fontweight='bold')
ax7.set_title('District Comparison Radar Chart (Top 5)', fontweight='bold', fontsize=14, pad=30)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax7.grid(True, alpha=0.3)

# Subplot 8: Box plots with swarm plots
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')
top_districts_list = merged_df.nlargest(6, 'camera_count')['name'].tolist()
camera_coords = []
district_labels = []
for district in top_districts_list:
    district_cameras = cameras_df[cameras_df['district'] == district]
    camera_coords.extend(district_cameras['latitude'].tolist())
    district_labels.extend([district] * len(district_cameras))

coord_df = pd.DataFrame({'latitude': camera_coords, 'district': district_labels})
box_plot = ax8.boxplot([coord_df[coord_df['district'] == d]['latitude'].values 
                       for d in top_districts_list], 
                      labels=top_districts_list, patch_artist=True)
for patch in box_plot['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

# Add swarm plot overlay
for i, district in enumerate(top_districts_list):
    district_data = coord_df[coord_df['district'] == district]['latitude'].values
    x_pos = np.random.normal(i+1, 0.04, len(district_data))
    ax8.scatter(x_pos, district_data, alpha=0.4, s=8, color='red')

ax8.set_xticklabels(top_districts_list, rotation=45, ha='right')
ax8.set_ylabel('Latitude', fontweight='bold', fontsize=11)
ax8.set_title('Camera Latitude Distribution by District', fontweight='bold', fontsize=14, pad=20)
ax8.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Subplot 9: Correlation matrix heatmap
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')
corr_metrics = merged_df[['population', 'area', 'camera_count', 'camera_density', 'pop_density']]
correlation_matrix = corr_metrics.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                     center=0, square=True, ax=ax9, cbar_kws={'shrink': 0.8})
ax9.set_title('District Metrics Correlation Matrix', fontweight='bold', fontsize=14, pad=20)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()