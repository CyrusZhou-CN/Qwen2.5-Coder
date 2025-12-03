import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('List of fishes found in Sweden.csv')

# Data preprocessing
df = df.dropna(subset=['Family', 'Habitat', 'Occurrence', 'Red List Status'])

# Create derived metrics
family_counts = df['Family'].value_counts()
df['family_rarity_index'] = df['Family'].map(lambda x: 1/family_counts[x] if x in family_counts else 1)

habitat_diversity = df.groupby('Family')['Habitat'].nunique()
df['habitat_specialization'] = df['Family'].map(habitat_diversity)

# Conservation concern mapping
concern_map = {
    'Not evaluated': 0, 'Least concern (LC)': 1, 'Near threatened (NT)': 2,
    'Vulnerable (VU)': 3, 'Endangered (EN)': 4, 'Critically endangered (CR)': 5,
    'Disappeared (RE)': 6
}
df['conservation_concern'] = df['Red List Status'].map(concern_map).fillna(0)

# Set up the figure
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color palettes
status_colors = {'Not evaluated': '#95a5a6', 'Least concern (LC)': '#2ecc71', 
                'Near threatened (NT)': '#f39c12', 'Vulnerable (VU)': '#e67e22',
                'Endangered (EN)': '#e74c3c', 'Critically endangered (CR)': '#8e44ad',
                'Disappeared (RE)': '#34495e'}

# Subplot 1: Stacked bar chart with scatter overlay
ax1 = plt.subplot(3, 3, 1)
top_families = family_counts.head(8)
family_status = df[df['Family'].isin(top_families.index)].groupby(['Family', 'Red List Status']).size().unstack(fill_value=0)

# Stacked bar chart
bottom = np.zeros(len(family_status))
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#34495e']
for i, col in enumerate(family_status.columns):
    ax1.bar(range(len(family_status)), family_status[col], bottom=bottom, 
            label=col, color=colors[i % len(colors)], alpha=0.8)
    bottom += family_status[col]

# Scatter overlay
ax1.scatter(range(len(top_families)), top_families.values, color='red', s=80, zorder=5, alpha=0.9)

ax1.set_title('Family Diversity with Red List Distribution', fontweight='bold', fontsize=11)
ax1.set_xlabel('Fish Family')
ax1.set_ylabel('Species Count')
ax1.set_xticks(range(len(family_status)))
ax1.set_xticklabels([f[:15] for f in family_status.index], rotation=45, ha='right')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

# Subplot 2: Treemap simulation with bubble overlay
ax2 = plt.subplot(3, 3, 2)
top6_families = top_families.head(6)
family_occurrence = df[df['Family'].isin(top6_families.index)].groupby('Family')['Occurrence'].nunique()

# Create simple grid layout
positions = [(0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5), (0, 0, 0.33, 0.5), 
             (0.33, 0, 0.34, 0.5), (0.67, 0, 0.33, 0.5), (0.25, 0.25, 0.5, 0.25)]

colors = plt.cm.Set3(np.linspace(0, 1, 6))
for i, (family, count) in enumerate(family_occurrence.items()):
    if i < len(positions):
        x, y, w, h = positions[i]
        rect = Rectangle((x, y), w, h, facecolor=colors[i], alpha=0.6, edgecolor='black')
        ax2.add_patch(rect)
        ax2.text(x + w/2, y + h/2, f'{family[:12]}\n{count}', ha='center', va='center', fontsize=8)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Family Diversity Treemap', fontweight='bold', fontsize=11)
ax2.set_xticks([])
ax2.set_yticks([])

# Subplot 3: Radar chart
ax3 = plt.subplot(3, 3, 3, projection='polar')
metrics = []
for family in top6_families.index:
    family_data = df[df['Family'] == family]
    species_count = len(family_data)
    habitat_div = len(family_data['Habitat'].unique())
    concern_level = family_data['conservation_concern'].mean()
    metrics.append([species_count, habitat_div, concern_level])

metrics = np.array(metrics)
if metrics.max(axis=0).sum() > 0:
    metrics_norm = (metrics - metrics.min(axis=0)) / (metrics.max(axis=0) - metrics.min(axis=0) + 1e-8)
else:
    metrics_norm = metrics

angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
angles += angles[:1]

for i, family in enumerate(top6_families.index):
    values = metrics_norm[i].tolist()
    values += values[:1]
    ax3.plot(angles, values, 'o-', linewidth=2, label=family[:12], alpha=0.8)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(['Species', 'Habitat', 'Concern'])
ax3.set_title('Top 6 Families Comparison', fontweight='bold', fontsize=11, pad=20)
ax3.legend(bbox_to_anchor=(1.2, 1.0), fontsize=7)

# Subplot 4: Clustered bar chart
ax4 = plt.subplot(3, 3, 4)
habitat_data = df.groupby('Habitat').agg({
    'Family': 'count',
    'conservation_concern': ['mean', 'std']
}).round(2)

habitats = habitat_data.index[:6]  # Limit to 6 for visibility
counts = habitat_data.loc[habitats, ('Family', 'count')]
errors = habitat_data.loc[habitats, ('conservation_concern', 'std')].fillna(0)

bars = ax4.bar(range(len(habitats)), counts, yerr=errors, capsize=5, 
               color=plt.cm.viridis(np.linspace(0, 1, len(habitats))), alpha=0.8)

ax4.set_title('Species Count by Habitat with Conservation Variability', fontweight='bold', fontsize=11)
ax4.set_xlabel('Habitat Type')
ax4.set_ylabel('Species Count')
ax4.set_xticks(range(len(habitats)))
ax4.set_xticklabels([h[:15] for h in habitats], rotation=45, ha='right')

# Subplot 5: Network-style scatter plot
ax5 = plt.subplot(3, 3, 5)

# Encode categorical variables
habitat_encoder = LabelEncoder()
occurrence_encoder = LabelEncoder()
df['habitat_encoded'] = habitat_encoder.fit_transform(df['Habitat'])
df['occurrence_encoded'] = occurrence_encoder.fit_transform(df['Occurrence'])

# Sample data for performance
sample_df = df.sample(min(100, len(df)), random_state=42)

for i, status in enumerate(sample_df['Red List Status'].unique()):
    status_data = sample_df[sample_df['Red List Status'] == status]
    color = list(status_colors.values())[i % len(status_colors)]
    ax5.scatter(status_data['habitat_encoded'], status_data['occurrence_encoded'], 
               c=color, label=status[:15], alpha=0.7, s=40)

ax5.set_title('Species Network by Habitat-Occurrence', fontweight='bold', fontsize=11)
ax5.set_xlabel('Habitat (encoded)')
ax5.set_ylabel('Occurrence (encoded)')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

# Subplot 6: Parallel coordinates (simplified)
ax6 = plt.subplot(3, 3, 6)

# Sample for performance
sample_data = df.sample(min(50, len(df)), random_state=42)
parallel_cols = ['habitat_encoded', 'occurrence_encoded', 'conservation_concern']
parallel_data = sample_data[parallel_cols]

# Normalize data
for col in parallel_cols:
    col_min, col_max = parallel_data[col].min(), parallel_data[col].max()
    if col_max > col_min:
        parallel_data[col] = (parallel_data[col] - col_min) / (col_max - col_min)

x_coords = range(len(parallel_cols))
for i in range(len(parallel_data)):
    ax6.plot(x_coords, parallel_data.iloc[i], alpha=0.4, linewidth=0.8)

ax6.set_title('Parallel Coordinates Plot', fontweight='bold', fontsize=11)
ax6.set_xticks(x_coords)
ax6.set_xticklabels(['Habitat', 'Occurrence', 'Conservation'])
ax6.set_ylabel('Normalized Values')

# Subplot 7: Simplified dendrogram
ax7 = plt.subplot(3, 3, 7)

# Use smaller sample for clustering
cluster_sample = df.sample(min(30, len(df)), random_state=42)
cluster_features = cluster_sample[['habitat_encoded', 'occurrence_encoded', 'conservation_concern']]

try:
    linkage_matrix = linkage(cluster_features, method='ward')
    dendrogram(linkage_matrix, ax=ax7, leaf_rotation=90, leaf_font_size=6)
    ax7.set_title('Species Clustering Dendrogram', fontweight='bold', fontsize=11)
except:
    ax7.text(0.5, 0.5, 'Clustering visualization\nnot available', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Species Clustering', fontweight='bold', fontsize=11)

# Subplot 8: Box plots by conservation status
ax8 = plt.subplot(3, 3, 8)

status_diversity = []
status_labels = []
for status in df['Red List Status'].unique():
    status_data = df[df['Red List Status'] == status]
    diversity_scores = status_data['habitat_specialization'].values
    if len(diversity_scores) > 0:
        status_diversity.append(diversity_scores)
        status_labels.append(status[:15])

if status_diversity:
    bp = ax8.boxplot(status_diversity, labels=status_labels, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax8.set_title('Habitat Specialization by Conservation Status', fontweight='bold', fontsize=11)
ax8.set_xlabel('Conservation Status')
ax8.set_ylabel('Habitat Specialization')
ax8.tick_params(axis='x', rotation=45)

# Subplot 9: Multi-dimensional scatter plot
ax9 = plt.subplot(3, 3, 9)

scatter = ax9.scatter(df['family_rarity_index'], df['habitat_specialization'], 
                     c=df['conservation_concern'], s=50, alpha=0.7, cmap='viridis')

ax9.set_title('Family Rarity vs Habitat Specialization', fontweight='bold', fontsize=11)
ax9.set_xlabel('Family Rarity Index')
ax9.set_ylabel('Habitat Specialization')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax9, shrink=0.8)
cbar.set_label('Conservation Concern', fontsize=9)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.4)

# Save the plot
plt.savefig('fish_species_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()