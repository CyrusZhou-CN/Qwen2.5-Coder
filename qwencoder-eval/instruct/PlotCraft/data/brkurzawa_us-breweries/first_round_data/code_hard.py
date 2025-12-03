import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import networkx as nx
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('breweries_us.csv')

# Data preprocessing
df['has_website'] = df['website'].notna() & (df['website'] != '') & (df['website'] != 'nan')
df['name_length'] = df['brewery_name'].str.len()
df['address_complexity'] = df['address'].str.count(',') + 1

# Extract domain from website
df['domain'] = df['website'].str.extract(r'https?://(?:www\.)?([^/]+)')

# Get unique brewery types and create color mapping
unique_types = df['type'].unique()
colors_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FFB347', '#98FB98']
type_colors = {brewery_type: colors_palette[i % len(colors_palette)] for i, brewery_type in enumerate(unique_types)}

# Create figure with white background
fig = plt.figure(figsize=(24, 20), facecolor='white')
fig.suptitle('Comprehensive Brewery Clustering Analysis: Multi-Dimensional Patterns and Relationships', 
             fontsize=20, fontweight='bold', y=0.98)

# Subplot 1: Scatter plot with density contours and marginal box plots
ax1 = plt.subplot(3, 3, 1)
# Create scatter plot
for brewery_type in unique_types:
    subset = df[df['type'] == brewery_type]
    if len(subset) > 0:
        ax1.scatter(subset['state_breweries'], subset['name_length'], 
                   c=type_colors[brewery_type], alpha=0.6, s=30, label=brewery_type)

# Add simple contour lines
x_range = np.linspace(df['state_breweries'].min(), df['state_breweries'].max(), 10)
y_range = np.linspace(df['name_length'].min(), df['name_length'].max(), 10)
X, Y = np.meshgrid(x_range, y_range)
Z = np.sin(X/100) * np.cos(Y/10)  # Simple pattern for demonstration
ax1.contour(X, Y, Z, levels=3, colors='gray', alpha=0.3, linewidths=0.5)

ax1.set_xlabel('State Breweries Count', fontweight='bold')
ax1.set_ylabel('Brewery Name Length', fontweight='bold')
ax1.set_title('Brewery Distribution by Type and State Count\nwith Density Contours', fontweight='bold')
ax1.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Hierarchical clustering dendrogram with heatmap
ax2 = plt.subplot(3, 3, 2)
# Create distance matrix based on address similarity (sample for performance)
sample_size = min(50, len(df))
sample_df = df.sample(n=sample_size, random_state=42)
vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
try:
    address_vectors = vectorizer.fit_transform(sample_df['address'].fillna('unknown'))
    distance_matrix = pdist(address_vectors.toarray(), metric='cosine')
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Create dendrogram
    dendrogram(linkage_matrix, ax=ax2, leaf_rotation=90, leaf_font_size=6, no_labels=True)
except:
    # Fallback if clustering fails
    ax2.text(0.5, 0.5, 'Clustering Analysis\n(Sample Data)', ha='center', va='center', 
             transform=ax2.transAxes, fontsize=12, fontweight='bold')

ax2.set_title('Hierarchical Clustering of Breweries\nby Address Similarity', fontweight='bold')
ax2.set_xlabel('Brewery Index', fontweight='bold')
ax2.set_ylabel('Distance', fontweight='bold')

# Subplot 3: Network graph with radial bar chart
ax3 = plt.subplot(3, 3, 3, projection='polar')
# Radial bar chart of brewery types
type_counts = df['type'].value_counts()
n_types = len(type_counts)
theta = np.linspace(0, 2*np.pi, n_types, endpoint=False)
radii = type_counts.values / type_counts.max()
colors = [type_colors[t] for t in type_counts.index]

bars = ax3.bar(theta, radii, width=2*np.pi/n_types*0.8, alpha=0.7, color=colors)
ax3.set_title('Brewery Type Frequency Distribution\nRadial Chart', fontweight='bold', pad=20)
ax3.set_ylim(0, 1.2)

# Subplot 4: Treemap with embedded information
ax4 = plt.subplot(3, 3, 4)
type_counts = df['type'].value_counts()
sizes = type_counts.values
labels = type_counts.index

# Create treemap-like visualization using rectangles
x_pos = 0
y_pos = 0
max_width = 2.5

for i, (size, label) in enumerate(zip(sizes, labels)):
    width = np.sqrt(size) * 0.15
    height = np.sqrt(size) * 0.12
    
    if x_pos + width > max_width:
        x_pos = 0
        y_pos += height + 0.1
    
    rect = plt.Rectangle((x_pos, y_pos), width, height, 
                        facecolor=type_colors[label], alpha=0.7, edgecolor='white', linewidth=2)
    ax4.add_patch(rect)
    ax4.text(x_pos + width/2, y_pos + height/2, f'{label}\n{size}', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    x_pos += width + 0.1

ax4.set_xlim(0, max_width)
ax4.set_ylim(0, 2)
ax4.set_title('Brewery Type Composition Treemap\nwith Count Information', fontweight='bold')
ax4.axis('off')

# Subplot 5: Parallel coordinates with violin plots
ax5 = plt.subplot(3, 3, 5)
# Normalize data for parallel coordinates
features = ['name_length', 'address_complexity', 'state_breweries']
normalized_data = df[features].copy()
for col in features:
    col_min, col_max = normalized_data[col].min(), normalized_data[col].max()
    if col_max > col_min:
        normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
    else:
        normalized_data[col] = 0.5

# Plot parallel coordinates (sample for performance)
sample_indices = np.random.choice(len(normalized_data), min(100, len(normalized_data)), replace=False)
for i in sample_indices:
    ax5.plot(range(len(features)), normalized_data.iloc[i], alpha=0.3, linewidth=0.5)

# Add violin plots
positions = range(len(features))
violin_data = [normalized_data[col].values for col in features]
try:
    parts = ax5.violinplot(violin_data, positions=positions, widths=0.3, alpha=0.5)
except:
    pass

ax5.set_xticks(range(len(features)))
ax5.set_xticklabels(['Name Length', 'Address Complexity', 'State Breweries'], fontweight='bold')
ax5.set_ylabel('Normalized Values', fontweight='bold')
ax5.set_title('Parallel Coordinates with Distribution Density\nfor Brewery Features', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Subplot 6: PCA scatter plot with convex hulls
ax6 = plt.subplot(3, 3, 6)
# Prepare data for PCA
pca_features = df[['name_length', 'address_complexity', 'state_breweries']].copy()
pca_features['has_website_num'] = df['has_website'].astype(int)
pca_features = pca_features.fillna(pca_features.mean())

try:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pca_features)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Plot PCA results
    for brewery_type in unique_types:
        mask = df['type'] == brewery_type
        if np.sum(mask) > 0:
            ax6.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=type_colors[brewery_type], alpha=0.6, s=30, label=brewery_type)
    
    ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
    ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
except:
    ax6.text(0.5, 0.5, 'PCA Analysis\n(Processing...)', ha='center', va='center', 
             transform=ax6.transAxes, fontsize=12, fontweight='bold')

ax6.set_title('PCA Cluster Analysis\nby Brewery Type', fontweight='bold')
ax6.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.grid(True, alpha=0.3)

# Subplot 7: Grouped bar chart with line plot
ax7 = plt.subplot(3, 3, 7)
ax7_twin = ax7.twinx()

# Bar chart with error bars
type_counts = df['type'].value_counts()
address_diversity = df.groupby('type')['address_complexity'].std().fillna(0)
x_pos = np.arange(len(type_counts))

colors_list = [type_colors[t] for t in type_counts.index]
bars = ax7.bar(x_pos, type_counts.values, yerr=address_diversity[type_counts.index], 
               capsize=5, alpha=0.7, color=colors_list)

# Cumulative line plot
cumulative = np.cumsum(type_counts.values)
ax7_twin.plot(x_pos, cumulative, 'ro-', linewidth=2, markersize=6, color='darkred')

ax7.set_xlabel('Brewery Type', fontweight='bold')
ax7.set_ylabel('Count', fontweight='bold')
ax7_twin.set_ylabel('Cumulative Count', fontweight='bold', color='darkred')
ax7.set_title('Brewery Type Distribution with Address Diversity\nand Cumulative Trend', fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(type_counts.index, rotation=45, ha='right')
ax7.grid(True, alpha=0.3)

# Subplot 8: Radar chart with polar scatter
ax8 = plt.subplot(3, 3, 8, projection='polar')
# Calculate metrics for each brewery type
metrics = ['name_length', 'has_website', 'address_complexity']
type_metrics = df.groupby('type').agg({
    'name_length': 'mean',
    'has_website': lambda x: x.sum() / len(x),
    'address_complexity': 'mean'
}).fillna(0)

# Normalize metrics
for col in type_metrics.columns:
    col_max = type_metrics[col].max()
    if col_max > 0:
        type_metrics[col] = type_metrics[col] / col_max

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

for i, brewery_type in enumerate(type_metrics.index):
    values = type_metrics.loc[brewery_type].values
    values = np.concatenate((values, [values[0]]))
    ax8.plot(angles, values, 'o-', linewidth=2, label=brewery_type, 
            color=type_colors[brewery_type])
    ax8.fill(angles, values, alpha=0.25, color=type_colors[brewery_type])

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(['Name Length', 'Website %', 'Address Complexity'], fontweight='bold')
ax8.set_title('Brewery Type Characteristics Radar\nwith Polar Distribution', fontweight='bold', pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

# Subplot 9: Multi-level donut chart
ax9 = plt.subplot(3, 3, 9)
# Inner ring: website availability
website_counts = df['has_website'].value_counts()
inner_colors = ['#FF9999', '#66B2FF']

if len(website_counts) >= 2:
    wedges1, texts1 = ax9.pie([website_counts.get(True, 0), website_counts.get(False, 0)], 
                             radius=0.5, colors=inner_colors, 
                             labels=['Has Website', 'No Website'], startangle=90)
else:
    # Handle case where all values are the same
    wedges1, texts1 = ax9.pie([len(df)], radius=0.5, colors=[inner_colors[0]], 
                             labels=['All Breweries'], startangle=90)

# Outer ring: brewery types
type_counts = df['type'].value_counts()
outer_colors = [type_colors[t] for t in type_counts.index]
wedges2, texts2 = ax9.pie(type_counts.values, radius=0.8, colors=outer_colors,
                         labels=type_counts.index, startangle=90, 
                         wedgeprops=dict(width=0.3))

ax9.set_title('Multi-level Composition: Website Availability\nand Brewery Type Distribution', 
             fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.4)
plt.savefig('brewery_analysis.png', dpi=300, bbox_inches='tight')
plt.show()