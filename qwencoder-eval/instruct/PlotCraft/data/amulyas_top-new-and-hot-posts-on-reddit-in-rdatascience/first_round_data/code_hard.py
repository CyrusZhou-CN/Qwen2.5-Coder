import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('reddit_datascience_newTopHot_posts.csv')

# Data preprocessing
df['Text Length'] = df['Text'].fillna('').str.len()
df['Title Length'] = df['Title'].str.len()
df = df.dropna(subset=['Score', 'Comment Count'])

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Reddit r/datascience Posts: Clustering and Engagement Pattern Analysis', 
             fontsize=20, fontweight='bold', y=0.98)

# Row 1: Post Type Analysis

# Subplot 1: Stacked bar chart with overlaid line plot
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Score distribution by post type
score_bins = pd.cut(df['Score'], bins=[0, 10, 50, 100, float('inf')], 
                   labels=['Low (0-10)', 'Medium (10-50)', 'High (50-100)', 'Very High (100+)'])
score_type_counts = pd.crosstab(df['Type'], score_bins)

# Stacked bar chart
score_type_counts.plot(kind='bar', stacked=True, ax=ax1, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])

# Overlaid line plot for average comments
ax1_twin = ax1.twinx()
avg_comments = df.groupby('Type')['Comment Count'].mean()
ax1_twin.plot(range(len(avg_comments)), avg_comments.values, 
              color='#2C3E50', marker='o', linewidth=3, markersize=8, label='Avg Comments')

ax1.set_title('Score Distribution by Post Type with Average Comments', fontweight='bold', pad=20)
ax1.set_xlabel('Post Type', fontweight='bold')
ax1.set_ylabel('Number of Posts', fontweight='bold')
ax1_twin.set_ylabel('Average Comment Count', fontweight='bold')
ax1.legend(title='Score Range', loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: Violin plot with box plots and outliers
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Violin plot
parts = ax2.violinplot([df[df['Type'] == t]['Score'].values for t in df['Type'].unique()], 
                       positions=range(len(df['Type'].unique())), showmeans=True)

# Color violin plots
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_alpha(0.7)

# Overlaid box plots
box_data = [df[df['Type'] == t]['Score'].values for t in df['Type'].unique()]
bp = ax2.boxplot(box_data, positions=range(len(df['Type'].unique())), 
                 widths=0.3, patch_artist=True, showfliers=False)

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors[i % len(colors)])
    patch.set_alpha(0.5)

# Add outlier scatter points
for i, t in enumerate(df['Type'].unique()):
    type_data = df[df['Type'] == t]['Score']
    Q1, Q3 = type_data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = type_data[(type_data < Q1 - 1.5*IQR) | (type_data > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        ax2.scatter([i] * len(outliers), outliers, alpha=0.6, s=20, color='red')

ax2.set_title('Score Distribution by Post Type (Violin + Box + Outliers)', fontweight='bold', pad=20)
ax2.set_xlabel('Post Type', fontweight='bold')
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_xticks(range(len(df['Type'].unique())))
ax2.set_xticklabels(df['Type'].unique())
ax2.grid(True, alpha=0.3)

# Subplot 3: Bubble chart
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

type_colors = {'Top - day': '#FF6B6B', 'New': '#4ECDC4', 'Hot': '#45B7D1'}
for post_type in df['Type'].unique():
    subset = df[df['Type'] == post_type]
    ax3.scatter(subset['Comment Count'], subset['Score'], 
               s=subset['Text Length']/50, alpha=0.6, 
               color=type_colors.get(post_type, '#96CEB4'), label=post_type)

ax3.set_title('Engagement Bubble Chart (Size = Text Length)', fontweight='bold', pad=20)
ax3.set_xlabel('Comment Count', fontweight='bold')
ax3.set_ylabel('Score', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Row 2: Content Category Clustering

# Subplot 4: Radar chart
ax4 = plt.subplot(3, 3, 4, projection='polar')
ax4.set_facecolor('white')

# Get top 5 flairs
top_flairs = df['Flair'].value_counts().head(5).index.tolist()
flair_metrics = df[df['Flair'].isin(top_flairs)].groupby('Flair').agg({
    'Score': 'mean',
    'Comment Count': 'mean',
    'Text Length': 'mean'
}).reset_index()

# Normalize metrics
scaler = StandardScaler()
metrics_normalized = scaler.fit_transform(flair_metrics[['Score', 'Comment Count', 'Text Length']])

# Radar chart setup
categories = ['Score', 'Comments', 'Text Length']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
for i, flair in enumerate(flair_metrics['Flair']):
    values = metrics_normalized[i].tolist()
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2, label=flair, color=colors[i])
    ax4.fill(angles, values, alpha=0.25, color=colors[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_title('Content Category Metrics Comparison', fontweight='bold', pad=30)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 5: Hierarchical clustering dendrogram
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Prepare data for clustering
cluster_data = df[['Score', 'Comment Count']].fillna(0)
cluster_sample = cluster_data.sample(n=min(100, len(cluster_data)), random_state=42)
sample_indices = cluster_sample.index

# Perform hierarchical clustering
linkage_matrix = linkage(cluster_sample, method='ward')

# Create dendrogram with flair color coding
flair_colors = {flair: colors[i % len(colors)] for i, flair in enumerate(df['Flair'].unique())}
leaf_colors = [flair_colors.get(df.loc[idx, 'Flair'], '#000000') for idx in sample_indices]

dendrogram(linkage_matrix, ax=ax5, leaf_rotation=90, color_threshold=0.7*max(linkage_matrix[:,2]))
ax5.set_title('Hierarchical Clustering of Posts by Engagement', fontweight='bold', pad=20)
ax5.set_xlabel('Post Index', fontweight='bold')
ax5.set_ylabel('Distance', fontweight='bold')

# Subplot 6: Parallel coordinates plot
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Prepare data for parallel coordinates
parallel_data = df[['Score', 'Comment Count', 'Text Length', 'Title Length', 'Flair']].dropna()
parallel_sample = parallel_data.sample(n=min(200, len(parallel_data)), random_state=42)

# Normalize data
features = ['Score', 'Comment Count', 'Text Length', 'Title Length']
normalized_data = parallel_sample[features].copy()
for feature in features:
    normalized_data[feature] = (normalized_data[feature] - normalized_data[feature].min()) / \
                              (normalized_data[feature].max() - normalized_data[feature].min())

# Plot parallel coordinates
for i, (_, row) in enumerate(parallel_sample.iterrows()):
    flair = row['Flair']
    color = flair_colors.get(flair, '#CCCCCC')
    values = [normalized_data.loc[row.name, feature] for feature in features]
    ax6.plot(range(len(features)), values, alpha=0.6, color=color, linewidth=1)

ax6.set_xticks(range(len(features)))
ax6.set_xticklabels(features, rotation=45)
ax6.set_title('Parallel Coordinates: Multi-dimensional Relationships', fontweight='bold', pad=20)
ax6.set_ylabel('Normalized Values', fontweight='bold')
ax6.grid(True, alpha=0.3)

# Row 3: Engagement Pattern Analysis

# Subplot 7: 2D histogram heatmap with contours (without marginal plots to avoid axis sharing issues)
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Create 2D histogram
hist, xedges, yedges = np.histogram2d(df['Comment Count'], df['Score'], bins=30)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot heatmap
im = ax7.imshow(hist.T, extent=extent, origin='lower', cmap='YlOrRd', alpha=0.8)

# Add contour lines
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
ax7.contour(X, Y, hist.T, levels=5, colors='black', alpha=0.6, linewidths=1)

ax7.set_title('Score vs Comment Count Correlation Heatmap', fontweight='bold', pad=20)
ax7.set_xlabel('Comment Count', fontweight='bold')
ax7.set_ylabel('Score', fontweight='bold')
plt.colorbar(im, ax=ax7, label='Frequency')

# Subplot 8: Network graph
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Create author network based on similar posting patterns
author_stats = df.groupby('Author').agg({
    'Score': ['sum', 'mean'],
    'Comment Count': 'mean',
    'Type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
}).reset_index()

author_stats.columns = ['Author', 'Total_Score', 'Avg_Score', 'Avg_Comments', 'Primary_Type']
top_authors = author_stats.nlargest(15, 'Total_Score')

# Create network
G = nx.Graph()
for _, author in top_authors.iterrows():
    G.add_node(author['Author'], 
               total_score=author['Total_Score'],
               avg_score=author['Avg_Score'])

# Add edges based on similar posting patterns
for i, author1 in top_authors.iterrows():
    for j, author2 in top_authors.iterrows():
        if i < j:
            similarity = 1 / (1 + abs(author1['Avg_Score'] - author2['Avg_Score']))
            if similarity > 0.7:
                G.add_edge(author1['Author'], author2['Author'], weight=similarity)

# Draw network
pos = nx.spring_layout(G, k=1, iterations=50)
node_sizes = [G.nodes[node]['total_score'] * 2 for node in G.nodes()]
nx.draw(G, pos, ax=ax8, node_size=node_sizes, node_color='#45B7D1', 
        alpha=0.7, with_labels=False, edge_color='gray', width=0.5)

ax8.set_title('Author Network by Posting Patterns', fontweight='bold', pad=20)
ax8.axis('off')

# Subplot 9: Multi-level grouped bar chart
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Create score ranges
df['Score_Range'] = pd.cut(df['Score'], bins=[0, 10, 50, float('inf')], 
                          labels=['Low', 'Medium', 'High'])

# Group by flair and score range
flair_score_counts = pd.crosstab(df['Flair'], df['Score_Range'])
top_flairs_for_chart = df['Flair'].value_counts().head(5).index

flair_score_subset = flair_score_counts.loc[top_flairs_for_chart]

# Grouped bar chart
x = np.arange(len(top_flairs_for_chart))
width = 0.25

bars1 = ax9.bar(x - width, flair_score_subset['Low'], width, label='Low', color='#FF6B6B', alpha=0.8)
bars2 = ax9.bar(x, flair_score_subset['Medium'], width, label='Medium', color='#4ECDC4', alpha=0.8)
bars3 = ax9.bar(x + width, flair_score_subset['High'], width, label='High', color='#45B7D1', alpha=0.8)

# Overlaid cumulative percentage line
ax9_twin = ax9.twinx()
cumulative_pct = flair_score_subset.div(flair_score_subset.sum(axis=1), axis=0).cumsum(axis=1)['High']
ax9_twin.plot(x, cumulative_pct * 100, color='#2C3E50', marker='o', linewidth=3, 
              markersize=8, label='Cumulative %')

ax9.set_title('Score Ranges by Flair Category with Cumulative Percentage', fontweight='bold', pad=20)
ax9.set_xlabel('Flair Category', fontweight='bold')
ax9.set_ylabel('Number of Posts', fontweight='bold')
ax9_twin.set_ylabel('Cumulative Percentage', fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(top_flairs_for_chart, rotation=45, ha='right')
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')
ax9.grid(True, alpha=0.3)

# Final layout adjustment - use subplots_adjust instead of tight_layout to avoid axis sharing issues
plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.4, left=0.05, right=0.95, bottom=0.1)
plt.savefig('reddit_datascience_analysis.png', dpi=300, bbox_inches='tight')
plt.show()