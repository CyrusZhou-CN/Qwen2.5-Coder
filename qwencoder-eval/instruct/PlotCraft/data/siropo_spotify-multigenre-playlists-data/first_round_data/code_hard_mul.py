import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
import networkx as nx
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
datasets = {
    'Metal': pd.read_csv('metal_music_data.csv'),
    'Indie/Alt': pd.read_csv('indie_alt_music_data.csv'),
    'Alternative': pd.read_csv('alternative_music_data.csv'),
    'Blues': pd.read_csv('blues_music_data.csv'),
    'Pop': pd.read_csv('pop_music_data.csv'),
    'Hip-Hop': pd.read_csv('hiphop_music_data.csv'),
    'Rock': pd.read_csv('rock_music_data.csv')
}

# Combine datasets with genre labels
combined_data = []
for genre, df in datasets.items():
    df_copy = df.copy()
    df_copy['Genre'] = genre
    combined_data.append(df_copy)

df_all = pd.concat(combined_data, ignore_index=True)

# Define audio features for analysis
audio_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']

# Normalize tempo to 0-1 scale
scaler = MinMaxScaler()
df_all['tempo_normalized'] = scaler.fit_transform(df_all[['tempo']])

# Create genre averages for analysis
genre_stats = df_all.groupby('Genre').agg({
    'danceability': 'mean',
    'energy': 'mean',
    'acousticness': 'mean',
    'valence': 'mean',
    'tempo_normalized': 'mean',
    'Popularity': 'mean'
}).reset_index()

# Define consistent color palette for genres
colors = {
    'Metal': '#8B0000',
    'Indie/Alt': '#4169E1', 
    'Alternative': '#32CD32',
    'Blues': '#1E90FF',
    'Pop': '#FF69B4',
    'Hip-Hop': '#FFD700',
    'Rock': '#FF4500'
}

# Create the comprehensive 3x2 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Subplot 1: Scatter plot with marginal histograms and density contours
ax1 = plt.subplot(2, 3, 1)
ax1.set_facecolor('white')

# Sample data for performance (use every 20th row)
sample_df = df_all.iloc[::20].copy()

# Create scatter plot with points colored by genre and sized by popularity
for genre in sample_df['Genre'].unique():
    genre_data = sample_df[sample_df['Genre'] == genre]
    sizes = (genre_data['Popularity'] + 1) * 2  # Add 1 to avoid zero sizes
    ax1.scatter(genre_data['energy'], genre_data['valence'], 
               c=colors[genre], s=sizes, alpha=0.6, label=genre, edgecolors='white', linewidth=0.5)

# Add density contours - Fixed the reshape issue
x = sample_df['energy'].values
y = sample_df['valence'].values
if len(x) > 10:  # Only create contours if we have enough data points
    try:
        # Create a grid for contour plotting
        xi = np.linspace(x.min(), x.max(), 30)
        yi = np.linspace(y.min(), y.max(), 30)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Calculate KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        zi = kde(np.vstack([xi_grid.ravel(), yi_grid.ravel()]))
        zi = zi.reshape(xi_grid.shape)
        
        ax1.contour(xi_grid, yi_grid, zi, colors='gray', alpha=0.3, linewidths=1)
    except:
        pass  # Skip contours if there's an issue

ax1.set_xlabel('Energy', fontweight='bold', fontsize=12)
ax1.set_ylabel('Valence', fontweight='bold', fontsize=12)
ax1.set_title('Energy vs Valence Clustering\n(Point size = Popularity)', fontweight='bold', fontsize=14)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Parallel coordinates with box plots overlay
ax2 = plt.subplot(2, 3, 2)
ax2.set_facecolor('white')

# Prepare data for parallel coordinates
features_for_parallel = ['danceability', 'energy', 'acousticness', 'valence', 'tempo_normalized']
parallel_data = df_all[features_for_parallel + ['Genre']].copy()

# Create parallel coordinates plot
for i, genre in enumerate(parallel_data['Genre'].unique()):
    genre_data = parallel_data[parallel_data['Genre'] == genre]
    
    # Sample for performance
    if len(genre_data) > 100:
        genre_data = genre_data.sample(100)
    
    for idx, row in genre_data.iterrows():
        values = [row[feat] for feat in features_for_parallel]
        ax2.plot(range(len(features_for_parallel)), values, 
                color=colors[genre], alpha=0.1, linewidth=0.5)

# Add genre mean lines
for genre in df_all['Genre'].unique():
    genre_means = df_all[df_all['Genre'] == genre][features_for_parallel].mean()
    ax2.plot(range(len(features_for_parallel)), genre_means.values, 
            color=colors[genre], alpha=0.8, linewidth=3, label=f'{genre} Mean')

ax2.set_xticks(range(len(features_for_parallel)))
ax2.set_xticklabels([f.title().replace('_', ' ') for f in features_for_parallel], rotation=45)
ax2.set_ylabel('Normalized Values', fontweight='bold', fontsize=12)
ax2.set_title('Parallel Coordinates\nAudio Features Distribution', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Subplot 3: Correlation heatmap with dendrogram
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor('white')

# Calculate correlation matrix for genre averages
corr_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo_normalized']
genre_corr_data = genre_stats[corr_features].T
corr_matrix = genre_corr_data.corr()

# Create heatmap
im = ax3.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

ax3.set_xticks(range(len(genre_stats)))
ax3.set_yticks(range(len(genre_stats)))
ax3.set_xticklabels(genre_stats['Genre'], rotation=45)
ax3.set_yticklabels(genre_stats['Genre'])
ax3.set_title('Genre Correlation Heatmap\nBased on Audio Features', fontweight='bold', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# Subplot 4: Radar chart with scatter overlay
ax4 = plt.subplot(2, 3, 4, projection='polar')
ax4.set_facecolor('white')

# Prepare radar chart data
radar_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo_normalized']
angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot radar chart for each genre
for genre in genre_stats['Genre']:
    values = genre_stats[genre_stats['Genre'] == genre][radar_features].iloc[0].tolist()
    values += values[:1]  # Complete the circle
    
    ax4.plot(angles, values, 'o-', linewidth=2, label=genre, color=colors[genre], alpha=0.8)
    ax4.fill(angles, values, alpha=0.15, color=colors[genre])

# Add individual track points as small dots (sample for performance)
sample_tracks = df_all.sample(min(300, len(df_all)))
for _, track in sample_tracks.iterrows():
    track_values = [track[feat] for feat in radar_features]
    track_values += track_values[:1]
    ax4.scatter(angles[:-1], track_values[:-1], c=colors[track['Genre']], 
               s=3, alpha=0.2)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels([f.title().replace('_', ' ') for f in radar_features], fontweight='bold')
ax4.set_ylim(0, 1)
ax4.set_title('Radar Chart Comparison\nGenre Profiles with Individual Tracks', 
              fontweight='bold', fontsize=14, pad=20)
ax4.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left', fontsize=8)

# Subplot 5: Network graph with bubble chart
ax5 = plt.subplot(2, 3, 5)
ax5.set_facecolor('white')

# Create network graph
G = nx.Graph()

# Add nodes (genres) with popularity as size
for _, row in genre_stats.iterrows():
    G.add_node(row['Genre'], popularity=row['Popularity'])

# Add edges based on feature similarity (correlation)
feature_matrix = genre_stats[radar_features].values
correlation_matrix = np.corrcoef(feature_matrix)

threshold = 0.5  # Lower similarity threshold to ensure some connections
for i, genre1 in enumerate(genre_stats['Genre']):
    for j, genre2 in enumerate(genre_stats['Genre']):
        if i < j and abs(correlation_matrix[i, j]) > threshold:
            G.add_edge(genre1, genre2, weight=abs(correlation_matrix[i, j]))

# Create layout
pos = nx.spring_layout(G, k=2, iterations=50)

# Draw network
for node in G.nodes():
    popularity = G.nodes[node]['popularity']
    size = (popularity + 10) * 15  # Scale for visibility
    ax5.scatter(pos[node][0], pos[node][1], s=size, c=colors[node], 
               alpha=0.8, edgecolors='black', linewidth=2)
    ax5.text(pos[node][0], pos[node][1], node, ha='center', va='center', 
            fontweight='bold', fontsize=9)

# Draw edges
for edge in G.edges():
    x1, y1 = pos[edge[0]]
    x2, y2 = pos[edge[1]]
    weight = G.edges[edge]['weight']
    ax5.plot([x1, x2], [y1, y2], 'k-', alpha=weight*0.7, linewidth=weight*2)

ax5.set_title('Network Graph with Bubble Chart\nGenre Similarity Network', 
              fontweight='bold', fontsize=14)
ax5.axis('off')

# Add legend for bubble sizes
legend_sizes = [100, 300, 500]
legend_labels = ['Low Pop.', 'Med Pop.', 'High Pop.']
legend_elements = []
for size, label in zip(legend_sizes, legend_labels):
    legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.7, label=label))
ax5.legend(handles=legend_elements, loc='upper right', title='Node Size', title_fontsize=10)

# Subplot 6: Feature dominance analysis
ax6 = plt.subplot(2, 3, 6)
ax6.set_facecolor('white')

# Calculate feature variance for each genre to show diversity
feature_variance = df_all.groupby('Genre')[radar_features].var()

# Create stacked bar chart showing feature variance (diversity within genres)
bottom = np.zeros(len(genre_stats))
bar_width = 0.6

for i, feature in enumerate(radar_features):
    values = feature_variance[feature].values
    bars = ax6.bar(range(len(genre_stats)), values, bottom=bottom, 
                   label=feature.title().replace('_', ' '), alpha=0.8, width=bar_width)
    bottom += values

ax6.set_xticks(range(len(genre_stats)))
ax6.set_xticklabels(genre_stats['Genre'], rotation=45)
ax6.set_xlabel('Genre', fontweight='bold', fontsize=12)
ax6.set_ylabel('Feature Variance (Diversity)', fontweight='bold', fontsize=12)
ax6.set_title('Feature Diversity Analysis\nVariance Within Genres', 
              fontweight='bold', fontsize=14)
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

# Overall layout adjustment
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('Comprehensive Musical Genre Clustering and Audio Feature Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('comprehensive_music_analysis.png', dpi=300, bbox_inches='tight')
plt.show()