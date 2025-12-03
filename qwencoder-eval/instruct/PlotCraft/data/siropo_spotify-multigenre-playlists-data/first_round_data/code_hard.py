import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# Load and combine all datasets
datasets = []
genre_labels = []

# File mappings
file_mappings = {
    'metal_music_data.csv': 'Metal',
    'indie_alt_music_data.csv': 'Indie/Alt', 
    'alternative_music_data.csv': 'Alternative',
    'blues_music_data.csv': 'Blues',
    'pop_music_data.csv': 'Pop',
    'hiphop_music_data.csv': 'Hip-Hop',
    'rock_music_data.csv': 'Rock'
}

combined_data = []
colors_list = ['#8B0000', '#FF6B35', '#F7931E', '#4A90E2', '#FF69B4', '#32CD32', '#9370DB']

# Load datasets with proper error handling
for filename, genre_name in file_mappings.items():
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['Genre_Label'] = genre_name
            combined_data.append(df)
            genre_labels.append(genre_name)
            print(f"Loaded {filename} with {len(df)} rows")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# If no files found, create sample data
if not combined_data:
    print("No data files found, creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    
    sample_genres = ['Metal', 'Pop', 'Rock', 'Blues', 'Hip-Hop']
    genre_labels = sample_genres
    
    for i, genre in enumerate(sample_genres):
        # Create genre-specific characteristics
        if genre == 'Metal':
            energy_base, valence_base = 0.8, 0.3
        elif genre == 'Pop':
            energy_base, valence_base = 0.7, 0.7
        elif genre == 'Rock':
            energy_base, valence_base = 0.75, 0.6
        elif genre == 'Blues':
            energy_base, valence_base = 0.5, 0.4
        else:  # Hip-Hop
            energy_base, valence_base = 0.6, 0.5
            
        df_sample = pd.DataFrame({
            'danceability': np.clip(np.random.normal(0.5 + (0.2 if genre in ['Pop', 'Hip-Hop'] else 0), 0.15, n_samples//len(sample_genres)), 0, 1),
            'energy': np.clip(np.random.normal(energy_base, 0.15, n_samples//len(sample_genres)), 0, 1),
            'acousticness': np.clip(np.random.normal(0.3 - (0.2 if genre == 'Metal' else 0), 0.2, n_samples//len(sample_genres)), 0, 1),
            'valence': np.clip(np.random.normal(valence_base, 0.2, n_samples//len(sample_genres)), 0, 1),
            'speechiness': np.clip(np.random.normal(0.1 + (0.3 if genre == 'Hip-Hop' else 0), 0.1, n_samples//len(sample_genres)), 0, 1),
            'loudness': np.random.normal(-8, 3, n_samples//len(sample_genres)),
            'tempo': np.clip(np.random.normal(120, 30, n_samples//len(sample_genres)), 60, 200),
            'Popularity': np.random.randint(0, 100, n_samples//len(sample_genres)),
            'duration_ms': np.clip(np.random.normal(200000, 50000, n_samples//len(sample_genres)), 60000, 600000),
            'Genre_Label': genre
        })
        
        combined_data.append(df_sample)

# Combine all data
df_all = pd.concat(combined_data, ignore_index=True)

# Select key audio features for analysis and ensure they are numeric
audio_features = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness', 'loudness', 'tempo']
required_cols = audio_features + ['Popularity', 'duration_ms', 'Genre_Label']

# Ensure all required columns exist
for col in required_cols:
    if col not in df_all.columns:
        if col == 'Genre_Label':
            continue
        else:
            df_all[col] = np.random.random(len(df_all)) if col in audio_features[:5] else np.random.normal(0, 1, len(df_all))

# Convert numeric columns to proper types and handle non-numeric values
for col in audio_features + ['Popularity', 'duration_ms']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# Clean the data
df_clean = df_all[required_cols].dropna()

# Ensure we have valid data ranges
for col in ['danceability', 'energy', 'acousticness', 'valence', 'speechiness']:
    df_clean[col] = df_clean[col].clip(0, 1)

df_clean['Popularity'] = df_clean['Popularity'].clip(0, 100)
df_clean['duration_ms'] = df_clean['duration_ms'].clip(30000, 600000)
df_clean['tempo'] = df_clean['tempo'].clip(60, 200)

# Create color palette for genres
colors = colors_list[:len(genre_labels)]
genre_colors = dict(zip(genre_labels, colors))

print(f"Final dataset shape: {df_clean.shape}")
print(f"Genres: {genre_labels}")

# Set up the figure
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')

# Row 1, Subplot 1: Scatter plot with KDE contours - Energy vs Valence
ax1 = plt.subplot(3, 3, 1)
for i, genre in enumerate(genre_labels):
    genre_data = df_clean[df_clean['Genre_Label'] == genre]
    if len(genre_data) > 0:
        ax1.scatter(genre_data['energy'], genre_data['valence'], 
                   c=colors[i], alpha=0.6, s=20, label=genre)
        
        # Add KDE contours for each genre
        if len(genre_data) > 10:
            try:
                sns.kdeplot(data=genre_data, x='energy', y='valence', 
                           color=colors[i], alpha=0.3, levels=2, ax=ax1)
            except:
                pass

ax1.set_xlabel('Energy', fontweight='bold')
ax1.set_ylabel('Valence', fontweight='bold')
ax1.set_title('Energy vs Valence Relationships Across Genres', fontweight='bold', fontsize=11)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Radar chart
ax2 = plt.subplot(3, 3, 2, projection='polar')
features_radar = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness']
angles = np.linspace(0, 2 * np.pi, len(features_radar), endpoint=False).tolist()
angles += angles[:1]

for i, genre in enumerate(genre_labels):
    genre_data = df_clean[df_clean['Genre_Label'] == genre]
    if len(genre_data) > 0:
        values = [genre_data[feature].mean() for feature in features_radar]
        values += values[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, label=genre, color=colors[i])
        ax2.fill(angles, values, alpha=0.1, color=colors[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(features_radar, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.set_title('Audio Feature Profiles by Genre', fontweight='bold', fontsize=11, pad=20)
ax2.legend(bbox_to_anchor=(1.3, 1.1), fontsize=8)

# Row 1, Subplot 3: Hierarchical clustering
ax3 = plt.subplot(3, 3, 3)
corr_data = []
valid_genres = []
for genre in genre_labels:
    genre_data = df_clean[df_clean['Genre_Label'] == genre]
    if len(genre_data) > 0:
        genre_means = [genre_data[feature].mean() for feature in features_radar]
        corr_data.append(genre_means)
        valid_genres.append(genre)

if len(corr_data) > 1:
    try:
        linkage_matrix = linkage(corr_data, method='ward')
        dendro = dendrogram(linkage_matrix, labels=valid_genres, ax=ax3)
        ax3.set_title('Genre Clustering by Audio Features', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Distance', fontweight='bold')
    except:
        ax3.text(0.5, 0.5, 'Clustering visualization unavailable', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Genre Clustering by Audio Features', fontweight='bold', fontsize=11)
else:
    ax3.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Genre Clustering by Audio Features', fontweight='bold', fontsize=11)

# Row 2, Subplot 4: Bubble chart - Tempo vs Popularity
ax4 = plt.subplot(3, 3, 4)
for i, genre in enumerate(genre_labels):
    genre_data = df_clean[df_clean['Genre_Label'] == genre]
    if len(genre_data) > 0:
        sizes = (genre_data['duration_ms'] / 5000).clip(10, 100)
        ax4.scatter(genre_data['tempo'], genre_data['Popularity'], 
                   s=sizes, c=colors[i], alpha=0.6, label=genre)

ax4.set_xlabel('Tempo (BPM)', fontweight='bold')
ax4.set_ylabel('Popularity', fontweight='bold')
ax4.set_title('Tempo vs Popularity (Bubble Size = Duration)', fontweight='bold', fontsize=11)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Violin plots - Popularity by Genre
ax5 = plt.subplot(3, 3, 5)
popularity_data = []
valid_genre_labels = []

for genre in genre_labels:
    genre_pop = df_clean[df_clean['Genre_Label'] == genre]['Popularity'].values
    if len(genre_pop) > 0:
        popularity_data.append(genre_pop)
        valid_genre_labels.append(genre)

if popularity_data:
    try:
        parts = ax5.violinplot(popularity_data, positions=range(len(valid_genre_labels)), 
                              showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax5.set_xticks(range(len(valid_genre_labels)))
        ax5.set_xticklabels(valid_genre_labels, rotation=45, fontweight='bold')
        ax5.set_ylabel('Popularity Score', fontweight='bold')
        ax5.set_title('Popularity Distribution by Genre', fontweight='bold', fontsize=11)
        ax5.grid(True, alpha=0.3)
    except:
        # Fallback to box plot
        box_data = [df_clean[df_clean['Genre_Label'] == genre]['Popularity'] for genre in valid_genre_labels]
        ax5.boxplot(box_data, labels=valid_genre_labels)
        ax5.set_ylabel('Popularity Score', fontweight='bold')
        ax5.set_title('Popularity Distribution by Genre', fontweight='bold', fontsize=11)
        ax5.tick_params(axis='x', rotation=45)

# Row 2, Subplot 6: Genre similarity matrix
ax6 = plt.subplot(3, 3, 6)
if len(corr_data) > 1:
    try:
        similarity_matrix = np.corrcoef(corr_data)
        im = ax6.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
        ax6.set_xticks(range(len(valid_genres)))
        ax6.set_yticks(range(len(valid_genres)))
        ax6.set_xticklabels(valid_genres, rotation=45, fontweight='bold')
        ax6.set_yticklabels(valid_genres, fontweight='bold')
        ax6.set_title('Genre Similarity Matrix', fontweight='bold', fontsize=11)

        for i in range(len(valid_genres)):
            for j in range(len(valid_genres)):
                text = ax6.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

        plt.colorbar(im, ax=ax6, shrink=0.8)
    except:
        ax6.text(0.5, 0.5, 'Similarity matrix unavailable', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Genre Similarity Matrix', fontweight='bold', fontsize=11)
else:
    ax6.text(0.5, 0.5, 'Insufficient data for similarity matrix', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Genre Similarity Matrix', fontweight='bold', fontsize=11)

# Row 3, Subplot 7: Parallel coordinates plot
ax7 = plt.subplot(3, 3, 7)
features_norm = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness']

for i, genre in enumerate(genre_labels):
    genre_data = df_clean[df_clean['Genre_Label'] == genre]
    if len(genre_data) > 0:
        sample_size = min(50, len(genre_data))
        if sample_size > 0:
            sample_data = genre_data.sample(sample_size)[features_norm]
            
            for idx, row in sample_data.iterrows():
                ax7.plot(range(len(features_norm)), row.values, 
                        color=colors[i], alpha=0.3, linewidth=0.5)

# Add mean lines for each genre
for i, genre in enumerate(genre_labels):
    genre_data = df_clean[df_clean['Genre_Label'] == genre]
    if len(genre_data) > 0:
        means = [genre_data[feature].mean() for feature in features_norm]
        ax7.plot(range(len(features_norm)), means, 
                color=colors[i], linewidth=3, label=genre)

ax7.set_xticks(range(len(features_norm)))
ax7.set_xticklabels(features_norm, rotation=45, fontweight='bold')
ax7.set_ylabel('Feature Value', fontweight='bold')
ax7.set_title('Parallel Coordinates: Audio Features', fontweight='bold', fontsize=11)
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax7.grid(True, alpha=0.3)

# Row 3, Subplot 8: 2D histogram - Loudness vs Acousticness
ax8 = plt.subplot(3, 3, 8)
try:
    hist, xedges, yedges = np.histogram2d(df_clean['loudness'], df_clean['acousticness'], bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax8.imshow(hist.T, origin='lower', extent=extent, cmap='Blues', alpha=0.7)

    for i, genre in enumerate(genre_labels):
        genre_data = df_clean[df_clean['Genre_Label'] == genre]
        if len(genre_data) > 0:
            sample_size = min(100, len(genre_data))
            if sample_size > 0:
                sample_data = genre_data.sample(sample_size)
                ax8.scatter(sample_data['loudness'], sample_data['acousticness'], 
                           c=colors[i], alpha=0.6, s=15, label=genre)

    ax8.set_xlabel('Loudness (dB)', fontweight='bold')
    ax8.set_ylabel('Acousticness', fontweight='bold')
    ax8.set_title('Loudness vs Acousticness Distribution', fontweight='bold', fontsize=11)
    ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.colorbar(im, ax=ax8, shrink=0.8)
except:
    # Fallback to simple scatter plot
    for i, genre in enumerate(genre_labels):
        genre_data = df_clean[df_clean['Genre_Label'] == genre]
        if len(genre_data) > 0:
            ax8.scatter(genre_data['loudness'], genre_data['acousticness'], 
                       c=colors[i], alpha=0.6, s=15, label=genre)
    ax8.set_xlabel('Loudness (dB)', fontweight='bold')
    ax8.set_ylabel('Acousticness', fontweight='bold')
    ax8.set_title('Loudness vs Acousticness Distribution', fontweight='bold', fontsize=11)
    ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Row 3, Subplot 9: K-means clustering
ax9 = plt.subplot(3, 3, 9)

try:
    cluster_features = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness']
    X = df_clean[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(len(genre_labels), 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled, cluster_labels)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax9.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='tab10', alpha=0.6, s=20)

    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax9.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')

    ax9.set_xlabel('First Principal Component', fontweight='bold')
    ax9.set_ylabel('Second Principal Component', fontweight='bold')
    ax9.set_title(f'K-Means Clustering (Silhouette: {silhouette_avg:.3f})', 
                 fontweight='bold', fontsize=11)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
except Exception as e:
    ax9.text(0.5, 0.5, f'Clustering unavailable', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('K-Means Clustering Analysis', fontweight='bold', fontsize=11)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.4, wspace=0.5)

# Save the plot
plt.savefig('spotify_multigenre_clustering_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()