import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

# Add genre labels and combine datasets
combined_data = []
for genre, df in datasets.items():
    df_copy = df.copy()
    df_copy['Genre'] = genre
    combined_data.append(df_copy)

df_combined = pd.concat(combined_data, ignore_index=True)

# Define audio features for analysis
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create figure with white background
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('white')

# Define color palette for genres
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
genre_colors = dict(zip(datasets.keys(), colors))

# Top-left: Correlation heatmap
ax1 = axes[0, 0]
correlation_matrix = df_combined[audio_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax1)
ax1.set_title('Audio Features Correlation Matrix', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('')
ax1.set_ylabel('')

# Find top 3 most correlated pairs (excluding diagonal and duplicates)
corr_pairs = []
for i in range(len(audio_features)):
    for j in range(i+1, len(audio_features)):
        corr_pairs.append((audio_features[i], audio_features[j], abs(correlation_matrix.iloc[i, j])))
top_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:3]

# Top-right: Scatter plot matrix for top 3 correlated pairs
ax2 = axes[0, 1]
# Use the first pair for the main scatter plot
feature1, feature2, _ = top_pairs[0]
for genre in datasets.keys():
    genre_data = df_combined[df_combined['Genre'] == genre]
    ax2.scatter(genre_data[feature1], genre_data[feature2], 
               c=genre_colors[genre], alpha=0.6, s=30, label=genre)

ax2.set_xlabel(feature1.capitalize(), fontweight='bold')
ax2.set_ylabel(feature2.capitalize(), fontweight='bold')
ax2.set_title(f'Scatter Plot: {feature1.capitalize()} vs {feature2.capitalize()}', 
              fontweight='bold', fontsize=14, pad=20)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Bottom-left: Bubble plot (Energy vs Valence, size = Popularity)
ax3 = axes[1, 0]
for genre in datasets.keys():
    genre_data = df_combined[df_combined['Genre'] == genre]
    # Sample data to avoid overcrowding
    if len(genre_data) > 200:
        genre_data = genre_data.sample(200, random_state=42)
    
    # Normalize popularity for bubble size (min 20, max 200)
    sizes = 20 + (genre_data['Popularity'] / 100) * 180
    
    ax3.scatter(genre_data['energy'], genre_data['valence'], 
               s=sizes, c=genre_colors[genre], alpha=0.6, label=genre)

ax3.set_xlabel('Energy', fontweight='bold')
ax3.set_ylabel('Valence', fontweight='bold')
ax3.set_title('Energy vs Valence (Bubble Size = Popularity)', 
              fontweight='bold', fontsize=14, pad=20)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Bottom-right: Violin plot with strip plot for acousticness
ax4 = axes[1, 1]
# Prepare data for violin plot
acousticness_data = []
genre_labels = []
for genre in datasets.keys():
    genre_data = df_combined[df_combined['Genre'] == genre]['acousticness']
    acousticness_data.append(genre_data)
    genre_labels.append(genre)

# Create violin plot
parts = ax4.violinplot(acousticness_data, positions=range(len(genre_labels)), 
                       showmeans=True, showmedians=True)

# Color the violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Add strip plot overlay
for i, genre in enumerate(datasets.keys()):
    genre_data = df_combined[df_combined['Genre'] == genre]['acousticness']
    # Sample for strip plot to avoid overcrowding
    if len(genre_data) > 100:
        genre_data = genre_data.sample(100, random_state=42)
    
    y_pos = np.random.normal(i, 0.04, len(genre_data))
    ax4.scatter(genre_data, y_pos, alpha=0.4, s=8, color=colors[i])

ax4.set_ylabel('Genre', fontweight='bold')
ax4.set_xlabel('Acousticness', fontweight='bold')
ax4.set_title('Acousticness Distribution Across Genres', 
              fontweight='bold', fontsize=14, pad=20)
ax4.set_yticks(range(len(genre_labels)))
ax4.set_yticklabels(genre_labels)
ax4.grid(True, alpha=0.3, axis='x')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.4)

# Add overall title
fig.suptitle('Comprehensive Audio Features Correlation Analysis Across Music Genres', 
             fontweight='bold', fontsize=16, y=0.98)

plt.show()