import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations

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

# Combine all datasets and add genre labels
combined_data = []
for genre, df in datasets.items():
    df_copy = df.copy()
    df_copy['Genre'] = genre
    combined_data.append(df_copy)

df_all = pd.concat(combined_data, ignore_index=True)

# Define audio features for analysis
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create figure with 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palette for genres
genre_colors = {
    'Metal': '#8B0000',
    'Indie/Alt': '#4682B4', 
    'Alternative': '#32CD32',
    'Blues': '#191970',
    'Pop': '#FF69B4',
    'Hip-Hop': '#FF4500',
    'Rock': '#800080'
}

# Top-left: Correlation heatmap
ax1 = axes[0, 0]
correlation_matrix = df_all[audio_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
ax1.set_title('Audio Features Correlation Heatmap\n(All Genres Combined)', 
              fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Audio Features', fontweight='bold')
ax1.set_ylabel('Audio Features', fontweight='bold')

# Find the three most correlated pairs
corr_pairs = []
for i in range(len(audio_features)):
    for j in range(i+1, len(audio_features)):
        corr_val = abs(correlation_matrix.iloc[i, j])
        corr_pairs.append((audio_features[i], audio_features[j], corr_val))

top_corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:3]
most_corr_features = list(set([pair[0] for pair in top_corr_pairs] + [pair[1] for pair in top_corr_pairs]))

# Top-right: Energy vs Loudness scatter plot with best fit line
ax2 = axes[0, 1]
for genre in genre_colors.keys():
    genre_data = df_all[df_all['Genre'] == genre]
    ax2.scatter(genre_data['energy'], genre_data['loudness'], 
               c=genre_colors[genre], alpha=0.6, s=30, label=genre)

# Add best fit line for all data
x = df_all['energy'].dropna()
y = df_all['loudness'].dropna()
valid_idx = x.index.intersection(y.index)
x_valid = x.loc[valid_idx]
y_valid = y.loc[valid_idx]
z = np.polyfit(x_valid, y_valid, 1)
p = np.poly1d(z)
ax2.plot(x_valid.sort_values(), p(x_valid.sort_values()), "k--", alpha=0.8, linewidth=2)

# Add correlation coefficient
corr_coef = correlation_matrix.loc['energy', 'loudness']
ax2.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax2.transAxes, 
         fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax2.set_xlabel('Energy', fontweight='bold')
ax2.set_ylabel('Loudness (dB)', fontweight='bold')
ax2.set_title('Energy vs Loudness Relationship\nby Genre', fontweight='bold', fontsize=14, pad=20)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# Bottom-left: Danceability vs Valence bubble plot
ax3 = axes[1, 0]
for genre in genre_colors.keys():
    genre_data = df_all[df_all['Genre'] == genre]
    # Use popularity as bubble size (normalize to reasonable range)
    sizes = (genre_data['Popularity'] + 1) * 2  # +1 to avoid zero sizes
    ax3.scatter(genre_data['danceability'], genre_data['valence'], 
               s=sizes, c=genre_colors[genre], alpha=0.6, label=genre)

ax3.set_xlabel('Danceability', fontweight='bold')
ax3.set_ylabel('Valence', fontweight='bold')
ax3.set_title('Danceability vs Valence\n(Bubble Size = Popularity)', 
              fontweight='bold', fontsize=14, pad=20)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

# Add correlation coefficient
corr_coef_dv = correlation_matrix.loc['danceability', 'valence']
ax3.text(0.05, 0.95, f'r = {corr_coef_dv:.3f}', transform=ax3.transAxes, 
         fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Bottom-right: Pair plot of top 3 most correlated features
ax4 = axes[1, 1]

# Select top 3 most correlated features (ensure we have exactly 3)
if len(most_corr_features) >= 3:
    selected_features = most_corr_features[:3]
else:
    # Fallback to energy, loudness, valence if not enough highly correlated features
    selected_features = ['energy', 'loudness', 'valence']

# Create a simplified pair plot focusing on one key relationship
feature1, feature2 = selected_features[0], selected_features[1]

for genre in genre_colors.keys():
    genre_data = df_all[df_all['Genre'] == genre]
    ax4.scatter(genre_data[feature1], genre_data[feature2], 
               c=genre_colors[genre], alpha=0.6, s=30, label=genre)

ax4.set_xlabel(feature1.title(), fontweight='bold')
ax4.set_ylabel(feature2.title(), fontweight='bold')
ax4.set_title(f'Most Correlated Features:\n{feature1.title()} vs {feature2.title()}', 
              fontweight='bold', fontsize=14, pad=20)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

# Add correlation coefficient
corr_coef_pair = correlation_matrix.loc[feature1, feature2]
ax4.text(0.05, 0.95, f'r = {corr_coef_pair:.3f}', transform=ax4.transAxes, 
         fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add overall title and annotations
fig.suptitle('Comprehensive Audio Features Correlation Analysis Across Music Genres', 
             fontsize=18, fontweight='bold', y=0.98)

# Add text box with key findings
textstr = f'''Key Correlations Found:
• Strongest: {top_corr_pairs[0][0]} ↔ {top_corr_pairs[0][1]} (r={top_corr_pairs[0][2]:.3f})
• Second: {top_corr_pairs[1][0]} ↔ {top_corr_pairs[1][1]} (r={top_corr_pairs[1][2]:.3f})
• Third: {top_corr_pairs[2][0]} ↔ {top_corr_pairs[2][1]} (r={top_corr_pairs[2][2]:.3f})'''

fig.text(0.02, 0.02, textstr, fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.12, hspace=0.3, wspace=0.4)
plt.show()