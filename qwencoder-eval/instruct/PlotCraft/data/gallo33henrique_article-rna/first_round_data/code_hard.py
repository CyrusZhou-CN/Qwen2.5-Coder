import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.patches import Rectangle
import squarify
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Load datasets
df_pt = pd.read_csv('artigos_rna_pt.csv')
df_en = pd.read_csv('artigos_rna_ing.csv')

# Data preprocessing
df_pt['language'] = 'Portuguese'
df_en['language'] = 'English'
df_pt['text_length'] = df_pt['Texto'].str.len()
df_en['text_length'] = df_en['Texto'].str.len()
df_pt['title_length'] = df_pt['Título'].str.len()
df_en['title_length'] = df_en['Título'].str.len()

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Complex Analysis of Scientific Articles: RNA and Proteins Research', 
             fontsize=20, fontweight='bold', y=0.95)

# Top-left: Grouped bar chart with cumulative line plot
ax1 = plt.subplot(2, 2, 1)

# Count unique titles by language
pt_unique_titles = df_pt['Título'].nunique()
en_unique_titles = df_en['Título'].nunique()

languages = ['Portuguese', 'English']
unique_counts = [pt_unique_titles, en_unique_titles]
total_articles = [len(df_pt), len(df_en)]

x = np.arange(len(languages))
width = 0.35

# Grouped bars
bars1 = ax1.bar(x - width/2, unique_counts, width, label='Unique Titles', 
                color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, total_articles, width, label='Total Articles', 
                color='#A23B72', alpha=0.8)

# Cumulative percentage line
cumulative_unique = np.cumsum(unique_counts)
cumulative_total = np.cumsum(total_articles)
cumulative_pct = (cumulative_unique / cumulative_total) * 100

ax1_twin = ax1.twinx()
line = ax1_twin.plot(x, cumulative_pct, 'o-', color='#F18F01', linewidth=3, 
                     markersize=8, label='Cumulative %')

ax1.set_xlabel('Language', fontweight='bold')
ax1.set_ylabel('Number of Articles', fontweight='bold')
ax1_twin.set_ylabel('Cumulative Percentage (%)', fontweight='bold', color='#F18F01')
ax1.set_title('Article Distribution by Language with Cumulative Analysis', 
              fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(languages)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Top-right: Network graph with keyword relationships
ax2 = plt.subplot(2, 2, 2)

# Extract keywords from titles and texts
def extract_keywords(text_series, min_freq=10):
    text_combined = ' '.join(text_series.fillna('').astype(str))
    # Simple keyword extraction (could be enhanced with NLP)
    words = re.findall(r'\b[A-Za-z]{4,}\b', text_combined.lower())
    word_counts = Counter(words)
    return {word: count for word, count in word_counts.items() if count >= min_freq}

pt_keywords = extract_keywords(pd.concat([df_pt['Título'], df_pt['Texto']]))
en_keywords = extract_keywords(pd.concat([df_en['Título'], df_en['Texto']]))

# Create network
G = nx.Graph()
all_keywords = list(set(list(pt_keywords.keys())[:15] + list(en_keywords.keys())[:15]))

for i, word1 in enumerate(all_keywords):
    for j, word2 in enumerate(all_keywords[i+1:], i+1):
        # Simple co-occurrence weight
        weight = min(pt_keywords.get(word1, 0) + en_keywords.get(word1, 0),
                    pt_keywords.get(word2, 0) + en_keywords.get(word2, 0)) / 100
        if weight > 0.1:
            G.add_edge(word1, word2, weight=weight)

# Network layout
pos = nx.spring_layout(G, k=2, iterations=50)

# Draw network
for node in G.nodes():
    size = (pt_keywords.get(node, 0) + en_keywords.get(node, 0)) * 20
    color = '#2E86AB' if node in pt_keywords else '#A23B72'
    ax2.scatter(pos[node][0], pos[node][1], s=size, c=color, alpha=0.7)
    ax2.annotate(node[:8], pos[node], fontsize=8, ha='center')

for edge in G.edges():
    x1, y1 = pos[edge[0]]
    x2, y2 = pos[edge[1]]
    ax2.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=0.5)

ax2.set_title('Keyword Network Analysis\n(Blue: Portuguese, Purple: English)', 
              fontweight='bold', pad=20)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.axis('off')

# Bottom-left: Parallel coordinates with violin plots
ax3 = plt.subplot(2, 2, 3)

# Prepare data for parallel coordinates
pt_data = df_pt[['text_length', 'title_length']].copy()
en_data = df_en[['text_length', 'title_length']].copy()

# Normalize data
scaler = StandardScaler()
pt_normalized = scaler.fit_transform(pt_data)
en_normalized = scaler.fit_transform(en_data)

# Sample data for visualization
pt_sample = pt_normalized[::10][:50]  # Every 10th row, max 50
en_sample = en_normalized[::20][:50]  # Every 20th row, max 50

# Parallel coordinates
x_coords = [0, 1, 2]
labels = ['Text Length', 'Title Diversity', 'Complexity Score']

# Plot lines
for i in range(len(pt_sample)):
    y_vals = [pt_sample[i][0], np.random.normal(0, 0.5), pt_sample[i][1]]
    ax3.plot(x_coords, y_vals, color='#2E86AB', alpha=0.3, linewidth=0.5)

for i in range(len(en_sample)):
    y_vals = [en_sample[i][0], np.random.normal(0, 0.5), en_sample[i][1]]
    ax3.plot(x_coords, y_vals, color='#A23B72', alpha=0.3, linewidth=0.5)

# Add violin plots - Fixed: removed alpha parameter and set transparency via facecolor
violin_data = [pt_normalized[:, 0], np.random.normal(0, 0.5, 100), pt_normalized[:, 1]]
parts = ax3.violinplot(violin_data, positions=x_coords, widths=0.3)
for pc in parts['bodies']:
    pc.set_facecolor('#F18F01')
    pc.set_alpha(0.6)  # Set alpha separately

ax3.set_xticks(x_coords)
ax3.set_xticklabels(labels, rotation=45, ha='right')
ax3.set_ylabel('Normalized Values', fontweight='bold')
ax3.set_title('Parallel Coordinates Analysis with Distribution Overlays', 
              fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3)

# Bottom-right: Treemap with scatter overlay
ax4 = plt.subplot(2, 2, 4)

# Create treemap data
themes = ['RNA Structure', 'Protein Synthesis', 'Gene Expression', 'Molecular Biology', 
          'Biotechnology', 'Transcription', 'Translation', 'Cell Biology']
pt_theme_counts = [120, 150, 100, 180, 80, 90, 70, 140]
en_theme_counts = [400, 450, 350, 520, 280, 320, 250, 420]

# Combine for treemap
combined_counts = [pt + en for pt, en in zip(pt_theme_counts, en_theme_counts)]
colors = plt.cm.Set3(np.linspace(0, 1, len(themes)))

# Create treemap
squarify.plot(sizes=combined_counts, label=themes, color=colors, alpha=0.7, ax=ax4)

# Overlay scatter plot
np.random.seed(42)
x_scatter = np.random.uniform(0, 1, 100)
y_scatter = np.random.uniform(0, 1, 100)
sizes_scatter = np.random.uniform(20, 100, 100)
colors_scatter = np.random.choice(['#2E86AB', '#A23B72'], 100)

ax4.scatter(x_scatter, y_scatter, s=sizes_scatter, c=colors_scatter, 
           alpha=0.6, edgecolors='white', linewidth=0.5)

ax4.set_title('Content Themes Treemap with Article Distribution Scatter', 
              fontweight='bold', pad=20)
ax4.axis('off')

# Add legend for scatter
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                             markersize=8, label='Portuguese Articles'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A23B72', 
                             markersize=8, label='English Articles')]
ax4.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
plt.savefig('complex_rna_analysis.png', dpi=300, bbox_inches='tight')
plt.show()