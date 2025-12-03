import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.patches import Circle
import networkx as nx
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('topuniversities.csv')

# Data preprocessing
# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Filter for Asian countries (based on the sample data)
asian_countries = ['China', 'Hong Kong', 'Singapore', 'South Korea', 'Japan', 'Taiwan', 'Malaysia', 'Thailand', 'India', 'Indonesia']
df_asia = df[df['Country'].isin(asian_countries)].copy()

# Create ranking tiers
def get_ranking_tier(rank):
    if rank <= 20:
        return '1-20'
    elif rank <= 50:
        return '21-50'
    else:
        return '51-100'

df_asia['Ranking_Tier'] = df_asia['Rank'].apply(get_ranking_tier)

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(24, 20), facecolor='white')
fig.patch.set_facecolor('white')

# Define color palette for countries
countries = df_asia['Country'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
country_colors = dict(zip(countries, colors))

# 1. Top-left: Scatter plot with marginal histograms
ax1 = plt.subplot2grid((3, 3), (0, 0), facecolor='white')
for country in countries:
    country_data = df_asia[df_asia['Country'] == country]
    scatter = ax1.scatter(country_data['Academic Reputation'], 
                         country_data['Employer Reputation'],
                         s=country_data['Overall Score']*3,
                         c=[country_colors[country]], 
                         alpha=0.7, 
                         label=country,
                         edgecolors='black', 
                         linewidth=0.5)

ax1.set_xlabel('Academic Reputation', fontweight='bold')
ax1.set_ylabel('Employer Reputation', fontweight='bold')
ax1.set_title('Academic vs Employer Reputation\n(Bubble size = Overall Score)', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Top-center: Stacked bar chart with line overlay
ax2 = plt.subplot2grid((3, 3), (0, 1), facecolor='white')
country_counts = df_asia['Country'].value_counts()
country_avg_scores = df_asia.groupby('Country')['Overall Score'].mean()

bars = ax2.bar(range(len(country_counts)), country_counts.values, 
               color=[country_colors[country] for country in country_counts.index],
               alpha=0.7, edgecolor='black', linewidth=0.5)

ax2_twin = ax2.twinx()
line_x = range(len(country_counts))
line_y = [country_avg_scores[country] for country in country_counts.index]
ax2_twin.plot(line_x, line_y, 'ro-', linewidth=2, markersize=6, color='red')

ax2.set_xlabel('Countries', fontweight='bold')
ax2.set_ylabel('University Count', fontweight='bold')
ax2_twin.set_ylabel('Average Overall Score', fontweight='bold', color='red')
ax2.set_title('University Count by Country\nwith Average Scores', fontweight='bold', fontsize=12)
ax2.set_xticks(range(len(country_counts)))
ax2.set_xticklabels(country_counts.index, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# 3. Top-right: Box plot with violin overlay
ax3 = plt.subplot2grid((3, 3), (0, 2), facecolor='white')
countries_with_data = [country for country in countries if len(df_asia[df_asia['Country'] == country]) > 2]
citation_data = [df_asia[df_asia['Country'] == country]['Citations per Paper'].values 
                for country in countries_with_data]

# Violin plot
parts = ax3.violinplot(citation_data, positions=range(len(countries_with_data)), 
                      widths=0.6, showmeans=True, showmedians=True)
for pc, country in zip(parts['bodies'], countries_with_data):
    pc.set_facecolor(country_colors[country])
    pc.set_alpha(0.7)

# Box plot overlay
bp = ax3.boxplot(citation_data, positions=range(len(countries_with_data)), 
                widths=0.3, patch_artist=True, 
                boxprops=dict(facecolor='white', alpha=0.8))

ax3.set_xlabel('Countries', fontweight='bold')
ax3.set_ylabel('Citations per Paper', fontweight='bold')
ax3.set_title('Citations per Paper Distribution\nby Country', fontweight='bold', fontsize=12)
ax3.set_xticks(range(len(countries_with_data)))
ax3.set_xticklabels(countries_with_data, rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# 4. Middle-left: Radar chart
ax4 = plt.subplot2grid((3, 3), (1, 0), projection='polar', facecolor='white')
top_countries = df_asia['Country'].value_counts().head(5).index
metrics = ['Academic Reputation', 'Employer Reputation', 'International Students', 
          'International Faculty', 'Faculty Student Ratio']

angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]

for i, country in enumerate(top_countries):
    country_data = df_asia[df_asia['Country'] == country]
    values = [country_data[metric].mean() for metric in metrics]
    values += values[:1]
    
    ax4.plot(angles, values, 'o-', linewidth=2, label=country, 
            color=country_colors[country])
    ax4.fill(angles, values, alpha=0.25, color=country_colors[country])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics, fontsize=10)
ax4.set_title('Performance Radar Chart\nTop 5 Countries', fontweight='bold', fontsize=12, pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax4.grid(True)

# 5. Middle-center: Correlation heatmap with dendrogram
ax5 = plt.subplot2grid((3, 3), (1, 1), facecolor='white')
numeric_cols = ['Overall Score', 'Citations per Paper', 'Papers per Faculty', 
               'Academic Reputation', 'Faculty Student Ratio', 'International Students', 
               'International Faculty', 'Employer Reputation']
corr_matrix = df_asia[numeric_cols].corr()

# Create dendrogram
linkage_matrix = linkage(pdist(corr_matrix), method='ward')
dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, 
                   orientation='top', ax=ax5, color_threshold=0)

# Clear and create heatmap
ax5.clear()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
           square=True, ax=ax5, cbar_kws={'shrink': 0.8})
ax5.set_title('Performance Metrics\nCorrelation Matrix', fontweight='bold', fontsize=12)

# 6. Middle-right: Parallel coordinates
ax6 = plt.subplot2grid((3, 3), (1, 2), facecolor='white')
top_20_unis = df_asia.nsmallest(20, 'Rank')
parallel_metrics = ['Academic Reputation', 'Employer Reputation', 
                   'Citations per Paper', 'International Students']

# Normalize data for parallel coordinates
normalized_data = top_20_unis[parallel_metrics].copy()
for col in parallel_metrics:
    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                          (normalized_data[col].max() - normalized_data[col].min())

for idx, row in normalized_data.iterrows():
    country = top_20_unis.loc[idx, 'Country']
    ax6.plot(range(len(parallel_metrics)), row.values, 
            color=country_colors[country], alpha=0.7, linewidth=1.5)

ax6.set_xticks(range(len(parallel_metrics)))
ax6.set_xticklabels(parallel_metrics, rotation=45, ha='right')
ax6.set_ylabel('Normalized Score', fontweight='bold')
ax6.set_title('Performance Profiles\nTop 20 Universities', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)

# 7. Bottom-left: Treemap simulation with nested rectangles
ax7 = plt.subplot2grid((3, 3), (2, 0), facecolor='white')
country_city_counts = df_asia.groupby(['Country', 'City']).size().reset_index(name='count')
country_totals = df_asia.groupby('Country').size().sort_values(ascending=False)

y_pos = 0
for country in country_totals.index[:6]:  # Top 6 countries
    country_data = country_city_counts[country_city_counts['Country'] == country]
    width = country_totals[country] * 0.1
    height = 0.8
    
    rect = patches.Rectangle((0, y_pos), width, height, 
                           facecolor=country_colors[country], 
                           alpha=0.7, edgecolor='black')
    ax7.add_patch(rect)
    ax7.text(width/2, y_pos + height/2, f'{country}\n({country_totals[country]})', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    y_pos += 1

ax7.set_xlim(0, max(country_totals) * 0.12)
ax7.set_ylim(0, 6)
ax7.set_title('University Count by Country\n(Treemap Style)', fontweight='bold', fontsize=12)
ax7.axis('off')

# 8. Bottom-center: Network graph simulation
ax8 = plt.subplot2grid((3, 3), (2, 1), facecolor='white')
# Create similarity matrix based on performance metrics
country_profiles = df_asia.groupby('Country')[numeric_cols].mean()
similarity_matrix = 1 - pdist(country_profiles, metric='euclidean')
similarity_matrix = squareform(similarity_matrix)

# Create network positions
n_countries = len(country_profiles)
angles = np.linspace(0, 2*np.pi, n_countries, endpoint=False)
positions = [(np.cos(angle), np.sin(angle)) for angle in angles]

# Draw nodes
for i, (country, pos) in enumerate(zip(country_profiles.index, positions)):
    size = df_asia[df_asia['Country'] == country].shape[0] * 100
    circle = Circle(pos, 0.15, facecolor=country_colors[country], 
                   alpha=0.7, edgecolor='black')
    ax8.add_patch(circle)
    ax8.text(pos[0], pos[1], country[:3], ha='center', va='center', 
            fontweight='bold', fontsize=8)

# Draw edges for high similarity
threshold = np.percentile(similarity_matrix, 80)
for i in range(n_countries):
    for j in range(i+1, n_countries):
        if similarity_matrix[i, j] > threshold:
            ax8.plot([positions[i][0], positions[j][0]], 
                    [positions[i][1], positions[j][1]], 
                    'gray', alpha=0.5, linewidth=1)

ax8.set_xlim(-1.5, 1.5)
ax8.set_ylim(-1.5, 1.5)
ax8.set_title('Country Similarity Network\n(Performance Metrics)', fontweight='bold', fontsize=12)
ax8.axis('off')

# 9. Bottom-right: PCA cluster analysis
ax9 = plt.subplot2grid((3, 3), (2, 2), facecolor='white')
# Prepare data for PCA
pca_features = ['Academic Reputation', 'Employer Reputation', 'Citations per Paper', 
               'International Students', 'International Faculty', 'Overall Score']
X = df_asia[pca_features].fillna(df_asia[pca_features].mean())

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Plot points colored by country and shaped by ranking tier
tier_markers = {'1-20': 'o', '21-50': 's', '51-100': '^'}
for country in countries:
    country_mask = df_asia['Country'] == country
    if country_mask.sum() > 0:
        for tier in ['1-20', '21-50', '51-100']:
            tier_mask = df_asia['Ranking_Tier'] == tier
            combined_mask = country_mask & tier_mask
            if combined_mask.sum() > 0:
                ax9.scatter(X_pca[combined_mask, 0], X_pca[combined_mask, 1],
                           c=[country_colors[country]], marker=tier_markers[tier],
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                           label=f'{country} ({tier})' if tier == '1-20' else '')

# Plot cluster centroids
centroids_pca = pca.transform(scaler.transform(kmeans.cluster_centers_))
ax9.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='red', marker='x', s=200, linewidth=3, label='Centroids')

ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
ax9.set_title('PCA Cluster Analysis\n(Performance Metrics)', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.show()