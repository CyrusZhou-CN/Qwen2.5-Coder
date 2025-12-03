import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
club_info = pd.read_csv('global_club_info.csv')
player_info = pd.read_csv('global_player_info.csv')
club_rankings = pd.read_csv('global_club_rankings.csv')

# Data preprocessing for club analysis (optimized)
# Calculate member count more efficiently
member_trophy_cols = [f'member_{i}_trophies' for i in range(1, 31)]
existing_cols = [col for col in member_trophy_cols if col in club_info.columns]

club_info['member_count'] = club_info[existing_cols].notna().sum(axis=1)
club_info['avg_member_trophies'] = club_info[existing_cols].mean(axis=1, skipna=True)

# Merge with rankings for badge info
club_data = pd.merge(club_info, club_rankings[['tag', 'badgeId']], on='tag', how='left')

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Define consistent color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#FF6B6B', '#4ECDC4']

# Subplot 1: Club trophies vs member count with color coding
ax1 = plt.subplot(3, 2, 1)
unique_badges = club_data['badgeId'].dropna().unique()[:6]  # Limit to 6 badges
for i, badge_id in enumerate(unique_badges):
    mask = club_data['badgeId'] == badge_id
    ax1.scatter(club_data[mask]['member_count'], club_data[mask]['trophies'], 
               c=colors[i], alpha=0.7, s=50, label=f'Badge {badge_id}')

ax1.set_xlabel('Member Count', fontweight='bold')
ax1.set_ylabel('Club Trophies', fontweight='bold')
ax1.set_title('Club Trophies vs Member Count by Badge Type', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Subplot 2: Dual-axis plot
ax2 = plt.subplot(3, 2, 2)
# Create trophy ranges
club_data['trophy_range'] = pd.cut(club_data['trophies'], bins=4, labels=['Low', 'Med', 'High', 'Elite'])
trophy_counts = club_data['trophy_range'].value_counts().sort_index()

# Bar chart
bars = ax2.bar(range(len(trophy_counts)), trophy_counts.values, 
               color=colors[:len(trophy_counts)], alpha=0.7)
ax2.set_xlabel('Trophy Range', fontweight='bold')
ax2.set_ylabel('Number of Clubs', fontweight='bold')
ax2.set_xticks(range(len(trophy_counts)))
ax2.set_xticklabels(trophy_counts.index)

# Line plot on secondary axis
ax2_twin = ax2.twinx()
size_groups = club_data.groupby('member_count')['avg_member_trophies'].mean().head(10)
ax2_twin.plot(range(len(size_groups)), size_groups.values, 
              color='red', marker='o', linewidth=2, markersize=6, label='Avg Trophies/Member')
ax2_twin.set_ylabel('Avg Trophies per Member', fontweight='bold', color='red')
ax2_twin.legend(loc='upper right', fontsize=8)
ax2.set_title('Club Distribution and Performance Metrics', fontweight='bold', fontsize=12)

# Subplot 3: Bubble chart
ax3 = plt.subplot(3, 2, 3)
# Sample data for performance
sample_clubs = club_data.dropna(subset=['member_count', 'avg_member_trophies', 'trophies']).head(50)
bubble_sizes = (sample_clubs['trophies'] / sample_clubs['trophies'].max()) * 300 + 30

# Color by club type
unique_types = sample_clubs['type'].dropna().unique()[:4]
type_colors = {t: colors[i] for i, t in enumerate(unique_types)}

for club_type in unique_types:
    mask = sample_clubs['type'] == club_type
    if mask.sum() > 0:
        ax3.scatter(sample_clubs[mask]['member_count'], sample_clubs[mask]['avg_member_trophies'],
                   s=bubble_sizes[mask], alpha=0.6, label=club_type, c=type_colors[club_type])

# Add trend line
valid_data = sample_clubs.dropna(subset=['member_count', 'avg_member_trophies'])
if len(valid_data) > 1:
    z = np.polyfit(valid_data['member_count'], valid_data['avg_member_trophies'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(valid_data['member_count'].min(), valid_data['member_count'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

ax3.set_xlabel('Member Count', fontweight='bold')
ax3.set_ylabel('Average Member Trophies', fontweight='bold')
ax3.set_title('Club Performance Bubble Chart', fontweight='bold', fontsize=12)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Player scatter plot with club membership
ax4 = plt.subplot(3, 2, 4)
# Sample players for performance
sample_players = player_info.dropna(subset=['trophies', 'highestTrophies']).head(100)
has_club = sample_players['club_tag'].notna()

ax4.scatter(sample_players[has_club]['trophies'], sample_players[has_club]['highestTrophies'], 
           c=colors[0], alpha=0.6, s=30, label='Has Club')
ax4.scatter(sample_players[~has_club]['trophies'], sample_players[~has_club]['highestTrophies'], 
           c=colors[1], alpha=0.6, s=30, label='No Club')

ax4.set_xlabel('Current Trophies', fontweight='bold')
ax4.set_ylabel('Highest Trophies', fontweight='bold')
ax4.set_title('Player Trophy Patterns by Club Membership', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)

# Subplot 5: Experience level analysis
ax5 = plt.subplot(3, 2, 5)
# Calculate top brawler performance efficiently
brawler_trophy_cols = [col for col in player_info.columns if '_trophies' in col and 'highest' not in col.lower()]
brawler_trophy_cols = [col for col in brawler_trophy_cols if not col.startswith('member_')][:10]  # Limit columns

if brawler_trophy_cols:
    sample_for_brawlers = player_info[['expLevel'] + brawler_trophy_cols].dropna().head(200)
    sample_for_brawlers['top3_avg'] = sample_for_brawlers[brawler_trophy_cols].apply(
        lambda x: x.nlargest(3).mean() if x.sum() > 0 else 0, axis=1)
    
    # Create experience bins
    sample_for_brawlers['exp_bin'] = pd.cut(sample_for_brawlers['expLevel'], 
                                           bins=4, labels=['Beginner', 'Intermediate', 'Advanced', 'Expert'])
    
    # Box plot instead of violin for performance
    exp_groups = [sample_for_brawlers[sample_for_brawlers['exp_bin'] == level]['top3_avg'].values 
                  for level in ['Beginner', 'Intermediate', 'Advanced', 'Expert']]
    
    bp = ax5.boxplot(exp_groups, labels=['Beginner', 'Intermediate', 'Advanced', 'Expert'],
                     patch_artist=True)
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)

ax5.set_xlabel('Experience Level', fontweight='bold')
ax5.set_ylabel('Top 3 Brawler Trophy Average', fontweight='bold')
ax5.set_title('Brawler Performance by Experience Level', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot 6: Simplified clustering visualization
ax6 = plt.subplot(3, 2, 6)
# Use a smaller sample for clustering
cluster_cols = ['trophies', 'highestTrophies', 'expLevel']
cluster_sample = player_info[cluster_cols].dropna().head(30)  # Small sample for performance

if len(cluster_sample) > 3:
    # Standardize the data
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_sample)
    
    # Perform hierarchical clustering
    try:
        linkage_matrix = linkage(cluster_data_scaled, method='ward')
        
        # Create dendrogram
        dendrogram(linkage_matrix, ax=ax6, leaf_rotation=90, leaf_font_size=6,
                   color_threshold=0.7*max(linkage_matrix[:,2]), no_labels=True)
        
        ax6.set_xlabel('Player Clusters', fontweight='bold')
        ax6.set_ylabel('Distance', fontweight='bold')
        ax6.set_title('Player Hierarchical Clustering', fontweight='bold', fontsize=12)
    except:
        # Fallback: simple scatter plot
        ax6.scatter(cluster_sample['trophies'], cluster_sample['highestTrophies'], 
                   c=colors[0], alpha=0.6, s=40)
        ax6.set_xlabel('Current Trophies', fontweight='bold')
        ax6.set_ylabel('Highest Trophies', fontweight='bold')
        ax6.set_title('Player Performance Scatter', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)

# Add overall title
fig.suptitle('Brawl Stars: Club Performance and Player Clustering Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.94)
plt.savefig('brawl_stars_analysis.png', dpi=300, bbox_inches='tight')
plt.show()