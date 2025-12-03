import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
import networkx as nx
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Load data
df_teams = pd.read_csv('matches_by_teams.csv')

# Data preprocessing - aggregate team performance metrics (sample subset for speed)
df_sample = df_teams.sample(n=min(1000, len(df_teams)), random_state=42)

team_metrics = df_sample.groupby('team_name').agg({
    'core_score': 'mean',
    'core_shooting_percentage': 'mean',
    'boost_amount_collected': 'mean',
    'core_shots': 'mean',
    'core_goals': 'mean',
    'core_saves': 'mean',
    'core_assists': 'mean',
    'positioning_time_defensive_third': 'mean',
    'positioning_time_neutral_third': 'mean',
    'positioning_time_offensive_third': 'mean',
    'movement_time_supersonic_speed': 'mean',
    'movement_time_ground': 'mean',
    'boost_bpm': 'mean',
    'boost_avg_amount': 'mean',
    'boost_time_zero_boost': 'mean',
    'demo_inflicted': 'mean',
    'demo_taken': 'mean',
    'winner': 'sum',
    'match_id': 'count'
}).reset_index()

# Remove teams with insufficient data and limit to top 20 teams for performance
team_metrics = team_metrics[team_metrics['match_id'] >= 3].head(20).reset_index(drop=True)

# Create figure
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Rocket League Championship Series: Team Clustering & Hierarchical Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Scatter plot with KDE contours
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Prepare data for clustering
perf_data = team_metrics[['core_score', 'core_shooting_percentage']].dropna()
if len(perf_data) > 4:
    # KMeans clustering
    n_clusters = min(4, len(perf_data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(perf_data)
    
    # Create scatter plot
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i in range(n_clusters):
        mask = clusters == i
        if np.sum(mask) > 0:
            boost_sizes = team_metrics.loc[perf_data.index[mask], 'boost_amount_collected'] / 200
            ax1.scatter(perf_data.iloc[mask, 0], perf_data.iloc[mask, 1], 
                       c=colors[i], s=boost_sizes, alpha=0.7, label=f'Cluster {i+1}')
    
    # Add simple contours if enough data
    if len(perf_data) > 10:
        try:
            x_range = np.linspace(perf_data.iloc[:, 0].min(), perf_data.iloc[:, 0].max(), 20)
            y_range = np.linspace(perf_data.iloc[:, 1].min(), perf_data.iloc[:, 1].max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)
            ax1.contour(xx, yy, np.ones_like(xx), levels=3, colors='gray', alpha=0.3)
        except:
            pass

ax1.set_xlabel('Core Score')
ax1.set_ylabel('Core Shooting %')
ax1.set_title('Performance Clustering Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Correlation heatmap
ax2 = plt.subplot(3, 3, 2)
perf_cols = ['core_shots', 'core_goals', 'core_saves', 'core_assists']
corr_data = team_metrics[perf_cols].dropna()

if len(corr_data) > 2:
    corr_matrix = corr_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax2, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})

ax2.set_title('Performance Metrics Correlation')

# Subplot 3: Network graph (simplified)
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Create simple network
G = nx.Graph()
top_teams = team_metrics.head(8)  # Limit to 8 teams for clarity

for _, team in top_teams.iterrows():
    G.add_node(team['team_name'][:8], wins=team['winner'])  # Truncate names

# Add edges based on similar performance
for i in range(len(top_teams)):
    for j in range(i+1, min(i+3, len(top_teams))):
        G.add_edge(top_teams.iloc[i]['team_name'][:8], 
                  top_teams.iloc[j]['team_name'][:8])

if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1, iterations=20)
    node_sizes = [max(50, G.nodes[node].get('wins', 1) * 50) for node in G.nodes()]
    nx.draw(G, pos, ax=ax3, node_size=node_sizes, node_color='#FF6B6B', 
            edge_color='gray', alpha=0.7, with_labels=True, font_size=8)

ax3.set_title('Team Performance Network')

# Subplot 4: Box plots for positioning
ax4 = plt.subplot(3, 3, 4)
pos_cols = ['positioning_time_defensive_third', 'positioning_time_neutral_third', 'positioning_time_offensive_third']
pos_data = team_metrics[pos_cols].dropna()

if len(pos_data) > 0:
    bp = ax4.boxplot([pos_data[col] for col in pos_cols], 
                     labels=['Defensive', 'Neutral', 'Offensive'], 
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#45B7D1')
        patch.set_alpha(0.7)

ax4.set_title('Positioning Time Distribution')
ax4.set_ylabel('Time (seconds)')

# Subplot 5: Movement scatter plot
ax5 = plt.subplot(3, 3, 5)
move_data = team_metrics[['movement_time_supersonic_speed', 'movement_time_ground']].dropna()

if len(move_data) > 3:
    n_clusters = min(3, len(move_data))
    move_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(move_data)
    
    for i in range(n_clusters):
        mask = move_clusters == i
        if np.sum(mask) > 0:
            ax5.scatter(move_data.iloc[mask, 0], move_data.iloc[mask, 1], 
                       c=colors[i], s=100, alpha=0.7, label=f'Movement {i+1}')

ax5.set_xlabel('Supersonic Speed Time')
ax5.set_ylabel('Ground Time')
ax5.set_title('Movement Pattern Clusters')
ax5.legend()

# Subplot 6: Radar chart (simplified)
ax6 = plt.subplot(3, 3, 6, projection='polar')
boost_cols = ['boost_bpm', 'boost_avg_amount', 'boost_time_zero_boost']
boost_data = team_metrics[boost_cols].dropna()

if len(boost_data) > 0:
    # Normalize data
    boost_norm = (boost_data - boost_data.min()) / (boost_data.max() - boost_data.min() + 1e-8)
    
    angles = np.linspace(0, 2*np.pi, len(boost_cols), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot top 3 teams
    for i, (_, team) in enumerate(boost_norm.head(3).iterrows()):
        values = team.tolist()
        values += values[:1]
        ax6.plot(angles, values, 'o-', linewidth=2, alpha=0.7, label=f'Team {i+1}')
        ax6.fill(angles, values, alpha=0.1)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(['BPM', 'Avg Amount', 'Zero Boost'])
ax6.set_title('Boost Management Profiles')
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 7: Scatter with trend line
ax7 = plt.subplot(3, 3, 7)
team_metrics['advanced_rating'] = team_metrics['core_score'] * 0.7 + team_metrics['winner'] * 0.3

rating_data = team_metrics[['advanced_rating', 'core_score']].dropna()
if len(rating_data) > 0:
    ax7.scatter(rating_data['advanced_rating'], rating_data['core_score'], 
               s=100, c='#96CEB4', alpha=0.7, edgecolors='white')
    
    # Add trend line
    if len(rating_data) > 1:
        z = np.polyfit(rating_data['advanced_rating'], rating_data['core_score'], 1)
        p = np.poly1d(z)
        ax7.plot(rating_data['advanced_rating'], p(rating_data['advanced_rating']), 
                "r--", alpha=0.8)

ax7.set_xlabel('Advanced Rating')
ax7.set_ylabel('Core Score')
ax7.set_title('Team Performance Hierarchy')

# Subplot 8: MDS plot (simplified)
ax8 = plt.subplot(3, 3, 8)
mds_features = ['core_score', 'boost_bpm', 'movement_time_supersonic_speed']
mds_data = team_metrics[mds_features].dropna()

if len(mds_data) > 4:
    # Standardize and apply MDS
    scaler = StandardScaler()
    mds_scaled = scaler.fit_transform(mds_data)
    
    mds = MDS(n_components=2, random_state=42, max_iter=100, n_init=1)
    mds_coords = mds.fit_transform(mds_scaled)
    
    # Simple clustering
    n_clusters = min(3, len(mds_coords))
    mds_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(mds_coords)
    
    for i in range(n_clusters):
        mask = mds_clusters == i
        if np.sum(mask) > 0:
            ax8.scatter(mds_coords[mask, 0], mds_coords[mask, 1], 
                       c=colors[i], s=100, alpha=0.7, label=f'Group {i+1}')

ax8.set_xlabel('MDS Dimension 1')
ax8.set_ylabel('MDS Dimension 2')
ax8.set_title('Multi-Dimensional Team Groups')
ax8.legend()

# Subplot 9: Grouped bar chart
ax9 = plt.subplot(3, 3, 9)
team_metrics['perf_tier'] = pd.cut(team_metrics['core_score'], bins=3, labels=['Low', 'Mid', 'High'])
demo_data = team_metrics[['perf_tier', 'demo_inflicted', 'demo_taken']].dropna()

if len(demo_data) > 0:
    tier_groups = demo_data.groupby('perf_tier').agg({
        'demo_inflicted': 'mean',
        'demo_taken': 'mean'
    }).reset_index()
    
    x = np.arange(len(tier_groups))
    width = 0.35
    
    ax9.bar(x - width/2, tier_groups['demo_inflicted'], width, 
           label='Demos Inflicted', color='#FF6B6B', alpha=0.8)
    ax9.bar(x + width/2, tier_groups['demo_taken'], width,
           label='Demos Taken', color='#4ECDC4', alpha=0.8)
    
    ax9.set_xticks(x)
    ax9.set_xticklabels(tier_groups['perf_tier'])

ax9.set_xlabel('Performance Tier')
ax9.set_ylabel('Demo Statistics')
ax9.set_title('Performance Tiers & Demo Analysis')
ax9.legend()

# Adjust layout and save
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('rocket_league_team_analysis.png', dpi=300, bbox_inches='tight')
plt.show()