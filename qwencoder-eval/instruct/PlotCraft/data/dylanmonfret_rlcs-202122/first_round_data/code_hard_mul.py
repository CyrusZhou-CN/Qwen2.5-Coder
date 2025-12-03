import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
teams_df = pd.read_csv('matches_by_teams.csv')
players_df = pd.read_csv('games_by_players.csv')
matches_players_df = pd.read_csv('matches_by_players.csv')

# Clean data and handle missing values - sample data for performance
teams_sample = teams_df.dropna(subset=['core_score', 'boost_bpm', 'positioning_time_offensive_third', 
                                      'positioning_time_defensive_third', 'demo_inflicted']).sample(n=min(1000, len(teams_df)), random_state=42)

players_sample = players_df.dropna(subset=['advanced_rating', 'core_shooting_percentage', 
                                          'boost_avg_amount', 'movement_percent_supersonic_speed',
                                          'positioning_percent_offensive_third']).sample(n=min(500, len(players_df)), random_state=42)

matches_sample = matches_players_df.dropna(subset=['advanced_rating']).sample(n=min(300, len(matches_players_df)), random_state=42)

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('white')

# Define consistent color palettes
region_colors = {'North America': '#FF6B6B', 'Europe': '#4ECDC4', 'Oceania': '#45B7D1', 
                'South America': '#96CEB4', 'Asia': '#FFEAA7', 'Middle East': '#DDA0DD'}
performance_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Subplot 1: Scatter plot with KDE contours and K-means clustering
ax1 = plt.subplot(3, 2, 1)

# Prepare data for clustering
cluster_data = teams_sample[['core_score', 'boost_bpm']].dropna()
if len(cluster_data) > 10:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Create scatter plot colored by winner status
    winners = teams_sample.loc[cluster_data.index, 'winner']
    colors = ['#FF6B6B' if w else '#4ECDC4' for w in winners]
    
    scatter = ax1.scatter(cluster_data['core_score'], cluster_data['boost_bpm'], 
                         c=colors, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    # Add simplified contours
    try:
        x = cluster_data['core_score']
        y = cluster_data['boost_bpm']
        if len(x) > 5:
            # Create a simple grid for contours
            x_range = np.linspace(x.min(), x.max(), 20)
            y_range = np.linspace(y.min(), y.max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)
            
            # Simple density estimation
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            kernel = gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            ax1.contour(xx, yy, f, colors='gray', alpha=0.3, linewidths=1, levels=3)
    except:
        pass
    
    # Add cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
               c='black', marker='x', s=100, linewidths=2, label='Cluster Centers')

ax1.set_xlabel('Core Score', fontweight='bold', fontsize=10)
ax1.set_ylabel('Boost BPM', fontweight='bold', fontsize=10)
ax1.set_title('Team Performance Clustering\nCore Score vs Boost BPM', 
              fontweight='bold', fontsize=12, pad=15)
ax1.grid(True, alpha=0.3)
ax1.legend(['Winners', 'Losers', 'Cluster Centers'], loc='upper right', fontsize=8)

# Subplot 2: Bubble chart
ax2 = plt.subplot(3, 2, 2)

# Prepare bubble chart data
bubble_data = teams_sample[['positioning_time_offensive_third', 'positioning_time_defensive_third', 
                           'demo_inflicted', 'team_region']].dropna()

# Create bubble chart
regions = bubble_data['team_region'].unique()[:5]  # Limit regions for clarity
for i, region in enumerate(regions):
    region_data = bubble_data[bubble_data['team_region'] == region]
    if len(region_data) > 0:
        ax2.scatter(region_data['positioning_time_offensive_third'], 
                   region_data['positioning_time_defensive_third'],
                   s=np.clip(region_data['demo_inflicted'] * 5, 10, 100), 
                   c=region_colors.get(region, '#999999'), 
                   alpha=0.6, label=region, edgecolors='white', linewidth=0.5)

ax2.set_xlabel('Offensive Third Time', fontweight='bold', fontsize=10)
ax2.set_ylabel('Defensive Third Time', fontweight='bold', fontsize=10)
ax2.set_title('Team Positioning by Region\nBubble Size = Demo Inflicted', 
              fontweight='bold', fontsize=12, pad=15)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Subplot 3: Simplified Network graph
ax3 = plt.subplot(3, 2, 3)

# Create simplified player network
G = nx.Graph()
player_data = matches_sample.groupby('player_id').agg({
    'advanced_rating': 'mean',
    'team_id': 'first'
}).head(20)  # Limit to 20 players

# Add nodes
for player_id, data in player_data.iterrows():
    G.add_node(player_id, rating=data['advanced_rating'], team=data['team_id'])

# Add edges based on team connections
players = list(G.nodes())
for i, player1 in enumerate(players):
    for player2 in players[i+1:i+5]:  # Limit connections
        if G.nodes[player1]['team'] == G.nodes[player2]['team']:
            G.add_edge(player1, player2)

if len(G.nodes()) > 0:
    # Simple community detection based on teams
    teams = [G.nodes[node]['team'] for node in G.nodes()]
    unique_teams = list(set(teams))
    node_colors = [performance_colors[unique_teams.index(team) % len(performance_colors)] for team in teams]
    
    # Draw network
    pos = nx.spring_layout(G, k=2, iterations=20)
    node_sizes = [max(20, G.nodes[node]['rating'] * 10) for node in G.nodes()]
    nx.draw(G, pos, ax=ax3, node_color=node_colors, node_size=node_sizes, 
            with_labels=False, edge_color='gray', alpha=0.7, width=0.5)

ax3.set_title('Player Network Analysis\nNode Size = Rating, Colors = Teams', 
              fontweight='bold', fontsize=12, pad=15)
ax3.axis('off')

# Subplot 4: Parallel coordinates plot
ax4 = plt.subplot(3, 2, 4)

# Prepare parallel coordinates data
parallel_metrics = ['core_shooting_percentage', 'boost_avg_amount', 
                   'movement_percent_supersonic_speed', 'positioning_percent_offensive_third']
parallel_data = players_sample[parallel_metrics + ['team_id']].dropna()

if len(parallel_data) > 0:
    # Normalize data
    normalized_data = parallel_data[parallel_metrics].copy()
    for col in parallel_metrics:
        col_min, col_max = normalized_data[col].min(), normalized_data[col].max()
        if col_max > col_min:
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
    
    # Sample teams for clarity
    sample_teams = parallel_data['team_id'].value_counts().head(5).index
    sample_data = parallel_data[parallel_data['team_id'].isin(sample_teams)]
    sample_normalized = normalized_data[parallel_data['team_id'].isin(sample_teams)]
    
    # Plot parallel coordinates
    x_pos = range(len(parallel_metrics))
    for i, (idx, row) in enumerate(sample_normalized.head(50).iterrows()):
        team_id = sample_data.loc[idx, 'team_id']
        color = performance_colors[hash(str(team_id)) % len(performance_colors)]
        ax4.plot(x_pos, row[parallel_metrics], color=color, alpha=0.4, linewidth=1)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(['Shooting %', 'Boost Avg', 'Supersonic %', 'Offensive %'], 
                   rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Normalized Values', fontweight='bold', fontsize=10)
ax4.set_title('Player Performance Metrics\nParallel Coordinates', 
              fontweight='bold', fontsize=12, pad=15)
ax4.grid(True, alpha=0.3)

# Subplot 5: Correlation heatmap
ax5 = plt.subplot(3, 2, 5)

# Prepare correlation data
corr_metrics = ['core_goals', 'core_saves', 'boost_amount_collected', 
               'movement_time_supersonic_speed', 'demo_inflicted']
corr_data = teams_sample[corr_metrics].dropna()

if len(corr_data) > 10:
    correlation_matrix = corr_data.corr()
    
    # Create heatmap
    im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(correlation_matrix.columns)))
    ax5.set_yticks(range(len(correlation_matrix.index)))
    ax5.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax5.set_yticklabels(correlation_matrix.index, fontsize=9)
    
    # Add correlation values
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            text = ax5.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                           fontsize=8, fontweight='bold')

ax5.set_title('Team Performance Correlation Matrix', 
              fontweight='bold', fontsize=12, pad=15)

# Subplot 6: Simplified Treemap
ax6 = plt.subplot(3, 2, 6)

# Prepare treemap data
treemap_data = teams_sample.groupby(['team_region', 'team_name']).agg({
    'core_score': 'mean',
    'winner': 'mean'
}).reset_index()

# Create simplified treemap
regions = treemap_data['team_region'].unique()[:4]  # Limit regions
y_offset = 0

for i, region in enumerate(regions):
    region_data = treemap_data[treemap_data['team_region'] == region].head(5)  # Limit teams per region
    region_height = 0.2
    
    x_offset = 0
    for j, (_, team) in enumerate(region_data.iterrows()):
        width = max(0.1, min(0.4, team['core_score'] / 5000))  # Normalize width
        win_rate = team['winner']
        
        # Color based on win rate
        color = plt.cm.RdYlGn(win_rate)
        
        rect = Rectangle((x_offset, y_offset), width, region_height, 
                        facecolor=color, edgecolor='white', linewidth=1)
        ax6.add_patch(rect)
        
        # Add team name if rectangle is large enough
        if width > 0.15:
            ax6.text(x_offset + width/2, y_offset + region_height/2, 
                    str(team['team_name'])[:6], ha='center', va='center', 
                    fontsize=7, fontweight='bold')
        
        x_offset += width + 0.05
    
    # Add region label
    ax6.text(-0.05, y_offset + region_height/2, region, 
            ha='right', va='center', fontweight='bold', fontsize=9)
    
    y_offset += region_height + 0.1

ax6.set_xlim(-0.3, 1.5)
ax6.set_ylim(-0.05, y_offset)
ax6.set_title('Team Composition Treemap\nSize = Core Score, Color = Win Rate', 
              fontweight='bold', fontsize=12, pad=15)
ax6.axis('off')

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.4)

# Add overall title
fig.suptitle('Rocket League Championship Series: Team Clustering & Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('rlcs_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()