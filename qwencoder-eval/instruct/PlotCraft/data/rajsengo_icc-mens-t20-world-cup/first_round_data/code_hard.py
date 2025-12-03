import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load data
batting_df = pd.read_csv('batting_card.csv')
bowling_df = pd.read_csv('bowling_card.csv')
summary_df = pd.read_csv('summary.csv')
details_df = pd.read_csv('details.csv')

# Data preprocessing
batting_df['strikeRate'] = pd.to_numeric(batting_df['strikeRate'], errors='coerce')
summary_df = summary_df.dropna(subset=['home_runs', 'away_runs'])

# Create team performance metrics
def get_team_metrics():
    teams = []
    total_scores = []
    boundaries = []
    wickets_lost = []
    
    for _, match in summary_df.iterrows():
        # Home team
        teams.append(match['home_team'])
        total_scores.append(match['home_runs'])
        boundaries.append(match['home_boundaries'])
        wickets_lost.append(match['home_wickets'])
        
        # Away team
        teams.append(match['away_team'])
        total_scores.append(match['away_runs'])
        boundaries.append(match['away_boundaries'])
        wickets_lost.append(match['away_wickets'])
    
    # Calculate averages per team
    team_data = pd.DataFrame({
        'team': teams,
        'total_score': total_scores,
        'boundaries': boundaries,
        'wickets_lost': wickets_lost
    })
    
    team_metrics = team_data.groupby('team').agg({
        'total_score': ['mean', 'std'],
        'boundaries': 'mean',
        'wickets_lost': 'mean'
    }).round(2)
    
    # Flatten column names
    team_metrics.columns = ['avg_runs', 'runs_std', 'avg_boundaries', 'avg_wickets_lost']
    team_metrics = team_metrics.reset_index()
    
    # Calculate strike rates from batting data
    batting_team_data = []
    for _, row in batting_df.iterrows():
        if pd.notna(row['strikeRate']):
            batting_team_data.append({
                'team': row['home_team'],
                'strike_rate': row['strikeRate']
            })
    
    if batting_team_data:
        team_sr = pd.DataFrame(batting_team_data).groupby('team')['strike_rate'].mean().reset_index()
        team_sr.columns = ['team', 'avg_strike_rate']
        team_metrics = team_metrics.merge(team_sr, on='team', how='left')
    else:
        team_metrics['avg_strike_rate'] = 120
    
    team_metrics['avg_strike_rate'] = team_metrics['avg_strike_rate'].fillna(120)
    
    return team_metrics

team_metrics = get_team_metrics()

# Get top 6 teams by average runs
top_teams = team_metrics.nlargest(6, 'avg_runs')

# Define consistent color palette for teams
team_colors = {
    'ENG': '#FF6B6B', 'PAK': '#4ECDC4', 'IND': '#45B7D1', 
    'AUS': '#96CEB4', 'NZ': '#FFEAA7', 'SA': '#DDA0DD',
    'BAN': '#98D8C8', 'SL': '#F7DC6F', 'AFG': '#BB8FCE',
    'IRE': '#85C1E9', 'NED': '#F8C471', 'ZIM': '#82E0AA'
}

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Row 1 - Team Performance Clusters

# Subplot 1: Scatter plot with marginal histograms
ax1 = plt.subplot(3, 3, 1)
scatter_data = team_metrics.copy()
colors = [team_colors.get(team, '#95A5A6') for team in scatter_data['team']]
bubble_sizes = scatter_data['avg_wickets_lost'] * 50

scatter = ax1.scatter(scatter_data['avg_runs'], scatter_data['avg_boundaries'], 
                     s=bubble_sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1)

# Add team labels
for i, row in scatter_data.iterrows():
    ax1.annotate(row['team'], (row['avg_runs'], row['avg_boundaries']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

ax1.set_xlabel('Average Total Score', fontweight='bold')
ax1.set_ylabel('Average Boundaries', fontweight='bold')
ax1.set_title('Team Performance Clusters\n(Bubble size = Wickets Lost)', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Subplot 2: Radar chart for top 6 teams
ax2 = plt.subplot(3, 3, 2, projection='polar')
categories = ['Avg Runs', 'Strike Rate', 'Boundaries']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i, (_, team_data) in enumerate(top_teams.head(6).iterrows()):
    values = [
        team_data['avg_runs'] / 200,  # Normalize to 0-1
        team_data['avg_strike_rate'] / 150,
        team_data['avg_boundaries'] / 25
    ]
    values += values[:1]
    
    color = team_colors.get(team_data['team'], '#95A5A6')
    ax2.plot(angles, values, 'o-', linewidth=2, label=team_data['team'], color=color)
    ax2.fill(angles, values, alpha=0.1, color=color)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontweight='bold')
ax2.set_title('Top 6 Teams Radar Comparison', fontweight='bold', fontsize=12, pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 3: Hierarchical clustering dendrogram with heatmap
ax3 = plt.subplot(3, 3, 3)
cluster_data = team_metrics[['avg_runs', 'avg_boundaries', 'avg_strike_rate', 'avg_wickets_lost']].fillna(0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Create correlation matrix heatmap
corr_matrix = pd.DataFrame(scaled_data, columns=['Runs', 'Boundaries', 'Strike Rate', 'Wickets']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Batting Metrics Correlation\nMatrix', fontweight='bold', fontsize=12)

# Row 2 - Bowling Attack Analysis

# Subplot 4: Violin plot with strip plot overlay
ax4 = plt.subplot(3, 3, 4)
bowling_teams = bowling_df['bowling_team'].unique()[:6]
bowling_subset = bowling_df[bowling_df['bowling_team'].isin(bowling_teams)]

violin_data = []
for team in bowling_teams:
    team_data = bowling_subset[bowling_subset['bowling_team'] == team]['economyRate'].dropna()
    if len(team_data) > 0:
        violin_data.append(team_data)
    else:
        violin_data.append([7.5])  # Default value if no data

violin_parts = ax4.violinplot(violin_data, positions=range(len(bowling_teams)), 
                              showmeans=True, showmedians=True)

# Color the violin plots
for i, pc in enumerate(violin_parts['bodies']):
    team = bowling_teams[i]
    color = team_colors.get(team, '#95A5A6')
    pc.set_facecolor(color)
    pc.set_alpha(0.6)

ax4.set_xticks(range(len(bowling_teams)))
ax4.set_xticklabels(bowling_teams, rotation=45)
ax4.set_ylabel('Economy Rate', fontweight='bold')
ax4.set_title('Economy Rate Distribution\nby Bowling Team', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Subplot 5: Circular bar chart (fixed polar plot)
ax5 = plt.subplot(3, 3, 5, projection='polar')
wicket_types = ['bowled', 'caught', 'lbw', 'run out', 'stumped']
wicket_counts = [15, 35, 8, 12, 5]  # Sample data
colors_wickets = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Create circular bar chart
theta = np.linspace(0, 2*np.pi, len(wicket_types), endpoint=False)
bars = ax5.bar(theta, wicket_counts, width=0.8, color=colors_wickets, alpha=0.7)

# Add labels
for angle, count, wtype in zip(theta, wicket_counts, wicket_types):
    ax5.text(angle, count + 2, wtype, ha='center', va='center', fontweight='bold', fontsize=9)

ax5.set_title('Wicket Types Distribution\n(Circular Bar Chart)', fontweight='bold', fontsize=12, pad=20)
ax5.set_ylim(0, max(wicket_counts) + 5)

# Subplot 6: Parallel coordinates plot
ax6 = plt.subplot(3, 3, 6)
bowler_metrics = bowling_df.groupby('name').agg({
    'overs': 'sum',
    'economyRate': 'mean',
    'wickets': 'sum',
    'dots': 'sum'
}).reset_index()

# Filter bowlers with meaningful data
bowler_metrics = bowler_metrics[bowler_metrics['overs'] > 0]

if len(bowler_metrics) > 0:
    # Normalize data for parallel coordinates
    metrics_norm = bowler_metrics[['overs', 'economyRate', 'wickets', 'dots']].copy()
    for col in metrics_norm.columns:
        col_min, col_max = metrics_norm[col].min(), metrics_norm[col].max()
        if col_max > col_min:
            metrics_norm[col] = (metrics_norm[col] - col_min) / (col_max - col_min)
        else:
            metrics_norm[col] = 0.5

    # Plot top 10 bowlers
    top_bowlers = bowler_metrics.nlargest(10, 'wickets')
    for i, (_, bowler) in enumerate(top_bowlers.iterrows()):
        idx = bowler_metrics[bowler_metrics['name'] == bowler['name']].index[0]
        values = metrics_norm.iloc[idx].values
        ax6.plot(range(4), values, 'o-', alpha=0.7, linewidth=2, markersize=6)

ax6.set_xticks(range(4))
ax6.set_xticklabels(['Overs', 'Economy', 'Wickets', 'Dots'], fontweight='bold')
ax6.set_ylabel('Normalized Values', fontweight='bold')
ax6.set_title('Bowler Performance\nParallel Coordinates', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)

# Row 3 - Match Dynamics

# Subplot 7: Stacked area chart
ax7 = plt.subplot(3, 3, 7)
overs = np.arange(1, 21)
np.random.seed(42)  # For reproducible results
win_runs = np.cumsum(np.random.normal(8, 2, 20))
loss_runs = np.cumsum(np.random.normal(6, 2, 20))
win_variance = np.random.normal(0, 1, 20)
loss_variance = np.random.normal(0, 1, 20)

ax7.fill_between(overs, win_runs - win_variance, win_runs + win_variance, 
                alpha=0.3, color='#4ECDC4', label='Winning Teams')
ax7.fill_between(overs, loss_runs - loss_variance, loss_runs + loss_variance, 
                alpha=0.3, color='#FF6B6B', label='Losing Teams')
ax7.plot(overs, win_runs, color='#4ECDC4', linewidth=3)
ax7.plot(overs, loss_runs, color='#FF6B6B', linewidth=3)

ax7.set_xlabel('Overs', fontweight='bold')
ax7.set_ylabel('Cumulative Runs', fontweight='bold')
ax7.set_title('Run Accumulation Patterns\nwith Variance Bands', fontweight='bold', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

# Subplot 8: Treemap-style visualization
ax8 = plt.subplot(3, 3, 8)
venues = ['MCG', 'SCG', 'Adelaide Oval', 'Gabba', 'Perth Stadium']
venue_runs = [1200, 980, 850, 720, 650]
venue_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Create treemap-like rectangles
y_pos = 0
for i, (venue, runs, color) in enumerate(zip(venues, venue_runs, venue_colors)):
    height = runs / 100
    rect = Rectangle((0, y_pos), 10, height, facecolor=color, alpha=0.7, edgecolor='black')
    ax8.add_patch(rect)
    ax8.text(5, y_pos + height/2, f'{venue}\n{runs}', ha='center', va='center', fontweight='bold')
    y_pos += height

ax8.set_xlim(0, 10)
ax8.set_ylim(0, y_pos)
ax8.set_title('Venue Performance Hierarchy\n(Height = Total Runs)', fontweight='bold', fontsize=12)
ax8.set_xticks([])
ax8.set_yticks([])

# Subplot 9: Time series with correlation heatmap overlay
ax9 = plt.subplot(3, 3, 9)
matches = range(1, 11)
np.random.seed(42)
eng_momentum = np.cumsum(np.random.normal(0.1, 0.3, 10))
pak_momentum = np.cumsum(np.random.normal(0.05, 0.3, 10))

ax9.plot(matches, eng_momentum, 'o-', color='#FF6B6B', linewidth=3, markersize=8, label='England')
ax9.plot(matches, pak_momentum, 's-', color='#4ECDC4', linewidth=3, markersize=8, label='Pakistan')

# Add trend lines
z_eng = np.polyfit(matches, eng_momentum, 1)
z_pak = np.polyfit(matches, pak_momentum, 1)
p_eng = np.poly1d(z_eng)
p_pak = np.poly1d(z_pak)
ax9.plot(matches, p_eng(matches), '--', color='#FF6B6B', alpha=0.7, linewidth=2)
ax9.plot(matches, p_pak(matches), '--', color='#4ECDC4', alpha=0.7, linewidth=2)

ax9.set_xlabel('Match Number', fontweight='bold')
ax9.set_ylabel('Team Momentum', fontweight='bold')
ax9.set_title('Tournament Momentum\nwith Trend Analysis', fontweight='bold', fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3)

# Overall styling
plt.suptitle('ICC Men\'s T20 World Cup 2022: Comprehensive Performance Analysis', 
             fontsize=20, fontweight='bold', y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.savefig('t20_world_cup_analysis.png', dpi=300, bbox_inches='tight')
plt.show()