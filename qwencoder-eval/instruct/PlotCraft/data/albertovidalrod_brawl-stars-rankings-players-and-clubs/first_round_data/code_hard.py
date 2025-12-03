import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load data
club_info = pd.read_csv('global_club_info.csv')
player_rankings = pd.read_csv('global_player_rankings.csv')
player_info = pd.read_csv('global_player_info.csv')
club_rankings = pd.read_csv('global_club_rankings.csv')

# Data preprocessing
# Extract member data from club_info
members_data = []
for idx, row in club_info.iterrows():
    for i in range(1, 31):
        if pd.notna(row.get(f'member_{i}_tag')):
            members_data.append({
                'club_tag': row['tag'],
                'club_name': row['name'],
                'club_type': row['type'],
                'club_trophies': row['trophies'],
                'required_trophies': row['requiredTrophies'],
                'member_tag': row[f'member_{i}_tag'],
                'member_name': row[f'member_{i}_name'],
                'member_role': row[f'member_{i}_role'],
                'member_trophies': row[f'member_{i}_trophies']
            })

members_df = pd.DataFrame(members_data)
members_df = members_df.dropna()

# Merge with player info - fix the merge issue
player_data = player_info.copy()
if 'nameColor' in player_rankings.columns:
    player_data = player_data.merge(player_rankings[['tag', 'nameColor']], on='tag', how='left')
else:
    player_data['nameColor'] = '0xffffffff'  # Default color

# Create figure with white background
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Row 1: Player Performance Clustering Analysis

# Subplot 1: Scatter plot with KDE contours
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Create experience level categories
player_data['exp_category'] = pd.cut(player_data['expLevel'], bins=5, labels=['Novice', 'Intermediate', 'Advanced', 'Expert', 'Master'])
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, category in enumerate(['Novice', 'Intermediate', 'Advanced', 'Expert', 'Master']):
    mask = player_data['exp_category'] == category
    if mask.sum() > 0:
        ax1.scatter(player_data[mask]['trophies'], player_data[mask]['highestTrophies'], 
                   c=colors[i], alpha=0.6, s=player_data[mask]['3vs3Victories']/100, 
                   label=category, edgecolors='white', linewidth=0.5)

# Add KDE contours
if len(player_data) > 10:
    x = player_data['trophies'].values
    y = player_data['highestTrophies'].values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if valid_mask.sum() > 10:
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        kde = gaussian_kde([x_valid, y_valid])
        xi = np.linspace(x_valid.min(), x_valid.max(), 50)
        yi = np.linspace(y_valid.min(), y_valid.max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = kde(np.vstack([Xi.flatten(), Yi.flatten()]))
        ax1.contour(Xi, Yi, zi.reshape(Xi.shape), colors='gray', alpha=0.3, linewidths=1)

ax1.set_xlabel('Current Trophies', fontweight='bold')
ax1.set_ylabel('Highest Trophies', fontweight='bold')
ax1.set_title('Player Performance Clustering by Experience Level', fontweight='bold', fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Violin plot with box plots and swarm
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Create name color categories
color_mapping = {
    '0xffffffff': 'White',
    '0xffff8afb': 'Pink', 
    '0xfff9c908': 'Yellow',
    '0xffcb5aff': 'Purple',
    '0xff1ba5f5': 'Blue',
    '0xffa8e132': 'Green'
}

player_data['color_category'] = player_data['nameColor'].map(color_mapping).fillna('Other')
trophy_data = []
categories = []

for category in player_data['color_category'].unique():
    if category != 'Other' and pd.notna(category):
        data = player_data[player_data['color_category'] == category]['trophies'].dropna().values
        if len(data) > 5:  # Only include categories with sufficient data
            trophy_data.append(data)
            categories.append(category)

if trophy_data:
    parts = ax2.violinplot(trophy_data, positions=range(len(categories)), showmeans=True, showmedians=True)
    
    # Overlay box plots
    bp = ax2.boxplot(trophy_data, positions=range(len(categories)), widths=0.3, 
                     patch_artist=True, showfliers=False)
    
    # Color the plots
    violin_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    for i, (pc, box) in enumerate(zip(parts['bodies'], bp['boxes'])):
        pc.set_facecolor(violin_colors[i % len(violin_colors)])
        pc.set_alpha(0.7)
        box.set_facecolor(violin_colors[i % len(violin_colors)])
        box.set_alpha(0.5)

    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45)
else:
    ax2.text(0.5, 0.5, 'Insufficient data for violin plot', ha='center', va='center', transform=ax2.transAxes)

ax2.set_ylabel('Trophy Count', fontweight='bold')
ax2.set_title('Trophy Distribution by Player Name Color', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Subplot 3: Parallel coordinates plot
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Select top players and normalize data
top_players = player_data.nlargest(50, 'trophies')
metrics = ['trophies', 'expLevel', 'soloVictories', 'duoVictories', '3vs3Victories']
parallel_data = top_players[metrics].copy().fillna(0)

# Normalize to 0-1 scale
scaler = StandardScaler()
parallel_normalized = pd.DataFrame(scaler.fit_transform(parallel_data), columns=metrics)

# Plot parallel coordinates
for i in range(min(20, len(parallel_normalized))):  # Limit to 20 lines for clarity
    ax3.plot(range(len(metrics)), parallel_normalized.iloc[i], alpha=0.6, linewidth=1)

ax3.set_xticks(range(len(metrics)))
ax3.set_xticklabels(['Trophies', 'Exp Level', 'Solo Wins', 'Duo Wins', '3v3 Wins'], rotation=45)
ax3.set_ylabel('Normalized Values', fontweight='bold')
ax3.set_title('Top Player Performance Patterns', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Row 2: Club Ecosystem and Member Dynamics

# Subplot 4: Network-style scatter plot
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# Fix the merge issue by checking column existence
if 'type' in club_info.columns and 'requiredTrophies' in club_info.columns:
    club_data = club_rankings.merge(club_info[['tag', 'type', 'requiredTrophies']], on='tag', how='left')
else:
    club_data = club_rankings.copy()
    club_data['type'] = 'unknown'
    club_data['requiredTrophies'] = 0

club_types = club_data['type'].dropna().unique()
type_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

for i, club_type in enumerate(club_types):
    if pd.notna(club_type):
        mask = club_data['type'] == club_type
        if mask.sum() > 0:
            ax4.scatter(club_data[mask]['trophies'], club_data[mask]['memberCount'],
                       s=club_data[mask]['requiredTrophies']/100 + 20, c=type_colors[i % len(type_colors)],
                       alpha=0.7, label=club_type, edgecolors='white', linewidth=1)

# Add trend lines
for i, club_type in enumerate(club_types):
    if pd.notna(club_type):
        mask = club_data['type'] == club_type
        if mask.sum() > 1:
            x = club_data[mask]['trophies'].dropna()
            y = club_data[mask]['memberCount'].dropna()
            if len(x) > 1 and len(y) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax4.plot(x, p(x), color=type_colors[i % len(type_colors)], linestyle='--', alpha=0.8)

ax4.set_xlabel('Club Trophies', fontweight='bold')
ax4.set_ylabel('Member Count', fontweight='bold')
ax4.set_title('Club Ecosystem: Trophies vs Members by Type', fontweight='bold', fontsize=12)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Stacked bar chart with line overlay
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Calculate role distributions
if len(members_df) > 0:
    role_counts = members_df.groupby(['club_name', 'member_role']).size().unstack(fill_value=0)
    avg_trophies = members_df.groupby('club_name')['member_trophies'].mean()

    # Select top 15 clubs for visibility
    top_clubs = role_counts.sum(axis=1).nlargest(15).index
    role_counts_top = role_counts.loc[top_clubs]
    avg_trophies_top = avg_trophies.loc[top_clubs]

    # Create stacked bar chart
    bottom = np.zeros(len(role_counts_top))
    role_colors = {'president': '#E74C3C', 'vicePresident': '#F39C12', 'senior': '#3498DB', 'member': '#95A5A6'}

    for role in ['member', 'senior', 'vicePresident', 'president']:
        if role in role_counts_top.columns:
            ax5.bar(range(len(role_counts_top)), role_counts_top[role], bottom=bottom, 
                   label=role.title(), color=role_colors.get(role, '#BDC3C7'), alpha=0.8)
            bottom += role_counts_top[role]

    # Add line overlay for average trophies
    ax5_twin = ax5.twinx()
    ax5_twin.plot(range(len(avg_trophies_top)), avg_trophies_top.values, 
                  color='red', marker='o', linewidth=2, markersize=4, label='Avg Trophies')

    ax5.set_xticks(range(len(role_counts_top)))
    ax5.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in role_counts_top.index], 
                       rotation=45, ha='right')
    ax5.legend(loc='upper left', fontsize=8)
    ax5_twin.legend(loc='upper right', fontsize=8)
    ax5_twin.set_ylabel('Average Member Trophies', fontweight='bold', color='red')
else:
    ax5.text(0.5, 0.5, 'No member data available', ha='center', va='center', transform=ax5.transAxes)

ax5.set_xlabel('Top Clubs', fontweight='bold')
ax5.set_ylabel('Member Count by Role', fontweight='bold')
ax5.set_title('Club Member Role Distribution & Performance', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot 6: Correlation heatmap
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Create club characteristics matrix
try:
    club_metrics = club_data[['trophies', 'memberCount', 'requiredTrophies']].copy().fillna(0)
    
    if len(members_df) > 0:
        member_metrics = members_df.groupby('club_tag').agg({
            'member_trophies': ['mean', 'std', 'max']
        }).round(2)
        member_metrics.columns = ['avg_member_trophies', 'std_member_trophies', 'max_member_trophies']
        member_metrics = member_metrics.fillna(0)

        # Merge and calculate correlation
        club_analysis = club_metrics.merge(member_metrics, left_on='tag', right_index=True, how='inner')
    else:
        club_analysis = club_metrics

    if len(club_analysis) > 1:
        correlation_matrix = club_analysis.corr()

        # Create heatmap
        im = ax6.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(correlation_matrix.columns)))
        ax6.set_yticks(range(len(correlation_matrix.index)))
        ax6.set_xticklabels([col.replace('_', '\n') for col in correlation_matrix.columns], rotation=45, ha='right')
        ax6.set_yticklabels([col.replace('_', '\n') for col in correlation_matrix.index])

        # Add correlation values
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                ax6.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center', transform=ax6.transAxes)
        
except Exception as e:
    ax6.text(0.5, 0.5, f'Error creating heatmap: {str(e)[:50]}', ha='center', va='center', transform=ax6.transAxes)

ax6.set_title('Club Characteristics Correlation Matrix', fontweight='bold', fontsize=12)

# Row 3: Brawler Performance Segmentation

# Subplot 7: Grouped violin plots for brawler power levels
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Extract brawler data for power levels 9, 10, 11
brawler_cols = [col for col in player_info.columns if col.endswith('_power')]
power_data = {9: [], 10: [], 11: []}

for col in brawler_cols[:10]:  # Use first 10 brawlers for clarity
    trophy_col = col.replace('_power', '_trophies')
    if trophy_col in player_info.columns:
        for power_level in [9, 10, 11]:
            mask = player_info[col] == power_level
            if mask.sum() > 0:
                trophies = player_info[mask][trophy_col].dropna().values
                if len(trophies) > 0:
                    power_data[power_level].extend(trophies)

# Create violin plots if we have data
if any(len(data) > 0 for data in power_data.values()):
    positions = []
    violin_data = []
    labels = []
    
    for power_level in [9, 10, 11]:
        if len(power_data[power_level]) > 5:
            positions.append(len(violin_data))
            violin_data.append(power_data[power_level])
            labels.append(f'Power {power_level}')
    
    if violin_data:
        parts = ax7.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        ax7.set_xticks(positions)
        ax7.set_xticklabels(labels)
    else:
        ax7.text(0.5, 0.5, 'Insufficient data for violin plot', ha='center', va='center', transform=ax7.transAxes)
else:
    ax7.text(0.5, 0.5, 'No brawler power data available', ha='center', va='center', transform=ax7.transAxes)

ax7.set_ylabel('Trophy Count', fontweight='bold')
ax7.set_title('Brawler Trophy Distribution by Power Level', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3)

# Subplot 8: Multi-dimensional scatter matrix
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Select high-performing brawlers data
brawler_metrics = []
for col in brawler_cols[:5]:  # Top 5 brawlers
    rank_col = col.replace('_power', '_rank')
    trophy_col = col.replace('_power', '_trophies')
    highest_col = col.replace('_power', '_highestTrophies')
    
    if all(c in player_info.columns for c in [rank_col, trophy_col, highest_col]):
        mask = (player_info[col] >= 10) & (player_info[trophy_col] > 0)
        if mask.sum() > 0:
            brawler_name = col.replace('_power', '')
            for idx in player_info[mask].index:
                brawler_metrics.append([
                    player_info.loc[idx, rank_col],
                    player_info.loc[idx, trophy_col],
                    player_info.loc[idx, highest_col],
                    brawler_name
                ])

if brawler_metrics:
    brawler_df = pd.DataFrame(brawler_metrics, columns=['rank', 'trophies', 'highest', 'brawler'])
    
    # Create scatter plot with regression lines
    brawlers = brawler_df['brawler'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(brawlers)))
    
    for i, brawler in enumerate(brawlers):
        mask = brawler_df['brawler'] == brawler
        if mask.sum() > 0:
            ax8.scatter(brawler_df[mask]['trophies'], brawler_df[mask]['highest'], 
                       c=[colors[i]], alpha=0.6, label=brawler, s=30)
            
            # Add regression line
            if mask.sum() > 1:
                x = brawler_df[mask]['trophies'].values
                y = brawler_df[mask]['highest'].values
                if len(x) > 1 and not (np.isnan(x).all() or np.isnan(y).all()):
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax8.plot(x, p(x), color=colors[i], linestyle='--', alpha=0.8)

    ax8.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    ax8.text(0.5, 0.5, 'No high-performance brawler data', ha='center', va='center', transform=ax8.transAxes)

ax8.set_xlabel('Current Trophies', fontweight='bold')
ax8.set_ylabel('Highest Trophies', fontweight='bold')
ax8.set_title('High-Performance Brawler Analysis', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3)

# Subplot 9: Dendrogram with heatmap
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Create player performance patterns matrix
performance_metrics = ['trophies', 'highestTrophies', 'expLevel', '3vs3Victories', 'soloVictories', 'duoVictories']
top_players_perf = player_data.nlargest(30, 'trophies')[performance_metrics].fillna(0)

if len(top_players_perf) > 3:
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(top_players_perf)

    # Perform hierarchical clustering
    linkage_matrix = linkage(scaled_data, method='ward')

    # Create dendrogram
    dendro = dendrogram(linkage_matrix, ax=ax9, orientation='top', 
                       labels=None, leaf_rotation=90, leaf_font_size=8)
else:
    ax9.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center', transform=ax9.transAxes)

ax9.set_title('Player Performance Clustering Dendrogram', fontweight='bold', fontsize=12)
ax9.set_xlabel('Player Index', fontweight='bold')
ax9.set_ylabel('Distance', fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.savefig('brawl_stars_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()