import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('GE_2024_Results.csv')

# Clean and convert data types efficiently
df['EVM Votes'] = pd.to_numeric(df['EVM Votes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['Postal Votes'] = pd.to_numeric(df['Postal Votes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
df['% of Votes'] = pd.to_numeric(df['% of Votes'], errors='coerce').fillna(0)

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('white')

# Define color palette for major parties
party_colors = {
    'Bharatiya Janata Party': '#FF6B35',
    'Indian National Congress': '#19398A',
    'Aam Aadmi Party': '#0072CE',
    'All India Trinamool Congress': '#20B2AA',
    'Dravida Munnetra Kazhagam': '#FF0000',
    'Samajwadi Party': '#FF2222',
    'Bahujan Samaj Party': '#22409A',
    'Independent': '#808080'
}

# Get top parties for consistent coloring
top_parties = df['Party'].value_counts().head(6).index.tolist()

# 1. Top-left: Party performance clusters
ax1 = plt.subplot(3, 3, 1)
party_summary = df.groupby('Party').agg({
    'Total Votes': 'sum',
    '% of Votes': 'mean'
}).reset_index()

# Sample top parties to avoid overcrowding
party_summary_top = party_summary[party_summary['Party'].isin(top_parties)]

for party in top_parties:
    party_data = party_summary_top[party_summary_top['Party'] == party]
    if not party_data.empty:
        color = party_colors.get(party, '#666666')
        ax1.scatter(party_data['% of Votes'], party_data['Total Votes'], 
                   c=color, s=150, alpha=0.8, label=party[:12])

ax1.set_xlabel('Average Vote Percentage', fontweight='bold')
ax1.set_ylabel('Total Votes (millions)', fontweight='bold')
ax1.set_title('Party Performance Clusters', fontweight='bold', fontsize=11)
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# Format y-axis to show millions
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# 2. Top-center: State-wise party dominance
ax2 = plt.subplot(3, 3, 2)
# Get top 8 states by number of constituencies
top_states = df['State'].value_counts().head(8).index
state_party = df[df['State'].isin(top_states)].groupby(['State', 'Party'])['Total Votes'].sum().unstack(fill_value=0)
state_party_subset = state_party[top_parties[:4]]  # Limit to top 4 parties

# Stacked bar chart
bottom = np.zeros(len(state_party_subset))
colors = [party_colors.get(party, '#666666') for party in state_party_subset.columns]

for i, party in enumerate(state_party_subset.columns):
    ax2.bar(range(len(state_party_subset)), state_party_subset[party], 
           bottom=bottom, label=party[:12], color=colors[i], alpha=0.8)
    bottom += state_party_subset[party]

ax2.set_xlabel('States', fontweight='bold')
ax2.set_ylabel('Total Votes (millions)', fontweight='bold')
ax2.set_title('State-wise Party Dominance', fontweight='bold', fontsize=11)
ax2.set_xticks(range(len(top_states)))
ax2.set_xticklabels([state[:8] for state in top_states], rotation=45, ha='right')
ax2.legend(fontsize=8, loc='upper right')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# 3. Top-right: Constituency competitiveness
ax3 = plt.subplot(3, 3, 3)
winner_data = df[df['Result'] == 'Won']['% of Votes'].dropna()
loser_data = df[df['Result'] == 'Lost']['% of Votes'].dropna()

# Sample data for performance
winner_sample = winner_data.sample(min(1000, len(winner_data)), random_state=42)
loser_sample = loser_data.sample(min(1000, len(loser_data)), random_state=42)

# Box plots instead of violin plots for better performance
bp1 = ax3.boxplot([winner_sample, loser_sample], positions=[1, 2], widths=0.6, patch_artist=True)
bp1['boxes'][0].set_facecolor('#4CAF50')
bp1['boxes'][1].set_facecolor('#FF5722')

ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Winners', 'Losers'])
ax3.set_ylabel('Vote Percentage', fontweight='bold')
ax3.set_title('Constituency Competitiveness', fontweight='bold', fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Middle-left: Vote distribution clusters
ax4 = plt.subplot(3, 3, 4)
# Sample data for performance
sample_size = min(2000, len(df))
df_sample = df.sample(sample_size, random_state=42)

evm_votes = df_sample['EVM Votes'].values
postal_votes = df_sample['Postal Votes'].values

# Remove extreme outliers
evm_q95 = np.percentile(evm_votes, 95)
postal_q95 = np.percentile(postal_votes, 95)
mask = (evm_votes <= evm_q95) & (postal_votes <= postal_q95) & (evm_votes > 0) & (postal_votes >= 0)

if np.sum(mask) > 10:
    h = ax4.hist2d(evm_votes[mask], postal_votes[mask], bins=20, cmap='Blues', alpha=0.8)
    plt.colorbar(h[3], ax=ax4, shrink=0.6)

ax4.set_xlabel('EVM Votes', fontweight='bold')
ax4.set_ylabel('Postal Votes', fontweight='bold')
ax4.set_title('Vote Distribution Patterns', fontweight='bold', fontsize=11)

# 5. Middle-center: Party correlation heatmap
ax5 = plt.subplot(3, 3, 5)
# Simplified correlation matrix
party_state_matrix = df.groupby(['Party', 'State']).size().unstack(fill_value=0)
top_parties_for_corr = df['Party'].value_counts().head(5).index
party_subset = party_state_matrix.loc[top_parties_for_corr]

# Calculate correlation
correlation_matrix = party_subset.T.corr()

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=ax5, cbar_kws={'shrink': 0.6})
ax5.set_title('Party Correlation Matrix', fontweight='bold', fontsize=11)
ax5.set_xticklabels([party[:8] for party in correlation_matrix.columns], rotation=45, ha='right')
ax5.set_yticklabels([party[:8] for party in correlation_matrix.index], rotation=0)

# 6. Middle-right: Victory margin analysis
ax6 = plt.subplot(3, 3, 6)
# Calculate victory margins efficiently
constituency_results = df.loc[df.groupby('Constituency')['% of Votes'].idxmax()]
constituency_runners_up = df.loc[df.groupby('Constituency')['% of Votes'].apply(lambda x: x.nlargest(2).index[-1] if len(x) >= 2 else x.idxmax())]

# Merge to get margins
margins_df = pd.merge(constituency_results[['Constituency', '% of Votes']], 
                     constituency_runners_up[['Constituency', '% of Votes']], 
                     on='Constituency', suffixes=('_winner', '_runner'))
margins_df['margin'] = margins_df['% of Votes_winner'] - margins_df['% of Votes_runner']

# Histogram of victory margins
margins = margins_df['margin'].dropna()
margins = margins[margins > 0]  # Valid margins only

ax6.hist(margins, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax6.axvline(margins.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {margins.mean():.1f}%')
ax6.axvline(margins.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {margins.median():.1f}%')

ax6.set_xlabel('Victory Margin (%)', fontweight='bold')
ax6.set_ylabel('Number of Constituencies', fontweight='bold')
ax6.set_title('Victory Margin Distribution', fontweight='bold', fontsize=11)
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Bottom-left: Regional voting behavior
ax7 = plt.subplot(3, 3, 7)
# Parallel coordinates plot
state_party_pct = df.groupby(['State', 'Party'])['% of Votes'].mean().unstack(fill_value=0)
top_5_parties = df['Party'].value_counts().head(4).index  # Reduced for clarity
selected_states = df['State'].value_counts().head(6).index

plot_data = state_party_pct.loc[selected_states, top_5_parties]

x_pos = range(len(top_5_parties))
colors = plt.cm.Set3(np.linspace(0, 1, len(selected_states)))

for i, state in enumerate(selected_states):
    ax7.plot(x_pos, plot_data.loc[state], 'o-', color=colors[i], 
            alpha=0.8, linewidth=2, markersize=5, label=state[:10])

ax7.set_xticks(x_pos)
ax7.set_xticklabels([party[:8] for party in top_5_parties], rotation=45, ha='right')
ax7.set_ylabel('Average Vote %', fontweight='bold')
ax7.set_title('Regional Voting Patterns', fontweight='bold', fontsize=11)
ax7.legend(fontsize=8, loc='upper right')
ax7.grid(True, alpha=0.3)

# 8. Bottom-center: Party vote share treemap simulation
ax8 = plt.subplot(3, 3, 8)
party_vote_share = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(6)

# Create pie chart as treemap alternative
colors = [party_colors.get(party, '#666666') for party in party_vote_share.index]
wedges, texts, autotexts = ax8.pie(party_vote_share.values, labels=[party[:10] for party in party_vote_share.index], 
                                  colors=colors, autopct='%1.1f%%', startangle=90)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax8.set_title('Party Vote Share Distribution', fontweight='bold', fontsize=11)

# 9. Bottom-right: Electoral competitiveness zones
ax9 = plt.subplot(3, 3, 9)
# Calculate competitiveness metrics
constituency_stats = df.groupby('Constituency').agg({
    '% of Votes': ['std', 'max', 'count'],
    'Total Votes': 'sum'
}).reset_index()

constituency_stats.columns = ['Constituency', 'vote_std', 'max_vote_pct', 'candidate_count', 'total_votes']
constituency_stats = constituency_stats.dropna()

# Create scatter plot with competitiveness zones
scatter = ax9.scatter(constituency_stats['vote_std'], constituency_stats['max_vote_pct'], 
                     c=constituency_stats['candidate_count'], cmap='viridis', 
                     alpha=0.6, s=30)

plt.colorbar(scatter, ax=ax9, shrink=0.6, label='Number of Candidates')

ax9.set_xlabel('Vote Std Deviation', fontweight='bold')
ax9.set_ylabel('Winner Vote Percentage', fontweight='bold')
ax9.set_title('Electoral Competitiveness Zones', fontweight='bold', fontsize=11)
ax9.grid(True, alpha=0.3)

# Overall title and layout
fig.suptitle('2024 Indian Lok Sabha Election: Comprehensive Clustering Analysis', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.subplots_adjust(hspace=0.35, wspace=0.35)
plt.savefig('lok_sabha_analysis.png', dpi=300, bbox_inches='tight')
plt.show()