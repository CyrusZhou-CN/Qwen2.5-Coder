import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the datasets
bowling_df = pd.read_csv('bowling_card.csv')
summary_df = pd.read_csv('summary.csv')

# Merge bowling data with match results to determine win/loss
# First, let's create a mapping of match_id to winner
match_results = summary_df[['id', 'winner']].copy()
match_results.columns = ['match_id', 'winner']

# Merge bowling data with match results
merged_df = bowling_df.merge(match_results, on='match_id', how='left')

# Determine if bowling team won the match
merged_df['team_won'] = merged_df['bowling_team'] == merged_df['winner']
merged_df['team_won'] = merged_df['team_won'].astype(int)

# Top plot: Scatter plot of economy rate vs wickets with color coding for win/loss
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Calculate correlation coefficient
correlation = merged_df['economyRate'].corr(merged_df['wickets'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['economyRate'], merged_df['wickets'])

# Scatter plot with best fit line
scatter = ax1.scatter(merged_df['economyRate'], merged_df['wickets'], 
                     c=merged_df['team_won'], cmap='coolwarm', alpha=0.7, s=60)

# Add best fit line
x_line = np.linspace(merged_df['economyRate'].min(), merged_df['economyRate'].max(), 100)
y_line = slope * x_line + intercept
ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit line (r={r_value:.2f})')

# Customize top plot
ax1.set_xlabel('Economy Rate')
ax1.set_ylabel('Wickets Taken')
ax1.set_title('Correlation Between Economy Rate and Wickets Taken\n(Color-coded by Match Outcome)')
ax1.legend()

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Match Outcome (0=Loss, 1=Win)')

# Add annotation with correlation info
ax1.annotate(f'Correlation: {r_value:.3f}\nP-value: {p_value:.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, verticalalignment='top')

# Bottom plot: Bubble chart of runs conceded vs wickets taken
# Group by player and match to get aggregate stats
agg_df = merged_df.groupby(['match_id', 'bowling_team', 'name', 'fullName']).agg({
    'conceded': 'sum',
    'wickets': 'sum',
    'overs': 'sum'
}).reset_index()

# Create bubble chart
bubble_scatter = ax2.scatter(agg_df['conceded'], agg_df['wickets'], 
                           s=agg_df['overs']*10,  # Scale overs to bubble size
                           c=pd.factorize(agg_df['bowling_team'])[0], 
                           alpha=0.6, edgecolors='black', linewidth=0.5)

# Customize bottom plot
ax2.set_xlabel('Runs Conceded')
ax2.set_ylabel('Wickets Taken')
ax2.set_title('Relationship Between Runs Conceded and Wickets Taken\n(Bubble Size = Overs Bowled, Color = Team)')

# Add legend for teams
team_names = agg_df['bowling_team'].unique()
team_colors = [plt.cm.Set3(i) for i in range(len(team_names))]
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=8, label=team) for team, color in zip(team_names, team_colors)]

ax2.legend(handles=legend_elements, title="Teams", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid for better readability
ax2.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Print some summary statistics
print("Summary Statistics:")
print(f"Total bowling records: {len(merged_df)}")
print(f"Average economy rate: {merged_df['economyRate'].mean():.2f}")
print(f"Average wickets per bowler: {merged_df['wickets'].mean():.2f}")
print(f"Average runs conceded: {merged_df['conceded'].mean():.2f}")
print(f"Correlation between economy rate and wickets: {correlation:.3f}")