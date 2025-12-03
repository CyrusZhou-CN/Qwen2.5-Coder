import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('games_by_players.csv')

# Data preprocessing - sample data for performance
np.random.seed(42)
sample_size = 10000
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

# Remove rows with missing values in key columns
key_columns = ['core_score', 'winner', 'boost_avg_amount', 'positioning_percent_offensive_third', 
               'core_goals', 'core_shots', 'core_saves', 'boost_bpm', 'movement_time_supersonic_speed',
               'positioning_time_offensive_third', 'core_shooting_percentage', 'advanced_rating', 'demo_inflicted']
df_clean = df_sample.dropna(subset=key_columns)

# Further reduce for visualization performance
df_viz = df_clean.sample(n=min(3000, len(df_clean)), random_state=42)

# Create figure with 2x2 subplots with more spacing
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Create custom grid layout for better spacing
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, top=0.92, bottom=0.08, left=0.08, right=0.95)

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Top-left: Scatter plot with marginal histogram (improved design)
ax1 = fig.add_subplot(gs[0, 0])

# Create bins for better visualization
score_bins = np.linspace(df_viz['core_score'].min(), df_viz['core_score'].max(), 30)
win_rates = []
bin_centers = []

for i in range(len(score_bins)-1):
    mask = (df_viz['core_score'] >= score_bins[i]) & (df_viz['core_score'] < score_bins[i+1])
    if mask.sum() > 0:
        win_rate = df_viz[mask]['winner'].mean()
        win_rates.append(win_rate)
        bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)

# Plot win rate vs score
ax1.scatter(bin_centers, win_rates, s=80, color=colors[0], alpha=0.8, edgecolors='white', linewidth=1.5)

# Add regression line
if len(bin_centers) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(bin_centers, win_rates)
    line_x = np.array(bin_centers)
    line_y = slope * line_x + intercept
    ax1.plot(line_x, line_y, color='#333333', linewidth=3, linestyle='--', alpha=0.9)

ax1.set_xlabel('Core Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
ax1.set_title('Core Score vs Win Rate', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

if len(bin_centers) > 1:
    ax1.text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11, fontweight='bold')

# Add marginal histogram at the top
ax1_hist = fig.add_axes([ax1.get_position().x0, ax1.get_position().y1 + 0.02, 
                         ax1.get_position().width, 0.08])
ax1_hist.hist(df_viz['core_score'], bins=25, alpha=0.7, color='#666666', density=True)
ax1_hist.set_xlim(ax1.get_xlim())
ax1_hist.set_xticks([])
ax1_hist.set_ylabel('Density', fontsize=9)
ax1_hist.spines['top'].set_visible(False)
ax1_hist.spines['right'].set_visible(False)
ax1_hist.spines['bottom'].set_visible(False)

# Top-right: Bubble plot
ax2 = fig.add_subplot(gs[0, 1])

bubble_sample = df_viz.sample(n=min(1000, len(df_viz)), random_state=42)

for winner_status in [True, False]:
    subset = bubble_sample[bubble_sample['winner'] == winner_status]
    color = colors[0] if winner_status else colors[1]
    label = 'Winners' if winner_status else 'Losers'
    
    # Normalize bubble sizes
    bubble_sizes = np.clip(subset['core_goals'] * 20 + 15, 15, 250)
    
    ax2.scatter(subset['boost_avg_amount'], subset['positioning_percent_offensive_third'],
               s=bubble_sizes, alpha=0.6, color=color, label=label, 
               edgecolors='white', linewidth=0.8)

ax2.set_xlabel('Average Boost Amount', fontsize=12, fontweight='bold')
ax2.set_ylabel('Offensive Third Position %', fontsize=12, fontweight='bold')
ax2.set_title('Boost vs Positioning (Bubble Size = Goals)', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Bottom-left: Correlation heatmap
ax3 = fig.add_subplot(gs[1, 0])

corr_metrics = ['core_shots', 'core_goals', 'core_saves', 'boost_bpm', 
                'movement_time_supersonic_speed', 'positioning_time_offensive_third']
corr_data = df_viz[corr_metrics].select_dtypes(include=[np.number])
corr_matrix = corr_data.corr()

# Create heatmap
im = ax3.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values as text
for i in range(len(corr_metrics)):
    for j in range(len(corr_metrics)):
        if not np.isnan(corr_matrix.iloc[i, j]):
            text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
            ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha="center", va="center", color=text_color, 
                    fontweight='bold', fontsize=10)

# Format labels
short_labels = ['Shots', 'Goals', 'Saves', 'Boost BPM', 'Supersonic Time', 'Offensive Time']
ax3.set_xticks(range(len(corr_metrics)))
ax3.set_yticks(range(len(corr_metrics)))
ax3.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
ax3.set_yticklabels(short_labels, fontsize=10)
ax3.set_title('Performance Metrics Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('Correlation', fontsize=11, fontweight='bold')

# Bottom-right: Scatter plot matrix (pair plot) - CORRECTED
ax4 = fig.add_subplot(gs[1, 1])

# Prepare data for pair plot
pair_metrics = ['core_shooting_percentage', 'advanced_rating', 'demo_inflicted']
pair_sample = df_viz[(df_viz['core_shooting_percentage'] <= 100) & 
                     (df_viz['core_shooting_percentage'] >= 0)].sample(n=min(800, len(df_viz)), random_state=42)

# Create a proper pair plot using seaborn
pair_data = pair_sample[pair_metrics + ['winner']].copy()
pair_data['Team Result'] = pair_data['winner'].map({True: 'Winners', False: 'Losers'})

# Remove the subplot ax4 and create seaborn pair plot
ax4.remove()

# Create pair plot in the bottom-right position
pair_ax = plt.subplot2grid((2, 2), (1, 1), fig=fig)
pair_ax.remove()

# Use seaborn's PairGrid for more control
g = sns.PairGrid(pair_data, vars=pair_metrics, hue='Team Result', 
                 palette=[colors[0], colors[1]], height=2.5, aspect=1)

# Map different plot types
g.map_diag(sns.histplot, alpha=0.7)
g.map_upper(sns.scatterplot, alpha=0.6, s=40)
g.map_lower(sns.scatterplot, alpha=0.6, s=40)

# Add regression lines
g.map_lower(sns.regplot, scatter=False, truncate=False)
g.map_upper(sns.regplot, scatter=False, truncate=False)

# Position the pair plot in the bottom-right
g.fig.set_size_inches(6, 6)
for ax in g.axes.flat:
    ax.tick_params(labelsize=9)
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, fontweight='bold')

# Adjust labels for better readability
label_map = {'core_shooting_percentage': 'Shooting %', 
             'advanced_rating': 'Rating', 
             'demo_inflicted': 'Demos'}

for ax in g.axes.flat:
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel in label_map:
        ax.set_xlabel(label_map[xlabel], fontsize=10, fontweight='bold')
    if ylabel in label_map:
        ax.set_ylabel(label_map[ylabel], fontsize=10, fontweight='bold')

g.add_legend(title='Team Result', bbox_to_anchor=(1.05, 0.5), loc='center left')

# Position the pair plot correctly
pos = gs[1, 1].get_position(fig)
g.fig.subplots_adjust(left=pos.x0, bottom=pos.y0, right=pos.x1, top=pos.y1)

# Add title for pair plot
fig.text(pos.x0 + pos.width/2, pos.y1 + 0.02, 'Performance Metrics Pair Plot', 
         ha='center', fontsize=14, fontweight='bold')

# Add shortened main title with proper positioning
fig.suptitle('RLCS Player Performance vs Team Success', 
             fontsize=18, fontweight='bold', y=0.97)

plt.savefig('rocket_league_correlation_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()