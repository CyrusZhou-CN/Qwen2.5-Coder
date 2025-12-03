import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df = pd.read_csv('ufc.csv')

# Data preprocessing - remove rows with missing values in key columns
performance_cols = ['Fighter_1_KD', 'Fighter_2_KD', 'Fighter_1_STR', 'Fighter_2_STR', 
                   'Fighter_1_TD', 'Fighter_2_TD', 'Fighter_1_SUB', 'Fighter_2_SUB', 'Round', 'Winner']
df_clean = df.dropna(subset=performance_cols)

# Create figure with 2x1 subplot layout and increased spacing
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 15))
fig.patch.set_facecolor('white')

# Add main title centered over entire figure
fig.suptitle('UFC Fighter Performance Analysis: Strikes vs Outcomes and Metric Correlations', 
             fontsize=16, fontweight='bold', y=0.95)

# Top plot: Scatter plot of Fighter 1 STR vs Fighter 2 STR
# Create color mapping for winners with distinct colors
winner_color_map = {'Fighter 1': '#2E86AB', 'Fighter 2': '#A23B72', 'Draw': '#F18F01'}

# Create separate scatter plots for each winner category to ensure proper coloring
for winner, color in winner_color_map.items():
    mask = df_clean['Winner'] == winner
    if mask.sum() > 0:  # Only plot if there are data points for this winner
        ax1.scatter(df_clean.loc[mask, 'Fighter_1_STR'], 
                   df_clean.loc[mask, 'Fighter_2_STR'], 
                   c=color, s=df_clean.loc[mask, 'Round']*25, 
                   alpha=0.7, edgecolors='white', linewidth=0.5, 
                   label=winner)

# Add diagonal reference line
max_str = max(df_clean['Fighter_1_STR'].max(), df_clean['Fighter_2_STR'].max())
ax1.plot([0, max_str], [0, max_str], 'k--', alpha=0.5, linewidth=1.5, 
         label='Equal Performance Line')

# Styling for top plot
ax1.set_xlabel('Fighter 1 Significant Strikes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Fighter 2 Significant Strikes', fontsize=12, fontweight='bold')
ax1.set_title('Significant Strikes Comparison by Fight Outcome', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linewidth=0.5)
ax1.set_facecolor('white')

# Create comprehensive legend combining winner colors and size information
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Get unique winners that actually exist in the data
unique_winners = df_clean['Winner'].unique()
winner_elements = [Patch(facecolor=winner_color_map[winner], label=winner, alpha=0.7) 
                  for winner in unique_winners if winner in winner_color_map]

# Get unique rounds that actually exist in the data for size legend
unique_rounds = sorted(df_clean['Round'].unique())
size_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=np.sqrt(round_num*25)/2, alpha=0.7, 
                       label=f'Round {int(round_num)}') 
                for round_num in unique_rounds]

# Add diagonal line to legend
line_element = [Line2D([0], [0], color='black', linestyle='--', alpha=0.5, 
                      label='Equal Performance')]

# Combine all legend elements with proper spacing
all_elements = (winner_elements + 
               [Line2D([0], [0], color='w', label='Point Size:')] + 
               size_elements + 
               [Line2D([0], [0], color='w', label='')] + 
               line_element)

# Create legend
legend1 = ax1.legend(handles=all_elements, title='Winner & Round Size', 
                    loc='upper right', title_fontsize=11, fontsize=9, 
                    framealpha=0.95, ncol=1)
legend1.get_title().set_fontweight('bold')

# Bottom plot: Correlation heatmap
# Prepare correlation data
corr_cols = ['Fighter_1_KD', 'Fighter_2_KD', 'Fighter_1_STR', 'Fighter_2_STR', 
            'Fighter_1_TD', 'Fighter_2_TD', 'Fighter_1_SUB', 'Fighter_2_SUB']
corr_data = df_clean[corr_cols]

# Rename columns for better readability
corr_data.columns = ['F1_KD', 'F2_KD', 'F1_STR', 'F2_STR', 'F1_TD', 'F2_TD', 'F1_SUB', 'F2_SUB']

# Calculate correlation matrix
correlation_matrix = corr_data.corr()

# Create heatmap with diverging color scheme
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                     square=True, ax=ax2, cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                     fmt='.2f', annot_kws={'fontsize': 9})

# Styling for bottom plot
ax2.set_title('Performance Metrics Correlation Matrix\n(KD=Knockdowns, STR=Strikes, TD=Takedowns, SUB=Submissions)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
ax2.set_ylabel('Performance Metrics', fontsize=12, fontweight='bold')

# Improve label readability with smaller font and better rotation
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha='center', fontsize=9)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)

# Adjust colorbar label
cbar = ax2.collections[0].colorbar
cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

# Overall layout adjustments with increased spacing
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)

plt.show()