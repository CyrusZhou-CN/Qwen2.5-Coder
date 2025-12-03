import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Load all datasets
engineering_df = pd.read_csv('EngineeringRanking.csv')
management_df = pd.read_csv('ManagementRanking.csv')
medical_df = pd.read_csv('MedicalRanking.csv')
overall_df = pd.read_csv('OverallRanking.csv')

# Set up the figure with white background
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define consistent color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']

# TOP-LEFT: Engineering rankings - TLR trends with scatter and error bars
years = [2016, 2017, 2018, 2019, 2020, 2021]
tlr_means = []
tlr_stds = []
all_tlr_data = []

for year in years:
    tlr_col = f'TLR_{str(year)[2:]}'
    if tlr_col in engineering_df.columns:
        tlr_values = engineering_df[tlr_col].dropna()
        if len(tlr_values) > 0:
            tlr_means.append(tlr_values.mean())
            tlr_stds.append(tlr_values.std())
            all_tlr_data.append(tlr_values.values)
        else:
            tlr_means.append(np.nan)
            tlr_stds.append(np.nan)
            all_tlr_data.append([])

# Plot line chart with error bars
ax1.errorbar(years, tlr_means, yerr=tlr_stds, color=colors[0], linewidth=3, 
             marker='o', markersize=8, capsize=5, capthick=2, label='Average TLR ± Std Dev')

# Add scatter plot for individual institutes
for i, year in enumerate(years):
    if len(all_tlr_data[i]) > 0:
        # Sample some points to avoid overcrowding
        sample_size = min(30, len(all_tlr_data[i]))
        sample_indices = np.random.choice(len(all_tlr_data[i]), sample_size, replace=False)
        sample_data = all_tlr_data[i][sample_indices]
        jitter = np.random.normal(0, 0.3, len(sample_data))
        ax1.scatter([year] * len(sample_data) + jitter, sample_data, 
                   alpha=0.4, color=colors[1], s=20, label='Individual Institutes' if i == 0 else "")

ax1.set_title('Engineering Rankings: TLR Score Evolution (2016-2021)', fontweight='bold', fontsize=14)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('TLR Score', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_facecolor('white')

# TOP-RIGHT: Management rankings - GO score distribution with median trend and institute count
go_data_by_year = []
go_medians = []
institute_counts = []

for year in years:
    go_col = f'GO_{str(year)[2:]}'
    if go_col in management_df.columns:
        go_values = management_df[go_col].dropna()
        if len(go_values) > 0:
            go_data_by_year.append(go_values.values)
            go_medians.append(go_values.median())
            institute_counts.append(len(go_values))
        else:
            go_data_by_year.append([])
            go_medians.append(np.nan)
            institute_counts.append(0)

# Create area chart for GO score distribution
for i, year in enumerate(years):
    if len(go_data_by_year[i]) > 0:
        # Create histogram data for area chart
        hist, bin_edges = np.histogram(go_data_by_year[i], bins=10, range=(0, 100))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Normalize histogram for area representation
        hist_norm = hist / max(hist) * 20  # Scale for visualization
        ax2.fill_between(bin_centers, year - hist_norm/2, year + hist_norm/2, 
                        alpha=0.6, color=colors[i % len(colors)])

# Plot median trend line
ax2_twin = ax2.twinx()
ax2.plot(go_medians, years, color='red', linewidth=3, marker='s', markersize=8, 
         label='Median GO Score')

# Plot institute count on secondary axis
ax2_twin.plot(years, institute_counts, color=colors[3], linewidth=2, marker='^', 
              markersize=6, label='Institute Count')
ax2_twin.set_ylabel('Number of Institutes', fontweight='bold', color=colors[3])
ax2_twin.tick_params(axis='y', labelcolor=colors[3])

ax2.set_title('Management Rankings: GO Score Distribution & Trends', fontweight='bold', fontsize=14)
ax2.set_xlabel('GO Score', fontweight='bold')
ax2.set_ylabel('Year', fontweight='bold')
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.set_facecolor('white')

# BOTTOM-LEFT: Medical rankings - Top 5 institutes component analysis
# Get top 5 institutes in 2021
top5_medical = medical_df.nsmallest(5, 'Rank_21')
components = ['TLR', 'RPC', 'GO', 'OI']

# Create stacked bar chart for 2021 composition
bottom = np.zeros(5)
for i, comp in enumerate(components):
    values = top5_medical[f'{comp}_21'].values
    ax3.bar(range(5), values, bottom=bottom, label=comp, color=colors[i], alpha=0.8)
    bottom += values

# Add line plots for component trends
for i, comp in enumerate(components):
    comp_trends = []
    for year in [2019, 2020, 2021]:  # Medical data available for these years
        comp_col = f'{comp}_{str(year)[2:]}'
        if comp_col in medical_df.columns:
            # Average of top 5 institutes for each year
            top5_ids = top5_medical['Institute Id'].values
            comp_values = medical_df[medical_df['Institute Id'].isin(top5_ids)][comp_col].mean()
            comp_trends.append(comp_values)
    
    if len(comp_trends) == 3:
        ax3_twin = ax3.twinx()
        ax3_twin.plot([2019, 2020, 2021], comp_trends, color=colors[i], 
                     linewidth=2, marker='o', alpha=0.7, linestyle='--')

ax3.set_title('Medical Rankings: Top 5 Institutes Score Composition (2021)', fontweight='bold', fontsize=14)
ax3.set_xlabel('Institute Rank', fontweight='bold')
ax3.set_ylabel('Score Components (Stacked)', fontweight='bold')
ax3.set_xticks(range(5))
ax3.set_xticklabels([f'#{i+1}' for i in range(5)])
ax3.legend(loc='upper right')
ax3.set_facecolor('white')

# BOTTOM-RIGHT: Overall rankings - Slope chart with correlation heatmap
# Get top 20 institutes with data for both 2017 and 2021
overall_filtered = overall_df.dropna(subset=['Rank_17', 'Rank_21'])
top20_overall = overall_filtered.nsmallest(20, 'Rank_21')

# Create correlation matrix as background heatmap
score_cols = ['TLR_21', 'RPC_21', 'GO_21', 'OI_21']
corr_data = overall_df[score_cols].corr()

# Plot heatmap as background
im = ax4.imshow(corr_data.values, cmap='RdYlBu_r', alpha=0.3, aspect='auto')

# Create slope chart
for idx, row in top20_overall.iterrows():
    rank_2017 = row['Rank_17']
    rank_2021 = row['Rank_21']
    
    # Color based on improvement/decline
    if rank_2021 < rank_2017:  # Improved (lower rank number is better)
        color = colors[4]  # Green for improvement
        alpha = 0.8
    elif rank_2021 > rank_2017:  # Declined
        color = colors[3]  # Red for decline
        alpha = 0.8
    else:
        color = 'gray'  # No change
        alpha = 0.5
    
    ax4.plot([0, 1], [rank_2017, rank_2021], color=color, alpha=alpha, linewidth=2)
    ax4.scatter([0, 1], [rank_2017, rank_2021], color=color, alpha=alpha, s=30)

# Customize slope chart
ax4.set_xlim(-0.1, 1.1)
ax4.set_ylim(25, 0)  # Inverted y-axis (rank 1 at top)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['2017', '2021'], fontweight='bold')
ax4.set_ylabel('Overall Rank', fontweight='bold')
ax4.set_title('Overall Rankings: Top 20 Institutes Rank Changes\n(Background: Score Correlation Matrix)', 
              fontweight='bold', fontsize=14)

# Add correlation matrix labels
ax4_corr = ax4.twinx()
ax4_corr.set_ylim(0, 4)
ax4_corr.set_yticks([0.5, 1.5, 2.5, 3.5])
ax4_corr.set_yticklabels(['TLR', 'RPC', 'GO', 'OI'], fontsize=10)
ax4_corr.tick_params(axis='y', length=0)

ax4.set_facecolor('white')

# Add colorbar for correlation matrix
cbar = plt.colorbar(im, ax=ax4, shrink=0.6, aspect=20)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# Add legend for slope chart
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[4], lw=2, label='Rank Improved'),
                   Line2D([0], [0], color=colors[3], lw=2, label='Rank Declined'),
                   Line2D([0], [0], color='gray', lw=2, label='No Change')]
ax4.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))

# Overall figure styling
plt.suptitle('Evolution of Indian Higher Education Institutions (2016-2021)', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Ensure all subplots have white background
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('white')

plt.show()