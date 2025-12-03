import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats

# Load all datasets
dental_df = pd.read_csv('DentalRanking.csv')
medical_df = pd.read_csv('MedicalRanking.csv')
research_df = pd.read_csv('ResearchRanking.csv')
university_df = pd.read_csv('UniversityRanking.csv')
overall_df = pd.read_csv('OverallRanking.csv')
engineering_df = pd.read_csv('EngineeringRanking.csv')
management_df = pd.read_csv('ManagementRanking.csv')

# Create figure with 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.patch.set_facecolor('white')

# Color palettes
colors_primary = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F7B0F']
colors_secondary = ['#87CEEB', '#DDA0DD', '#FFE4B5', '#FFA07A', '#D8BFD8', '#98FB98']

# 1. Top-left: Engineering rankings - Multi-line time series with filled area
ax1 = axes[0, 0]

# Get top 5 consistently ranked engineering institutes
years = [2016, 2017, 2018, 2019, 2020, 2021]
score_cols = ['Score_16', 'Score_17', 'Score_18', 'Score_19', 'Score_20', 'Score_21']

# Find institutes with data for all years
complete_data = engineering_df.dropna(subset=score_cols)
top5_institutes = complete_data.nsmallest(5, 'Rank_21')

# Plot individual trajectories
for i, (_, institute) in enumerate(top5_institutes.iterrows()):
    scores = [institute[col] for col in score_cols]
    ax1.plot(years, scores, marker='o', linewidth=2.5, 
             color=colors_primary[i], label=institute['Institute Name'][:30] + '...')

# Add filled area showing score range
all_scores_by_year = []
for col in score_cols:
    year_scores = engineering_df[col].dropna()
    all_scores_by_year.append([year_scores.min(), year_scores.max()])

min_scores = [scores[0] for scores in all_scores_by_year]
max_scores = [scores[1] for scores in all_scores_by_year]

ax1.fill_between(years, min_scores, max_scores, alpha=0.2, color='gray', 
                 label='Score Range (All Institutes)')

ax1.set_title('**Engineering Rankings: Top 5 Institute Trajectories**', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('NIRF Score', fontsize=12)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# 2. Top-right: Management rankings - Slope chart
ax2 = axes[0, 1]

# Get top 10 management institutes with both 2016 and 2021 data
mgmt_complete = management_df.dropna(subset=['Score_16', 'Score_21', 'TLR_21']).head(10)

for i, (_, institute) in enumerate(mgmt_complete.iterrows()):
    score_2016 = institute['Score_16']
    score_2021 = institute['Score_21']
    tlr_2021 = institute['TLR_21']
    rank_2021 = institute['Rank_21']
    
    # Line connecting 2016 to 2021
    ax2.plot([2016, 2021], [score_2016, score_2021], 
             color=colors_primary[int(rank_2021-1) % len(colors_primary)], 
             linewidth=2, alpha=0.7)
    
    # Scatter points sized by TLR
    ax2.scatter(2016, score_2016, s=tlr_2021*3, alpha=0.6, 
                color=colors_primary[int(rank_2021-1) % len(colors_primary)])
    ax2.scatter(2021, score_2021, s=tlr_2021*3, alpha=0.8, 
                color=colors_primary[int(rank_2021-1) % len(colors_primary)])

ax2.set_title('**Management Rankings: 2016-2021 Score Evolution**', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('NIRF Score', fontsize=12)
ax2.set_xticks([2016, 2021])
ax2.grid(True, alpha=0.3)

# 3. Middle-left: Medical vs Dental comparison
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()

# Medical data - top 10 average scores with confidence bands
medical_years = [2019, 2020, 2021]
medical_score_cols = ['Score_19', 'Score_20', 'Score_21']

medical_avg_scores = []
medical_std_scores = []
for col in medical_score_cols:
    top10_scores = medical_df.nsmallest(10, 'Rank_21')[col].dropna()
    medical_avg_scores.append(top10_scores.mean())
    medical_std_scores.append(top10_scores.std())

# Plot medical line with confidence bands
ax3.plot(medical_years, medical_avg_scores, color='#2E86AB', linewidth=3, 
         marker='o', markersize=8, label='Medical (Top 10 Avg)')
ax3.fill_between(medical_years, 
                 np.array(medical_avg_scores) - np.array(medical_std_scores),
                 np.array(medical_avg_scores) + np.array(medical_std_scores),
                 alpha=0.3, color='#2E86AB')

# Dental participation count
dental_years = [2020, 2021]
dental_counts = [dental_df['Score_20'].notna().sum(), dental_df['Score_21'].notna().sum()]

ax3_twin.bar(dental_years, dental_counts, alpha=0.6, color='#F18F01', 
             width=0.5, label='Dental Institutes Count')

ax3.set_title('**Medical vs Dental: Performance & Participation**', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Medical Score (Top 10 Average)', fontsize=12, color='#2E86AB')
ax3_twin.set_ylabel('Dental Institutes Count', fontsize=12, color='#F18F01')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 4. Middle-right: University category analysis - Stacked area chart
ax4 = axes[1, 1]

# Define score ranges
score_ranges = [(0, 40), (40, 60), (60, 80), (80, 100)]
range_labels = ['0-40', '40-60', '60-80', '80-100']
univ_years = [2016, 2017, 2018, 2019, 2020, 2021]
univ_score_cols = ['Score_16', 'Score_17', 'Score_18', 'Score_19', 'Score_20', 'Score_21']

# Calculate distributions for each year
distributions = []
medians = []

for col in univ_score_cols:
    year_scores = university_df[col].dropna()
    year_dist = []
    for low, high in score_ranges:
        count = ((year_scores >= low) & (year_scores < high)).sum()
        year_dist.append(count)
    distributions.append(year_dist)
    medians.append(year_scores.median())

distributions = np.array(distributions).T

# Create stacked area chart
ax4.stackplot(univ_years, *distributions, labels=range_labels, 
              colors=colors_secondary[:4], alpha=0.8)

# Add median trend line
ax4_twin2 = ax4.twinx()
ax4_twin2.plot(univ_years, medians, color='red', linewidth=3, 
               marker='s', markersize=6, label='Median Score')

ax4.set_title('**University Score Distribution Evolution**', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Number of Universities', fontsize=12)
ax4_twin2.set_ylabel('Median Score', fontsize=12, color='red')
ax4.legend(loc='upper left', fontsize=10)
ax4_twin2.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# 5. Bottom-left: Research institute performance - Box plots with violin plots
ax5 = axes[2, 0]

# Research data (only 2021 available)
research_scores = research_df['Score'].values
positions = [1]

# Create violin plot
violin_parts = ax5.violinplot([research_scores], positions=positions, widths=0.6, 
                              showmeans=True, showmedians=True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('#A23B72')
    pc.set_alpha(0.7)

# Overlay box plot
box_parts = ax5.boxplot([research_scores], positions=positions, widths=0.3, 
                        patch_artist=True, showfliers=False)
box_parts['boxes'][0].set_facecolor('#592E83')
box_parts['boxes'][0].set_alpha(0.8)

ax5.set_title('**Research Institute Score Distribution (2021)**', fontsize=14, fontweight='bold', pad=20)
ax5.set_ylabel('NIRF Score', fontsize=12)
ax5.set_xticks([1])
ax5.set_xticklabels(['Research Institutes'])
ax5.grid(True, alpha=0.3)

# 6. Bottom-right: Overall ranking stability - Heatmap with line plots
ax6 = axes[2, 1]

# Get top 20 overall institutes with complete ranking data
overall_years = [2017, 2018, 2019, 2020, 2021]
overall_rank_cols = ['Rank_17', 'Rank_18', 'Rank_19', 'Rank_20', 'Rank_21']

top20_complete = overall_df.dropna(subset=overall_rank_cols).head(20)
rank_matrix = top20_complete[overall_rank_cols].values

# Create heatmap
im = ax6.imshow(rank_matrix, cmap='RdYlBu_r', aspect='auto', alpha=0.8)
ax6.set_xticks(range(len(overall_years)))
ax6.set_xticklabels(overall_years)
ax6.set_yticks(range(len(top20_complete)))
ax6.set_yticklabels([name[:20] + '...' for name in top20_complete['Institute Name']], fontsize=8)

# Add coefficient of variation lines for different parameters
ax6_twin3 = ax6.twinx()
param_cols = ['TLR', 'RPC', 'GO', 'OI', 'Perception']
param_years = [2017, 2018, 2019, 2020, 2021]

for i, param in enumerate(param_cols):
    cv_values = []
    for year in param_years:
        col_name = f'{param}_{year}'
        if col_name in overall_df.columns:
            values = overall_df[col_name].dropna()
            cv = values.std() / values.mean() if values.mean() != 0 else 0
            cv_values.append(cv)
        else:
            cv_values.append(np.nan)
    
    ax6_twin3.plot(range(len(param_years)), cv_values, 
                   color=colors_primary[i], linewidth=2, marker='o', 
                   label=f'{param} CV', alpha=0.7)

ax6.set_title('**Overall Ranking Stability & Parameter Variation**', fontsize=14, fontweight='bold', pad=20)
ax6.set_xlabel('Year', fontsize=12)
ax6.set_ylabel('Top 20 Institutes', fontsize=12)
ax6_twin3.set_ylabel('Coefficient of Variation', fontsize=12)
ax6_twin3.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)

# Add colorbar for heatmap
cbar = plt.colorbar(im, ax=ax6, shrink=0.6)
cbar.set_label('Rank Position', fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.show()