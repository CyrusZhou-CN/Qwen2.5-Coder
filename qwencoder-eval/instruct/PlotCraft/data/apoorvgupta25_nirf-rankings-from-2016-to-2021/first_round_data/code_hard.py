import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
datasets = {
    'Engineering': pd.read_csv('EngineeringRanking.csv'),
    'Medical': pd.read_csv('MedicalRanking.csv'),
    'Management': pd.read_csv('ManagementRanking.csv'),
    'Pharmacy': pd.read_csv('PharmacyRanking.csv'),
    'Architecture': pd.read_csv('ArchitectureRanking.csv'),
    'Law': pd.read_csv('LawRanking.csv'),
    'Dental': pd.read_csv('DentalRanking.csv'),
    'College': pd.read_csv('CollegeRanking.csv'),
    'University': pd.read_csv('UniversityRanking.csv'),
    'Overall': pd.read_csv('OverallRanking.csv')
}

# Create figure with white background
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01']
colors_categories = {
    'Engineering': '#2E86AB',
    'Medical': '#A23B72', 
    'Management': '#F18F01',
    'Pharmacy': '#52B788',
    'Architecture': '#E76F51',
    'Law': '#8E44AD',
    'Dental': '#F39C12',
    'College': '#16A085',
    'University': '#E74C3C',
    'Overall': '#34495E'
}
colors_states = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
colors_tiers = ['#1B4332', '#40916C', '#95D5B2', '#D8F3DC']

# Top-left: Multi-layered time series for top 3 categories
years = [2016, 2017, 2018, 2019, 2020, 2021]
top_categories = ['Engineering', 'Medical', 'Management']

for i, category in enumerate(top_categories):
    df = datasets[category]
    yearly_scores = []
    yearly_min = []
    yearly_max = []
    
    for year in years:
        score_col = f'Score_{str(year)[2:]}'
        if score_col in df.columns:
            scores = df[score_col].dropna()
            if len(scores) > 0:
                yearly_scores.append(scores.mean())
                yearly_min.append(scores.min())
                yearly_max.append(scores.max())
            else:
                yearly_scores.append(np.nan)
                yearly_min.append(np.nan)
                yearly_max.append(np.nan)
        else:
            yearly_scores.append(np.nan)
            yearly_min.append(np.nan)
            yearly_max.append(np.nan)
    
    # Remove NaN values for plotting
    valid_indices = ~np.isnan(yearly_scores)
    valid_years = np.array(years)[valid_indices]
    valid_scores = np.array(yearly_scores)[valid_indices]
    valid_min = np.array(yearly_min)[valid_indices]
    valid_max = np.array(yearly_max)[valid_indices]
    
    if len(valid_years) > 0:
        # Fill area between min and max with higher transparency
        ax1.fill_between(valid_years, valid_min, valid_max, alpha=0.15, color=colors_main[i])
        # Plot trend line
        ax1.plot(valid_years, valid_scores, marker='o', linewidth=3, markersize=8, 
                color=colors_main[i], label=f'{category} (Avg)', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor=colors_main[i])

ax1.set_title('Average Scores Evolution with Score Ranges\n(Top 3 Categories: 2016-2021)', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('NIRF Score', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='lightgray')
ax1.set_facecolor('white')

# Top-right: Slope chart with scatter points for rank changes 2020-2021 (ALL CATEGORIES)
combined_data = []
for category, df in datasets.items():
    if 'Rank_20' in df.columns and 'Rank_21' in df.columns:
        df_subset = df[['Institute Name', 'Rank_20', 'Rank_21', 'Score_20', 'Score_21']].dropna()
        # Filter for top 20 institutes (either 2020 or 2021 rank <= 20)
        df_subset = df_subset[(df_subset['Rank_20'] <= 20) | (df_subset['Rank_21'] <= 20)]
        df_subset['Category'] = category
        df_subset['Score_Change'] = abs(df_subset['Score_21'] - df_subset['Score_20'])
        combined_data.append(df_subset)

if combined_data:
    slope_df = pd.concat(combined_data, ignore_index=True)
    # Sort by 2021 rank and take top 20
    slope_df = slope_df.sort_values('Rank_21').head(20)
    
    for idx, row in slope_df.iterrows():
        # Line thickness based on score change magnitude
        line_width = max(1, min(8, row['Score_Change'] * 2))
        color = colors_categories[row['Category']]
        
        # Draw slope line
        ax2.plot([2020, 2021], [row['Rank_20'], row['Rank_21']], 
                color=color, linewidth=line_width, alpha=0.7)
        
        # Scatter points
        ax2.scatter(2020, row['Rank_20'], color=color, s=80, alpha=0.8, edgecolors='white', linewidth=2)
        ax2.scatter(2021, row['Rank_21'], color=color, s=80, alpha=0.8, edgecolors='white', linewidth=2)

ax2.set_title('Rank Changes Between 2020-2021\n(Top 20 Institutes, Line Thickness = Score Change)', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('NIRF Rank', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.set_xticks([2020, 2021])
ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='lightgray')
ax2.set_facecolor('white')

# Create legend for categories present in the data
present_categories = slope_df['Category'].unique()
legend_elements = [plt.Line2D([0], [0], color=colors_categories[cat], lw=3, label=cat) 
                  for cat in present_categories]
ax2.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

# Bottom-left: Stacked area chart by state with performance trends
state_participation = {}
state_performance = {}

for year in years:
    year_str = str(year)[2:]
    state_counts = {}
    state_scores = {}
    
    for category, df in datasets.items():
        if f'Score_{year_str}' in df.columns:
            year_data = df[['State', f'Score_{year_str}']].dropna()
            for state in year_data['State'].unique():
                state_counts[state] = state_counts.get(state, 0) + len(year_data[year_data['State'] == state])
                state_scores[state] = year_data[year_data['State'] == state][f'Score_{year_str}'].mean()
    
    state_participation[year] = state_counts
    state_performance[year] = state_scores

# Get top 5 states by total participation
all_states = set()
for year_data in state_participation.values():
    all_states.update(year_data.keys())

state_totals = {state: sum(state_participation[year].get(state, 0) for year in years) for state in all_states}
top_states = sorted(state_totals.items(), key=lambda x: x[1], reverse=True)[:5]
top_state_names = [state[0] for state in top_states]

# Create stacked area chart
bottom = np.zeros(len(years))
for i, state in enumerate(top_state_names):
    values = [state_participation[year].get(state, 0) for year in years]
    ax3.fill_between(years, bottom, bottom + values, alpha=0.7, 
                    color=colors_states[i], label=f'{state} (Count)')
    bottom += values

# Overlay performance trend lines with brighter colors
ax3_twin = ax3.twinx()
bright_colors = ['#2D5016', '#1F6B5C', '#B8860B', '#CC5500', '#B22222']  # Brighter versions
for i, state in enumerate(top_state_names):
    perf_values = [state_performance[year].get(state, np.nan) for year in years]
    valid_indices = ~np.isnan(perf_values)
    if np.any(valid_indices):
        valid_years_perf = np.array(years)[valid_indices]
        valid_perf = np.array(perf_values)[valid_indices]
        ax3_twin.plot(valid_years_perf, valid_perf, marker='s', linewidth=2, 
                     markersize=6, color=bright_colors[i], linestyle='--', alpha=0.9)

ax3.set_title('State Participation & Performance Trends\n(Institute Count + Average Score Trends)', 
              fontsize=16, fontweight='bold', pad=20)
ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Institutes', fontsize=12, fontweight='bold')
ax3_twin.set_ylabel('Average Score', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='lightgray')
ax3.set_facecolor('white')

# Bottom-right: Ranking tiers with TLR and RPC trends
tier_counts = {year: {'1-10': 0, '11-25': 0, '26-50': 0, '51-100': 0} for year in years}
tlr_scores = {year: [] for year in years}
rpc_scores = {year: [] for year in years}

for year in years:
    year_str = str(year)[2:]
    for category, df in datasets.items():
        if f'Rank_{year_str}' in df.columns:
            ranks = df[f'Rank_{year_str}'].dropna()
            for rank in ranks:
                try:
                    rank_val = float(rank)
                    if 1 <= rank_val <= 10:
                        tier_counts[year]['1-10'] += 1
                    elif 11 <= rank_val <= 25:
                        tier_counts[year]['11-25'] += 1
                    elif 26 <= rank_val <= 50:
                        tier_counts[year]['26-50'] += 1
                    elif 51 <= rank_val <= 100:
                        tier_counts[year]['51-100'] += 1
                except:
                    continue
        
        # Collect TLR and RPC scores
        if f'TLR_{year_str}' in df.columns:
            tlr_scores[year].extend(df[f'TLR_{year_str}'].dropna().tolist())
        if f'RPC_{year_str}' in df.columns:
            rpc_scores[year].extend(df[f'RPC_{year_str}'].dropna().tolist())

# Plot grouped bar chart
x_pos = np.arange(len(years))
width = 0.2
tiers = ['1-10', '11-25', '26-50', '51-100']

for i, tier in enumerate(tiers):
    values = [tier_counts[year][tier] for year in years]
    ax4.bar(x_pos + i * width, values, width, label=f'Rank {tier}', 
           color=colors_tiers[i], alpha=0.8, edgecolor='white', linewidth=1)

# Overlay TLR and RPC trend lines
ax4_twin = ax4.twinx()
tlr_avg = [np.mean(tlr_scores[year]) if tlr_scores[year] else np.nan for year in years]
rpc_avg = [np.mean(rpc_scores[year]) if rpc_scores[year] else np.nan for year in years]

# Calculate percentage changes
tlr_pct_change = [0] + [((tlr_avg[i] - tlr_avg[i-1]) / tlr_avg[i-1] * 100) 
                       if not np.isnan(tlr_avg[i]) and not np.isnan(tlr_avg[i-1]) and tlr_avg[i-1] != 0 
                       else 0 for i in range(1, len(tlr_avg))]
rpc_pct_change = [0] + [((rpc_avg[i] - rpc_avg[i-1]) / rpc_avg[i-1] * 100) 
                       if not np.isnan(rpc_avg[i]) and not np.isnan(rpc_avg[i-1]) and rpc_avg[i-1] != 0 
                       else 0 for i in range(1, len(rpc_avg))]

ax4_twin.plot(x_pos, tlr_pct_change, marker='o', linewidth=3, markersize=8, 
             color='#E63946', label='TLR % Change', markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor='#E63946')
ax4_twin.plot(x_pos, rpc_pct_change, marker='s', linewidth=3, markersize=8, 
             color='#457B9D', label='RPC % Change', markerfacecolor='#457B9D', 
             markeredgewidth=2, markeredgecolor='#457B9D')

ax4.set_title('Ranking Tiers Distribution & Score Changes\n(Institute Counts + TLR/RPC % Change)', 
              fontsize=16, fontweight='bold', pad=20)
ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Institutes', fontsize=12, fontweight='bold')
ax4_twin.set_ylabel('Percentage Change (%)', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos + width * 1.5)
ax4.set_xticklabels(years)
ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax4_twin.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='lightgray')
ax4.set_facecolor('white')

# Final layout adjustment
plt.tight_layout(pad=3.0)
plt.show()