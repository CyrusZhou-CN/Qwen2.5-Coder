import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('piracy.csv', delimiter=';')

# Clean the data - remove any completely empty rows
df = df.dropna(how='all')

# Get the actual column names from the first row if needed
if df.columns[0].startswith('name;party'):
    # The header is in the first column, need to split
    header_row = df.columns[0].split(';')
    df.columns = header_row
    
# If columns are still combined, split them
if len(df.columns) == 1:
    # Split the single column into multiple columns
    df_split = df.iloc[:, 0].str.split(';', expand=True)
    df_split.columns = ['name', 'party', 'state', 'money_pro', 'money_con', 'years', 'stance', 'chamber', 'house']
    df = df_split

# Ensure we have the right column names
expected_cols = ['name', 'party', 'state', 'money_pro', 'money_con', 'years', 'stance', 'chamber', 'house']
if list(df.columns) != expected_cols:
    df.columns = expected_cols

# Remove any rows that are actually headers
df = df[df['party'].isin(['D', 'R', 'I'])]

# Clean and convert data types
df = df.dropna(subset=['party', 'chamber', 'stance'])

# Convert numeric columns, handling 'NA' strings
def safe_numeric_convert(series):
    return pd.to_numeric(series.replace('NA', np.nan), errors='coerce').fillna(0)

df['money_pro'] = safe_numeric_convert(df['money_pro'])
df['money_con'] = safe_numeric_convert(df['money_con'])
df['years'] = safe_numeric_convert(df['years'])

# Create derived columns
df['total_money'] = df['money_pro'] + df['money_con']
df['money_ratio'] = np.where(df['money_con'] > 0, df['money_pro'] / df['money_con'], df['money_pro'])

# Set up the figure
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes
party_colors = {'D': '#1f77b4', 'R': '#d62728', 'I': '#2ca02c'}
chamber_colors = {'house': '#ff7f0e', 'senate': '#9467bd'}
stance_colors = {'yes': '#2ca02c', 'no': '#d62728', 'unknown': '#7f7f7f'}

# Row 1, Subplot 1: Stacked bar chart with line plot overlay
ax1 = plt.subplot(3, 3, 1)
try:
    stance_party = pd.crosstab(df['party'], df['stance'])
    stance_party.plot(kind='bar', stacked=True, ax=ax1, 
                     color=[stance_colors.get(col, '#7f7f7f') for col in stance_party.columns])
    
    ax1_twin = ax1.twinx()
    avg_pro_by_party = df.groupby('party')['money_pro'].mean()
    ax1_twin.plot(range(len(avg_pro_by_party)), avg_pro_by_party.values, 'ko-', linewidth=2, markersize=8)
    
    ax1.set_title('Stance Distribution by Party with Average Pro-Piracy Money', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Party')
    ax1.set_ylabel('Count')
    ax1_twin.set_ylabel('Average Pro Money ($)')
    ax1.legend(title='Stance', loc='upper left', fontsize=8)
    ax1.tick_params(axis='x', rotation=0)
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1: {str(e)}', ha='center', va='center', transform=ax1.transAxes)

# Row 1, Subplot 2: Scatter plot with marginal histograms
ax2 = plt.subplot(3, 3, 2)
try:
    for party in df['party'].unique():
        if party in party_colors:
            party_data = df[df['party'] == party]
            ax2.scatter(party_data['money_pro'], party_data['money_con'], 
                       c=party_colors[party], label=party, alpha=0.6, s=50)
    
    ax2.set_title('Pro vs Con Money by Party', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Pro-Piracy Money ($)')
    ax2.set_ylabel('Anti-Piracy Money ($)')
    ax2.legend(title='Party', fontsize=8)
    ax2.grid(True, alpha=0.3)
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2: {str(e)}', ha='center', va='center', transform=ax2.transAxes)

# Row 1, Subplot 3: Violin plot with box plots
ax3 = plt.subplot(3, 3, 3)
try:
    parties = [p for p in df['party'].unique() if p in party_colors]
    years_data = [df[df['party'] == party]['years'].values for party in parties]
    
    if len(years_data) > 0 and all(len(data) > 0 for data in years_data):
        violin_parts = ax3.violinplot(years_data, positions=range(len(parties)), showmeans=True)
        
        for i, party in enumerate(parties):
            if i < len(violin_parts['bodies']):
                violin_parts['bodies'][i].set_facecolor(party_colors[party])
                violin_parts['bodies'][i].set_alpha(0.7)
        
        box_parts = ax3.boxplot(years_data, positions=range(len(parties)), widths=0.3, patch_artist=True)
        for patch, party in zip(box_parts['boxes'], parties):
            patch.set_facecolor(party_colors[party])
            patch.set_alpha(0.5)
        
        ax3.set_xticks(range(len(parties)))
        ax3.set_xticklabels(parties)
    
    ax3.set_title('Years of Service Distribution by Party', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Party')
    ax3.set_ylabel('Years of Service')
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3: {str(e)}', ha='center', va='center', transform=ax3.transAxes)

# Row 2, Subplot 4: Grouped bar chart with error bars
ax4 = plt.subplot(3, 3, 4)
try:
    chamber_stats = df.groupby('chamber').agg({
        'money_pro': ['mean', 'std'],
        'money_con': ['mean', 'std']
    }).round(0)
    
    chambers = chamber_stats.index
    x_pos = np.arange(len(chambers))
    width = 0.35
    
    pro_means = chamber_stats[('money_pro', 'mean')].values
    pro_stds = chamber_stats[('money_pro', 'std')].values
    con_means = chamber_stats[('money_con', 'mean')].values
    con_stds = chamber_stats[('money_con', 'std')].values
    
    ax4.bar(x_pos - width/2, pro_means, width, yerr=pro_stds, 
            label='Pro-Piracy', color='#ff7f0e', alpha=0.8, capsize=5)
    ax4.bar(x_pos + width/2, con_means, width, yerr=con_stds,
            label='Anti-Piracy', color='#1f77b4', alpha=0.8, capsize=5)
    
    ax4.set_title('Average Money by Chamber with Standard Deviation', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Chamber')
    ax4.set_ylabel('Average Money ($)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([c.title() for c in chambers])
    ax4.legend(fontsize=8)
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4: {str(e)}', ha='center', va='center', transform=ax4.transAxes)

# Row 2, Subplot 5: Side-by-side pie charts
ax5 = plt.subplot(3, 3, 5)
try:
    chambers = df['chamber'].unique()
    ax5.set_title('Stance Composition by Chamber', fontweight='bold', fontsize=11)
    
    for i, chamber in enumerate(chambers):
        chamber_data = df[df['chamber'] == chamber]
        stance_counts = chamber_data['stance'].value_counts()
        
        if len(stance_counts) > 0:
            # Create pie chart
            colors = [stance_colors.get(stance, '#7f7f7f') for stance in stance_counts.index]
            wedges, texts, autotexts = ax5.pie(stance_counts.values, labels=stance_counts.index, 
                                              autopct='%1.1f%%', colors=colors,
                                              center=(i*2, 0), radius=0.8)
            
            # Add chamber title
            ax5.text(i*2, -1.5, chamber.title(), ha='center', fontweight='bold', fontsize=10)
    
    ax5.set_xlim(-1.5, len(chambers)*2-0.5)
    ax5.set_ylim(-2, 1.5)
    ax5.axis('equal')
except Exception as e:
    ax5.text(0.5, 0.5, f'Error in subplot 5: {str(e)}', ha='center', va='center', transform=ax5.transAxes)

# Row 2, Subplot 6: Scatter plot with trend lines
ax6 = plt.subplot(3, 3, 6)
try:
    for chamber in df['chamber'].unique():
        chamber_data = df[df['chamber'] == chamber]
        color = chamber_colors.get(chamber, '#7f7f7f')
        
        ax6.scatter(chamber_data['years'], chamber_data['total_money'], 
                   c=color, label=chamber.title(), alpha=0.6, s=50)
        
        # Add trend line if we have enough data
        if len(chamber_data) > 1:
            z = np.polyfit(chamber_data['years'], chamber_data['total_money'], 1)
            p = np.poly1d(z)
            years_sorted = np.sort(chamber_data['years'])
            ax6.plot(years_sorted, p(years_sorted), color=color, linestyle='--', linewidth=2)
    
    ax6.set_title('Years vs Total Money by Chamber with Trend Lines', fontweight='bold', fontsize=11)
    ax6.set_xlabel('Years of Service')
    ax6.set_ylabel('Total Money ($)')
    ax6.legend(title='Chamber', fontsize=8)
    ax6.grid(True, alpha=0.3)
except Exception as e:
    ax6.text(0.5, 0.5, f'Error in subplot 6: {str(e)}', ha='center', va='center', transform=ax6.transAxes)

# Row 3, Subplot 7: Heatmap
ax7 = plt.subplot(3, 3, 7)
try:
    heatmap_data = df.groupby(['party', 'chamber'])['money_pro'].mean().unstack(fill_value=0)
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax7, 
                cbar_kws={'label': 'Average Pro Money ($)'})
    ax7.set_title('Average Pro Money by Party-Chamber Combinations', fontweight='bold', fontsize=11)
    ax7.set_xlabel('Chamber')
    ax7.set_ylabel('Party')
except Exception as e:
    ax7.text(0.5, 0.5, f'Error in subplot 7: {str(e)}', ha='center', va='center', transform=ax7.transAxes)

# Row 3, Subplot 8: Strip plot with box plot overlay
ax8 = plt.subplot(3, 3, 8)
try:
    stances = df['stance'].unique()
    ratio_data = []
    
    for i, stance in enumerate(stances):
        stance_data = df[df['stance'] == stance]
        valid_ratios = stance_data[stance_data['money_ratio'] < 100]['money_ratio']
        
        if len(valid_ratios) > 0:
            ratio_data.append(valid_ratios.values)
            # Add jitter to x-coordinates
            x_coords = np.random.normal(i, 0.1, len(valid_ratios))
            ax8.scatter(x_coords, valid_ratios, alpha=0.6, 
                       c=stance_colors.get(stance, '#7f7f7f'), s=30)
        else:
            ratio_data.append([0])  # Add empty data to maintain position
    
    if len(ratio_data) > 0:
        box_parts = ax8.boxplot(ratio_data, positions=range(len(stances)), widths=0.3, patch_artist=True)
        for patch, stance in zip(box_parts['boxes'], stances):
            patch.set_facecolor(stance_colors.get(stance, '#7f7f7f'))
            patch.set_alpha(0.7)
    
    ax8.set_title('Money Ratio Distribution by Stance', fontweight='bold', fontsize=11)
    ax8.set_xlabel('Stance')
    ax8.set_ylabel('Money Ratio (Pro/Con)')
    ax8.set_xticks(range(len(stances)))
    ax8.set_xticklabels(stances)
except Exception as e:
    ax8.text(0.5, 0.5, f'Error in subplot 8: {str(e)}', ha='center', va='center', transform=ax8.transAxes)

# Row 3, Subplot 9: Parallel coordinates plot
ax9 = plt.subplot(3, 3, 9)
try:
    # Create stance numeric mapping
    stance_map = {'yes': 1, 'no': 0, 'unknown': 0.5}
    df_norm = df.copy()
    df_norm['stance_num'] = df_norm['stance'].map(stance_map)
    
    # Normalize data
    cols_to_norm = ['years', 'money_pro', 'money_con', 'stance_num']
    for col in cols_to_norm:
        col_data = df_norm[col]
        if col_data.max() != col_data.min():
            df_norm[f'{col}_norm'] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
        else:
            df_norm[f'{col}_norm'] = 0.5
    
    # Sample data to avoid overcrowding
    sample_size = min(100, len(df_norm))
    sample_df = df_norm.sample(sample_size) if len(df_norm) > 0 else df_norm
    
    for idx, row in sample_df.iterrows():
        y_vals = [row[f'{col}_norm'] for col in cols_to_norm]
        party_color = party_colors.get(row['party'], '#7f7f7f')
        ax9.plot(range(len(cols_to_norm)), y_vals, alpha=0.3, color=party_color)
    
    ax9.set_title('Parallel Coordinates: Years, Pro Money, Con Money, Stance', fontweight='bold', fontsize=11)
    ax9.set_xticks(range(len(cols_to_norm)))
    ax9.set_xticklabels(['Years', 'Pro Money', 'Con Money', 'Stance'])
    ax9.set_ylabel('Normalized Values')
    ax9.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=party) 
                      for party, color in party_colors.items() if party in df['party'].unique()]
    ax9.legend(handles=legend_elements, title='Party', loc='upper right', fontsize=8)
    
except Exception as e:
    ax9.text(0.5, 0.5, f'Error in subplot 9: {str(e)}', ha='center', va='center', transform=ax9.transAxes)

# Adjust layout and save
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)
plt.savefig('piracy_analysis.png', dpi=300, bbox_inches='tight')
plt.show()