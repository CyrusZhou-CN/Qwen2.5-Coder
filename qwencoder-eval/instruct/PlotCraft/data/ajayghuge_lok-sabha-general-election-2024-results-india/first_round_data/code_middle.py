import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('GE_2024_Results.csv')

# Data preprocessing - more efficient approach
# Convert '% of Votes' to numeric, handling any non-numeric values
df['Vote_Percentage'] = pd.to_numeric(df['% of Votes'], errors='coerce')

# Remove rows with invalid vote percentages
df = df.dropna(subset=['Vote_Percentage'])

# Calculate party-level statistics more efficiently
party_stats = df.groupby('Party', as_index=False).agg({
    'Vote_Percentage': ['sum', 'mean', 'count']
}).round(2)

# Flatten column names
party_stats.columns = ['Party', 'Total_Vote_Percentage', 'Avg_Vote_Percentage', 'Constituencies_Contested']

# Get top 10 parties by total vote percentage
top_10_parties = party_stats.nlargest(10, 'Total_Vote_Percentage').reset_index(drop=True)

# Define colors for major parties
party_colors = {
    'Bharatiya Janata Party': '#FF6B35',
    'Indian National Congress': '#19AAED', 
    'Samajwadi Party': '#FF2222',
    'All India Trinamool Congress': '#20C646',
    'Dravida Munnetra Kazhagam': '#DC143C',
    'Janata Dal  (United)': '#138808',
    'Telugu Desam': '#F0E68C',
    'Nationalist Congress Party  (Sharadchandra Pawar)': '#00B2A0',
    'Shiv Sena (Uddhav Balasaheb Thackrey)': '#F37020',
    'Lok Janshakti Party(Ram Vilas)': '#4169E1'
}

# Assign colors to parties
colors = [party_colors.get(party, '#708090') for party in top_10_parties['Party']]

# Create the composite visualization
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')

# Create subplots with custom spacing
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# Left subplot: Horizontal bar chart of total vote percentages
y_pos = np.arange(len(top_10_parties))
bars = ax1.barh(y_pos, top_10_parties['Total_Vote_Percentage'], 
                color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

# Customize the bar chart
ax1.set_yticks(y_pos)
# Truncate long party names for better display
party_labels = [name[:30] + '...' if len(name) > 30 else name for name in top_10_parties['Party']]
ax1.set_yticklabels(party_labels, fontsize=10)
ax1.invert_yaxis()  # Top party at the top
ax1.set_xlabel('Total Vote Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Top 10 Political Parties by Total Vote Share\n2024 Lok Sabha Elections', 
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.set_facecolor('#FAFAFA')

# Add value labels on bars
for i, value in enumerate(top_10_parties['Total_Vote_Percentage']):
    ax1.text(value + max(top_10_parties['Total_Vote_Percentage']) * 0.01, i, 
             f'{value:.1f}%', va='center', fontsize=9, fontweight='bold')

# Right subplot: Scatter plot of constituencies contested vs average vote percentage
scatter = ax2.scatter(top_10_parties['Constituencies_Contested'], 
                     top_10_parties['Avg_Vote_Percentage'],
                     c=colors, s=120, alpha=0.85, edgecolors='black', linewidth=1.2)

# Customize the scatter plot
ax2.set_xlabel('Number of Constituencies Contested', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Vote Percentage per Constituency (%)', fontsize=12, fontweight='bold')
ax2.set_title('Electoral Strategy: Reach vs Efficiency\n2024 Lok Sabha Elections', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('#FAFAFA')

# Add party labels to scatter points with better positioning
for i, row in top_10_parties.iterrows():
    # Use shorter names for annotations
    party_name = row['Party']
    if len(party_name) > 25:
        party_name = party_name[:22] + '...'
    
    ax2.annotate(party_name, 
                (row['Constituencies_Contested'], row['Avg_Vote_Percentage']),
                xytext=(8, 8), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                ha='left')

# Create a simplified legend with top 5 parties only to avoid clutter
legend_parties = top_10_parties.head(5)
legend_colors = colors[:5]
legend_elements = []

for party, color in zip(legend_parties['Party'], legend_colors):
    short_name = party[:20] + '...' if len(party) > 20 else party
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.85, 
                                       edgecolor='black', linewidth=0.5))

# Add legend
fig.legend(legend_elements, 
          [party[:20] + '...' if len(party) > 20 else party for party in legend_parties['Party']], 
          loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), 
          fontsize=10, frameon=True, fancybox=True, shadow=True)

# Add summary statistics as text
summary_text = f"Total Parties Analyzed: {len(party_stats)}\nTop 10 Parties Cover: {top_10_parties['Total_Vote_Percentage'].sum():.1f}% of Total Votes"
fig.text(0.02, 0.98, summary_text, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.92)

# Save the plot
plt.savefig('lok_sabha_2024_party_analysis.png', dpi=300, bbox_inches='tight')
plt.show()