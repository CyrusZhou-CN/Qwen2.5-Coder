import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import squarify

# Load and preprocess data
df = pd.read_csv('IPL_Sold_players_2013_23.csv')

# Convert price from string to numeric (removing commas and converting to crores)
df['Price_Numeric'] = df['Price'].str.replace(',', '').astype(float) / 10000000

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('IPL Auction Spending Patterns: Comprehensive Analysis (2013-2023)', 
             fontsize=20, fontweight='bold', y=0.95)

# Subplot 1: Top-left - Stacked bar chart with overlay pie charts
ax1 = plt.subplot(2, 2, 1, facecolor='white')

# Calculate team spending by player type
team_spending = df.groupby(['Team', 'Type'])['Price_Numeric'].sum().unstack(fill_value=0)
team_total = team_spending.sum(axis=1).sort_values(ascending=False)
team_spending = team_spending.loc[team_total.index]

# Create stacked bar chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
team_spending.plot(kind='bar', stacked=True, ax=ax1, color=colors[:len(team_spending.columns)], 
                   width=0.8, alpha=0.8)

ax1.set_title('Team Spending by Player Type with Composition Details', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Teams', fontweight='bold')
ax1.set_ylabel('Spending (Crores)', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Add small pie charts for top 3 teams
top_teams = team_total.head(3).index
for i, team in enumerate(top_teams):
    team_data = team_spending.loc[team]
    team_data = team_data[team_data > 0]
    
    # Create small pie chart
    pie_ax = fig.add_axes([0.15 + i*0.12, 0.75, 0.08, 0.08])
    pie_ax.pie(team_data.values, colors=colors[:len(team_data)], 
               startangle=90, wedgeprops=dict(width=0.7))
    pie_ax.set_title(team[:3], fontsize=8, fontweight='bold')

# Subplot 2: Top-right - Treemap with annotations
ax2 = plt.subplot(2, 2, 2, facecolor='white')

# Prepare data for treemap
team_type_spending = df.groupby(['Team', 'Type'])['Price_Numeric'].sum().reset_index()
team_nationality = df.groupby(['Team', 'Nationality'])['Price_Numeric'].sum().unstack(fill_value=0)
team_nationality_pct = team_nationality.div(team_nationality.sum(axis=1), axis=0) * 100

# Create treemap data
sizes = team_type_spending['Price_Numeric'].values
labels = [f"{row['Team'][:3]}\n{row['Type'][:4]}\n₹{row['Price_Numeric']:.1f}Cr" 
          for _, row in team_type_spending.iterrows()]

# Generate treemap
squarify.plot(sizes=sizes, label=labels, alpha=0.7, 
              color=plt.cm.Set3(np.linspace(0, 1, len(sizes))), ax=ax2)

ax2.set_title('Team Spending Hierarchy by Player Type', fontweight='bold', fontsize=14, pad=20)
ax2.axis('off')

# Add nationality annotations for top teams
for i, team in enumerate(team_total.head(5).index):
    if team in team_nationality_pct.index:
        indian_pct = team_nationality_pct.loc[team, 'Indian'] if 'Indian' in team_nationality_pct.columns else 0
        overseas_pct = team_nationality_pct.loc[team, 'Overseas'] if 'Overseas' in team_nationality_pct.columns else 0
        ax2.text(0.02, 0.98 - i*0.05, f"{team[:8]}: {indian_pct:.0f}% IND, {overseas_pct:.0f}% OS", 
                transform=ax2.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Subplot 3: Bottom-left - Waffle chart with pie chart
ax3 = plt.subplot(2, 2, 3, facecolor='white')

# Create waffle chart
total_spending = df.groupby('Team')['Price_Numeric'].sum().sort_values(ascending=False)
waffle_size = 20  # 20x20 grid
total_squares = waffle_size * waffle_size
spending_per_square = total_spending.sum() / total_squares

# Calculate squares per team
team_squares = (total_spending / spending_per_square).round().astype(int)
team_colors = plt.cm.Set3(np.linspace(0, 1, len(team_squares)))

# Create waffle grid
square_size = 1
current_square = 0
for i, (team, squares) in enumerate(team_squares.items()):
    for _ in range(int(squares)):
        if current_square >= total_squares:
            break
        row = current_square // waffle_size
        col = current_square % waffle_size
        
        rect = Rectangle((col * square_size, row * square_size), square_size, square_size,
                        facecolor=team_colors[i], edgecolor='white', linewidth=0.5)
        ax3.add_patch(rect)
        current_square += 1

ax3.set_xlim(0, waffle_size)
ax3.set_ylim(0, waffle_size)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('Overall Budget Distribution (Each Square ≈ ₹{:.1f}Cr)'.format(spending_per_square), 
              fontweight='bold', fontsize=14, pad=20)

# Add legend for waffle chart
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=team_colors[i], label=team[:8]) 
                  for i, team in enumerate(team_squares.index[:8])]
ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

# Add small pie chart for nationality composition
nationality_spending = df.groupby('Nationality')['Price_Numeric'].sum()
pie_ax3 = fig.add_axes([0.15, 0.15, 0.12, 0.12])
pie_ax3.pie(nationality_spending.values, labels=nationality_spending.index, 
            autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF'])
pie_ax3.set_title('Nationality Split', fontsize=10, fontweight='bold')

# Subplot 4: Bottom-right - Stacked area chart
ax4 = plt.subplot(2, 2, 4, facecolor='white')

# Prepare data for cumulative spending
team_player_data = []
for team in df['Team'].unique():
    team_df = df[df['Team'] == team].sort_values('Price_Numeric', ascending=False)
    team_df['Cumulative'] = team_df['Price_Numeric'].cumsum()
    team_df['Player_Index'] = range(len(team_df))
    team_player_data.append(team_df)

# Create stacked area chart by player type
type_colors = {'Batter': '#FF6B6B', 'Bowler': '#4ECDC4', 'All-Rounder': '#45B7D1', 
               'Wicket-Keeper': '#96CEB4', 'Uncapped': '#FFEAA7'}

x_offset = 0
team_boundaries = []
for team_df in team_player_data:
    for player_type in team_df['Type'].unique():
        type_data = team_df[team_df['Type'] == player_type]
        if len(type_data) > 0:
            x_vals = np.arange(x_offset, x_offset + len(type_data))
            y_vals = type_data['Price_Numeric'].values
            ax4.fill_between(x_vals, 0, y_vals, alpha=0.7, 
                           color=type_colors.get(player_type, '#CCCCCC'),
                           label=player_type if x_offset == 0 else "")
    
    team_boundaries.append(x_offset + len(team_df))
    x_offset += len(team_df)

# Add vertical lines for team boundaries
for boundary in team_boundaries[:-1]:
    ax4.axvline(x=boundary, color='black', linestyle='--', alpha=0.5, linewidth=1)

ax4.set_title('Cumulative Spending Progression by Player Type', fontweight='bold', fontsize=14, pad=20)
ax4.set_xlabel('Players (sorted by price within teams)', fontweight='bold')
ax4.set_ylabel('Price (Crores)', fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

# Add team labels
team_centers = []
start = 0
for i, team_df in enumerate(team_player_data):
    center = start + len(team_df) / 2
    team_centers.append(center)
    start += len(team_df)

# Add team labels at bottom
for i, (center, team) in enumerate(zip(team_centers[::2], df['Team'].unique()[::2])):
    ax4.text(center, ax4.get_ylim()[0] - (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.05, 
             team[:6], ha='center', va='top', fontsize=8, rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
plt.show()