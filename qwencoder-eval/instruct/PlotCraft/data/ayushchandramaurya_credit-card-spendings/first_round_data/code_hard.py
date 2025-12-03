import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('Credit card transactions - India - Simple.csv')

# Clean and prepare data efficiently
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
df['City_Clean'] = df['City'].str.replace(', India', '')

# Create color palettes
card_colors = {'Gold': '#FFD700', 'Platinum': '#E5E4E2', 'Silver': '#C0C0C0', 'Signature': '#8B4513'}
city_colors = {'Delhi': '#FF6B6B', 'Greater Mumbai': '#4ECDC4', 'Bengaluru': '#45B7D1', 'Ahmedabad': '#96CEB4', 'Chennai': '#FFEAA7'}

# Pre-compute aggregated data to avoid repeated calculations
city_card_agg = df.groupby(['City_Clean', 'Card Type']).agg({
    'Amount': ['sum', 'mean', 'count']
}).round(0)
city_card_agg.columns = ['Total_Amount', 'Avg_Amount', 'Transaction_Count']
city_card_agg = city_card_agg.reset_index()

city_totals = df.groupby('City_Clean')['Amount'].sum().sort_values(ascending=False)
card_totals = df.groupby('Card Type')['Amount'].sum()

# Get top cities and card types for efficiency
top_cities = city_totals.head(4).index.tolist()
card_types = card_totals.index.tolist()

# Create the 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.patch.set_facecolor('white')

# Subplot 1,1: Stacked bar chart with scatter overlay
ax1 = axes[0, 0]
bottom = np.zeros(len(top_cities))

for card in card_types:
    amounts = []
    for city in top_cities:
        amount = city_card_agg[(city_card_agg['City_Clean'] == city) & 
                              (city_card_agg['Card Type'] == card)]['Total_Amount'].sum()
        amounts.append(amount)
    
    ax1.bar(top_cities, amounts, bottom=bottom, label=card, color=card_colors[card], alpha=0.8)
    
    # Add scatter points (simplified)
    for i, city in enumerate(top_cities):
        count = city_card_agg[(city_card_agg['City_Clean'] == city) & 
                             (city_card_agg['Card Type'] == card)]['Transaction_Count'].sum()
        if count > 0:
            ax1.scatter(i, bottom[i] + amounts[i]/2, s=min(count/20, 100), color='black', alpha=0.5)
    
    bottom += amounts

ax1.set_title('City Spending by Card Type', fontweight='bold', fontsize=10)
ax1.set_ylabel('Amount (₹)')
ax1.legend(fontsize=8)
ax1.tick_params(axis='x', rotation=45, labelsize=8)

# Subplot 1,2: Pie chart with donut
ax2 = axes[0, 1]
card_amounts = [card_totals[card] for card in card_types]
wedges1, texts1, autotexts1 = ax2.pie(card_amounts, labels=card_types, autopct='%1.1f%%',
                                      colors=[card_colors[card] for card in card_types],
                                      radius=1, startangle=90)

city_amounts = [city_totals[city] for city in top_cities]
wedges2, texts2, autotexts2 = ax2.pie(city_amounts, labels=top_cities, autopct='%1.1f%%',
                                      colors=[city_colors.get(city, '#CCCCCC') for city in top_cities],
                                      radius=0.6, startangle=90)

ax2.set_title('Spending Composition', fontweight='bold', fontsize=10)

# Subplot 1,3: Simplified treemap
ax3 = axes[0, 2]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Create simple rectangles for top combinations
treemap_data = []
for city in top_cities[:3]:
    for card in card_types[:3]:
        amount = city_card_agg[(city_card_agg['City_Clean'] == city) & 
                              (city_card_agg['Card Type'] == card)]['Total_Amount'].sum()
        if amount > 0:
            treemap_data.append((city, card, amount))

treemap_data.sort(key=lambda x: x[2], reverse=True)

x, y = 0, 0
for i, (city, card, amount) in enumerate(treemap_data[:6]):
    width = 3
    height = 2
    
    rect = Rectangle((x, y), width, height, facecolor=card_colors[card], 
                    edgecolor='white', linewidth=1, alpha=0.7)
    ax3.add_patch(rect)
    ax3.text(x + width/2, y + height/2, f'{city[:8]}\n{card}', 
            ha='center', va='center', fontsize=7, fontweight='bold')
    
    x += width
    if x >= 9:
        x = 0
        y += height

ax3.set_title('Spending Treemap', fontweight='bold', fontsize=10)
ax3.set_xticks([])
ax3.set_yticks([])

# Subplot 2,1: Horizontal stacked bar
ax4 = axes[1, 0]
left = np.zeros(len(top_cities))

for card in card_types:
    amounts = []
    for city in top_cities:
        amount = city_card_agg[(city_card_agg['City_Clean'] == city) & 
                              (city_card_agg['Card Type'] == card)]['Total_Amount'].sum()
        amounts.append(amount)
    
    ax4.barh(top_cities, amounts, left=left, label=card, color=card_colors[card], alpha=0.8)
    left += amounts

ax4.set_title('Horizontal City Distribution', fontweight='bold', fontsize=10)
ax4.set_xlabel('Amount (₹)')
ax4.legend(fontsize=8)

# Subplot 2,2: Waffle chart
ax5 = axes[1, 1]
total_spending = df['Amount'].sum()
card_percentages = {card: (amount/total_spending)*100 for card, amount in card_totals.items()}

# Create 10x10 grid
grid_size = 10
squares_per_card = {card: max(1, int(pct/100 * grid_size * grid_size)) for card, pct in card_percentages.items()}

current_square = 0
for i in range(grid_size):
    for j in range(grid_size):
        card = None
        temp_current = current_square
        for card_type, count in squares_per_card.items():
            if temp_current < count:
                card = card_type
                break
            temp_current -= count
        
        if card and current_square < grid_size * grid_size:
            rect = Rectangle((j, i), 1, 1, facecolor=card_colors[card], 
                           edgecolor='white', linewidth=0.5)
            ax5.add_patch(rect)
        current_square += 1

ax5.set_xlim(0, grid_size)
ax5.set_ylim(0, grid_size)
ax5.set_title('Waffle Chart: Card Distribution', fontweight='bold', fontsize=10)
ax5.set_xticks([])
ax5.set_yticks([])

# Subplot 2,3: Simplified sunburst
ax6 = axes[1, 2]
city_amounts_top = [city_totals[city] for city in top_cities]
ax6.pie(city_amounts_top, radius=0.5, 
        colors=[city_colors.get(city, '#CCCCCC') for city in top_cities],
        startangle=90)

# Outer ring
outer_data = []
outer_colors = []
for city in top_cities:
    for card in card_types:
        amount = city_card_agg[(city_card_agg['City_Clean'] == city) & 
                              (city_card_agg['Card Type'] == card)]['Total_Amount'].sum()
        if amount > 0:
            outer_data.append(amount)
            outer_colors.append(card_colors[card])

if outer_data:
    ax6.pie(outer_data, radius=1, colors=outer_colors, startangle=90)

ax6.set_title('Sunburst: City-Card', fontweight='bold', fontsize=10)

# Subplot 3,1: Simplified area chart
ax7 = axes[2, 0]
# Sample data for performance
sample_df = df.sample(min(1000, len(df))).sort_values('Date')
sample_df = sample_df.dropna(subset=['Date'])

if not sample_df.empty:
    sample_df['Days'] = (sample_df['Date'] - sample_df['Date'].min()).dt.days
    
    for i, card in enumerate(card_types):
        card_data = sample_df[sample_df['Card Type'] == card]
        if not card_data.empty:
            # Simple cumulative plot
            cumsum = card_data.groupby('Days')['Amount'].sum().cumsum()
            ax7.plot(cumsum.index, cumsum.values, color=card_colors[card], 
                    label=card, linewidth=2, alpha=0.7)

ax7.set_title('Cumulative Spending Over Time', fontweight='bold', fontsize=10)
ax7.set_xlabel('Days')
ax7.set_ylabel('Cumulative Amount')
ax7.legend(fontsize=8)

# Subplot 3,2: Bubble chart
ax8 = axes[2, 1]
for i, city in enumerate(top_cities):
    for j, card in enumerate(card_types):
        subset = city_card_agg[(city_card_agg['City_Clean'] == city) & 
                              (city_card_agg['Card Type'] == card)]
        if not subset.empty:
            amount = subset['Total_Amount'].iloc[0]
            count = subset['Transaction_Count'].iloc[0]
            ax8.scatter(i, j, s=amount/50000, c=count, 
                       cmap='viridis', alpha=0.6, edgecolors='black')

ax8.set_xticks(range(len(top_cities)))
ax8.set_xticklabels(top_cities, rotation=45, fontsize=8)
ax8.set_yticks(range(len(card_types)))
ax8.set_yticklabels(card_types, fontsize=8)
ax8.set_title('Bubble Chart: City vs Card', fontweight='bold', fontsize=10)

# Subplot 3,3: Simplified parallel coordinates
ax9 = axes[2, 2]
# Sample for performance
sample_size = min(200, len(df))
df_sample = df.sample(sample_size)
df_sample['Amount_Norm'] = df_sample['Amount'] / df_sample['Amount'].max()

city_map = {city: i for i, city in enumerate(top_cities)}
card_map = {card: i for i, card in enumerate(card_types)}

# Create spending ranges
df_sample['Amount_Range'] = pd.cut(df_sample['Amount'], bins=3, labels=['Low', 'Medium', 'High'])
range_colors = {'Low': '#90EE90', 'Medium': '#FFD700', 'High': '#FF6B6B'}

for _, row in df_sample.iterrows():
    if row['City_Clean'] in city_map and row['Card Type'] in card_map:
        x_vals = [0, 1, 2]
        y_vals = [city_map[row['City_Clean']], 
                 card_map[row['Card Type']], 
                 row['Amount_Norm']]
        
        color = range_colors.get(row['Amount_Range'], '#CCCCCC')
        ax9.plot(x_vals, y_vals, color=color, alpha=0.4, linewidth=0.8)

ax9.set_xticks([0, 1, 2])
ax9.set_xticklabels(['City', 'Card Type', 'Amount'], fontsize=8)
ax9.set_title('Parallel Coordinates', fontweight='bold', fontsize=10)

# Adjust layout and save
plt.tight_layout(pad=1.5)
plt.savefig('credit_card_composition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()