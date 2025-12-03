import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('Credit card transactions - India - Simple.csv')

# Data preprocessing
df['Amount'] = df['Amount'].astype(float)

# Get unique cities and card types
cities = df['City'].unique()
card_types = df['Card Type'].unique()

# Create color palette for card types
colors = plt.cm.Set2(np.linspace(0, 1, len(card_types)))
color_map = dict(zip(card_types, colors))

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], 
                      hspace=0.3, wspace=0.3)

# Main scatter plot
ax_main = fig.add_subplot(gs[0, 0])

# Create jittered x-positions for cities
city_positions = {city: i for i, city in enumerate(cities)}
jitter_strength = 0.3

# Plot scatter points for each card type
for i, card_type in enumerate(card_types):
    card_data = df[df['Card Type'] == card_type]
    
    # Create jittered x positions
    x_positions = []
    for city in card_data['City']:
        base_pos = city_positions[city]
        jitter = np.random.uniform(-jitter_strength, jitter_strength)
        x_positions.append(base_pos + jitter)
    
    # Size points proportionally to amount (normalized)
    sizes = (card_data['Amount'] - card_data['Amount'].min()) / (card_data['Amount'].max() - card_data['Amount'].min()) * 100 + 20
    
    ax_main.scatter(x_positions, card_data['Amount'], 
                   c=[color_map[card_type]], label=card_type, 
                   s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)

# Add trend lines for each card type
for card_type in card_types:
    card_data = df[df['Card Type'] == card_type]
    city_numeric = [city_positions[city] for city in card_data['City']]
    
    # Calculate trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(city_numeric, card_data['Amount'])
    line_x = np.array([0, len(cities)-1])
    line_y = slope * line_x + intercept
    
    ax_main.plot(line_x, line_y, color=color_map[card_type], 
                linestyle='--', alpha=0.8, linewidth=2)
    
    # Add correlation annotation
    ax_main.text(0.02, 0.98 - list(card_types).index(card_type) * 0.05, 
                f'{card_type}: r={r_value:.3f}', 
                transform=ax_main.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color_map[card_type], alpha=0.3))

# Customize main plot
ax_main.set_xlabel('Cities', fontweight='bold', fontsize=12)
ax_main.set_ylabel('Amount (INR)', fontweight='bold', fontsize=12)
ax_main.set_title('Credit Card Spending by City and Card Type\n(Point size proportional to amount)', 
                 fontweight='bold', fontsize=14, pad=20)
ax_main.set_xticks(range(len(cities)))
ax_main.set_xticklabels([city.replace(', India', '') for city in cities], rotation=45, ha='right')
ax_main.legend(title='Card Type', title_fontsize=11, fontsize=10, loc='upper right')
ax_main.grid(True, alpha=0.3)
ax_main.set_facecolor('white')

# Box plot for card types (right subplot)
ax_box = fig.add_subplot(gs[0, 1])

# Prepare data for box plot
box_data = [df[df['Card Type'] == card_type]['Amount'] for card_type in card_types]
box_colors = [color_map[card_type] for card_type in card_types]

bp = ax_box.boxplot(box_data, labels=card_types, patch_artist=True, 
                   vert=True, widths=0.6)

# Color the box plots
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_box.set_ylabel('Amount (INR)', fontweight='bold', fontsize=12)
ax_box.set_title('Amount Distribution\nby Card Type', fontweight='bold', fontsize=12)
ax_box.tick_params(axis='x', rotation=45)
ax_box.grid(True, alpha=0.3)
ax_box.set_facecolor('white')

# City-wise spending distribution (bottom subplot)
ax_city = fig.add_subplot(gs[1, 0])

city_means = df.groupby('City')['Amount'].mean().sort_values(ascending=False)
bars = ax_city.bar(range(len(city_means)), city_means.values, 
                  color='steelblue', alpha=0.7, edgecolor='white', linewidth=1)

ax_city.set_xlabel('Cities', fontweight='bold', fontsize=12)
ax_city.set_ylabel('Average Amount (INR)', fontweight='bold', fontsize=12)
ax_city.set_title('Average Spending by City', fontweight='bold', fontsize=12)
ax_city.set_xticks(range(len(city_means)))
ax_city.set_xticklabels([city.replace(', India', '') for city in city_means.index], 
                       rotation=45, ha='right')
ax_city.grid(True, alpha=0.3, axis='y')
ax_city.set_facecolor('white')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax_city.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)

# Statistical summary (bottom right)
ax_stats = fig.add_subplot(gs[1, 1])
ax_stats.axis('off')

# Calculate statistics
total_transactions = len(df)
avg_amount = df['Amount'].mean()
total_amount = df['Amount'].sum()
card_type_counts = df['Card Type'].value_counts()

stats_text = f"""Statistical Summary:
Total Transactions: {total_transactions:,}
Average Amount: ₹{avg_amount:,.0f}
Total Volume: ₹{total_amount:,.0f}

Card Type Distribution:
"""

for card_type, count in card_type_counts.items():
    percentage = (count / total_transactions) * 100
    stats_text += f"{card_type}: {count:,} ({percentage:.1f}%)\n"

ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

# Set white background for the entire figure
fig.patch.set_facecolor('white')

# Final layout adjustment
plt.tight_layout()
plt.show()