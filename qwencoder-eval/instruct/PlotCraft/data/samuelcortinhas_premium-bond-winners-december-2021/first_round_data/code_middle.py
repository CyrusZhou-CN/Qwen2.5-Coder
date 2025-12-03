import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the months and their corresponding file names
month_files = {
    1: 'prize-january-2022.csv',
    2: 'prize-february-2022.csv', 
    3: 'prize-march-2022.csv',
    4: 'prize-april-2022.csv',
    5: 'prize-may-2022.csv',
    6: 'prize-june-2022.csv',
    7: 'prize-july-2022.csv',
    8: 'prize-august-2022.csv',
    9: 'prize-september-2022.csv',
    10: 'prize-october-2022.csv',
    11: 'prize-november-2022.csv',
    12: 'prize-december-2022.csv'
}

# Function to clean prize values
def clean_prize_value(prize_str):
    """Convert prize string to numeric value"""
    try:
        return int(prize_str.replace('£', '').replace(',', ''))
    except:
        return 0

# Load and process data efficiently
monthly_data = {}
for month_num, filename in month_files.items():
    if os.path.exists(filename):
        try:
            # Load only the Prize Value column to speed up processing
            df = pd.read_csv(filename, usecols=['Prize Value'])
            # Clean prize values
            df['Prize Value Clean'] = df['Prize Value'].apply(clean_prize_value)
            # Filter out any zero values
            df = df[df['Prize Value Clean'] > 0]
            monthly_data[month_num] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

# Define prize tiers
prize_tiers = [1000000, 100000, 50000, 25000, 10000, 5000, 1000]
tier_labels = ['£1,000,000', '£100,000', '£50,000', '£25,000', '£10,000', '£5,000', '£1,000']

# Calculate monthly statistics
months = sorted(monthly_data.keys())
monthly_totals = []
tier_counts = {tier: [] for tier in prize_tiers}

for month in months:
    df = monthly_data[month]
    
    # Calculate total monthly prize value
    total_value = df['Prize Value Clean'].sum()
    monthly_totals.append(total_value)
    
    # Count prizes by tier
    for tier in prize_tiers:
        count = len(df[df['Prize Value Clean'] == tier])
        tier_counts[tier].append(count)

# Create month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x_months = [month_labels[m-1] for m in months]

# Create the visualization
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Top plot: Line chart of total monthly prize values
ax1.plot(x_months, monthly_totals, marker='o', linewidth=3, markersize=8, 
         color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
ax1.set_title('Total Monthly Prize Value Distribution - 2022', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Total Prize Value (£)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('white')

# Format y-axis for currency (millions)
def currency_formatter(x, p):
    return f'£{x/1e6:.1f}M'
ax1.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))

# Bottom plot: Stacked area chart of prize tier composition
# Prepare data for stacking
tier_data = []
for tier in prize_tiers:
    tier_data.append(tier_counts[tier])

# Define colors for each tier
colors = ['#8B0000', '#DC143C', '#FF6347', '#FFA500', '#FFD700', '#32CD32', '#4169E1']

# Create stacked area chart
ax2.stackplot(x_months, *tier_data, labels=tier_labels, colors=colors, alpha=0.8)
ax2.set_title('Prize Tier Distribution by Month - 2022', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Prizes', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_facecolor('white')

# Add legend for the stacked area chart
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, 
           fancybox=True, shadow=True, framealpha=0.9)

# Styling improvements
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Rotate x-axis labels for better readability
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Save the plot
plt.savefig('premium_bond_analysis_2022.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()