import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Check which files exist and load them
available_files = []
months_mapping = {}

# List all CSV files in the directory
for file in os.listdir('.'):
    if file.startswith('prize-') and file.endswith('.csv'):
        available_files.append(file)

# Extract month information from filenames and sort chronologically
month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']

# Load available data files
monthly_data = {}
available_months = []

for file in available_files:
    # Extract month from filename
    month_part = file.replace('prize-', '').replace('-2022.csv', '')
    if month_part in month_order:
        df = pd.read_csv(file)
        monthly_data[month_part] = df
        available_months.append(month_part)

# Sort months chronologically
available_months.sort(key=lambda x: month_order.index(x))

print(f"Found data for {len(available_months)} months: {available_months}")

# Define prize categories and colors
prize_categories = ['£1,000,000', '£100,000', '£50,000', '£25,000', '£10,000', '£5,000', '£1,000']
colors = ['#8B0000', '#FF4500', '#FFD700', '#32CD32', '#4169E1', '#9370DB', '#FF69B4']

# Create month name mapping
month_names_map = {
    'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
    'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug',
    'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec'
}

month_names = [month_names_map[month] for month in available_months]

# Process data for visualization
million_winners = []
total_prize_values = []
prize_composition = {category: [] for category in prize_categories}

def convert_prize_to_numeric(prize_str):
    """Convert prize string to numeric value"""
    try:
        return int(str(prize_str).replace('£', '').replace(',', ''))
    except:
        return 0

for month in available_months:
    df = monthly_data[month]
    
    # Clean the Prize Value column
    df['Prize Value'] = df['Prize Value'].astype(str).str.strip()
    
    # Count £1,000,000 winners
    million_count = len(df[df['Prize Value'] == '£1,000,000'])
    million_winners.append(million_count)
    
    # Calculate total prize value
    df['Prize_Numeric'] = df['Prize Value'].apply(convert_prize_to_numeric)
    total_value = df['Prize_Numeric'].sum()
    total_prize_values.append(total_value / 1000000)  # Convert to millions
    
    # Count prizes by category
    for category in prize_categories:
        count = len(df[df['Prize Value'] == category])
        prize_composition[category].append(count)

# Create the composite visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
fig.patch.set_facecolor('white')

# Top plot: Line chart for million winners + bar chart for total prize values
ax1_twin = ax1.twinx()

# Bar chart for total prize values
bars = ax1.bar(month_names, total_prize_values, alpha=0.6, color='#4A90E2', 
               label='Total Prize Value (£M)', width=0.6)

# Line chart for million winners
line = ax1_twin.plot(month_names, million_winners, color='#8B0000', marker='o', 
                     linewidth=3, markersize=8, label='£1M Winners')

# Styling for top plot
ax1.set_title('Monthly Premium Bond Performance: Million Pound Winners vs Total Prize Distribution', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Month (2022)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Prize Value (£ Millions)', fontsize=12, fontweight='bold', color='#4A90E2')
ax1_twin.set_ylabel('Number of £1M Winners', fontsize=12, fontweight='bold', color='#8B0000')

# Color the y-axis labels to match the data
ax1.tick_params(axis='y', labelcolor='#4A90E2')
ax1_twin.tick_params(axis='y', labelcolor='#8B0000')

# Add grid for better readability
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Create combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
           fancybox=True, shadow=True)

# Bottom plot: Stacked area chart for prize composition
# Only include categories that have data
active_categories = []
active_colors = []
active_data = []

for i, category in enumerate(prize_categories):
    if any(count > 0 for count in prize_composition[category]):
        active_categories.append(category)
        active_colors.append(colors[i])
        active_data.append(prize_composition[category])

if active_data:
    prize_data_array = np.array(active_data)
    ax2.stackplot(month_names, *prize_data_array, labels=active_categories, 
                  colors=active_colors, alpha=0.8)

# Styling for bottom plot
ax2.set_title('Monthly Prize Structure Evolution: Distribution Across Prize Tiers', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Month (2022)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Winners', fontsize=12, fontweight='bold')

# Add grid for better readability
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# Legend for stacked area chart
if active_categories:
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, 
               fancybox=True, shadow=True)

# Ensure all text is readable and properly spaced
plt.setp(ax1.get_xticklabels(), fontsize=10, rotation=45)
plt.setp(ax2.get_xticklabels(), fontsize=10, rotation=45)
plt.setp(ax1.get_yticklabels(), fontsize=10)
plt.setp(ax1_twin.get_yticklabels(), fontsize=10)
plt.setp(ax2.get_yticklabels(), fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, right=0.85)

# Add some summary statistics as text
total_million_winners = sum(million_winners)
total_prize_value = sum(total_prize_values)
fig.suptitle(f'2022 Premium Bond Analysis - Total £1M Winners: {total_million_winners}, Total Prizes: £{total_prize_value:.1f}M', 
             fontsize=14, y=0.98)

plt.savefig('premium_bond_analysis_2022.png', dpi=300, bbox_inches='tight')
plt.show()