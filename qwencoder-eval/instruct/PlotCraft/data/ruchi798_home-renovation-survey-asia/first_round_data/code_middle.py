import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('Home Renovation Survey _ Asia - Data.csv')

# Print column names to debug
print("Available columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")

# Create figure with 2x2 subplot grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palettes for each subplot
colors1 = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Blue-red palette
colors2 = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']  # Teal-orange palette
colors3 = ['#6A4C93', '#C44569', '#F8B500', '#F38BA8']  # Purple-pink palette
colors4 = ['#1B4332', '#40916C', '#95D5B2', '#D8F3DC']  # Green palette

# Use the correct column names from the dataset
house_type_col = "What's the type of house you live in:"
motivation_col = "Why would you renovate your home if you want to?"
neighborhood_col = "Which type of neighborhood do you live in?"
budget_col = "What's your budget for home renovation if you need one?"
room_col = "Which room or which part do you want to renovate most at present?"
decor_col = "What aspects of home decor / renovation do you value the most?"

# 1. Top-left: Stacked bar chart - Renovation motivations by house type
try:
    house_types = df[house_type_col].dropna().value_counts().index[:4]
    motivations = df[motivation_col].dropna().value_counts().index[:4]

    # Create cross-tabulation
    house_motivation_data = []
    house_labels = []
    for house in house_types:
        house_data = df[df[house_type_col] == house]
        motivation_counts = []
        for motivation in motivations:
            count = len(house_data[house_data[motivation_col] == motivation])
            motivation_counts.append(count)
        house_motivation_data.append(motivation_counts)
        # Shorten house type labels
        short_house = house.replace(' house', '').replace('Apartment/Unit', 'Apartment')
        house_labels.append(short_house)

    # Plot stacked bar chart
    house_motivation_data = np.array(house_motivation_data).T
    x_pos = np.arange(len(house_labels))
    bottom = np.zeros(len(house_labels))

    for i, motivation in enumerate(motivations):
        # Shorten motivation labels
        short_motivation = motivation.replace('Improve the functionality of home', 'Functionality')\
                                   .replace('Change of lifestyle', 'Lifestyle')\
                                   .replace('Lower energy costs', 'Energy costs')
        ax1.bar(x_pos, house_motivation_data[i], bottom=bottom, label=short_motivation, 
                color=colors1[i % len(colors1)], alpha=0.8)
        bottom += house_motivation_data[i]

    ax1.set_title('Renovation Motivations by House Type', fontweight='bold', fontsize=14, pad=20)
    ax1.set_xlabel('House Type', fontweight='bold')
    ax1.set_ylabel('Number of Responses', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(house_labels, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1:\n{str(e)}', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Renovation Motivations by House Type (Error)', fontweight='bold', fontsize=14)

# 2. Top-right: Pie chart - Preferred rooms/parts to renovate
try:
    room_data = df[room_col].dropna().value_counts()
    room_labels = room_data.index[:5]  # Top 5 rooms
    room_values = room_data.values[:5]

    # Handle any remaining categories as "Others"
    if len(room_data) > 5:
        others_count = room_data.values[5:].sum()
        room_labels = list(room_labels) + ['Others']
        room_values = list(room_values) + [others_count]

    wedges, texts, autotexts = ax2.pie(room_values, labels=room_labels, autopct='%1.1f%%', 
                                       colors=colors2[:len(room_values)], startangle=90)
    ax2.set_title('Preferred Rooms/Parts to Renovate', fontweight='bold', fontsize=14, pad=20)

    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2:\n{str(e)}', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Preferred Rooms/Parts to Renovate (Error)', fontweight='bold', fontsize=14)

# 3. Bottom-left: Stacked bar chart - Budget ranges by neighborhood type
try:
    neighborhoods = df[neighborhood_col].dropna().value_counts().index
    budgets = df[budget_col].dropna().value_counts().index[:4]

    # Create cross-tabulation for budget by neighborhood
    neighborhood_budget_data = []
    neighborhood_labels = []
    for neighborhood in neighborhoods:
        neighborhood_data = df[df[neighborhood_col] == neighborhood]
        budget_counts = []
        for budget in budgets:
            count = len(neighborhood_data[neighborhood_data[budget_col] == budget])
            budget_counts.append(count)
        neighborhood_budget_data.append(budget_counts)
        neighborhood_labels.append(neighborhood)

    # Plot stacked bar chart
    neighborhood_budget_data = np.array(neighborhood_budget_data).T
    x_pos = np.arange(len(neighborhood_labels))
    bottom = np.zeros(len(neighborhood_labels))

    for i, budget in enumerate(budgets):
        # Shorten budget labels
        short_budget = budget.replace('Between $40K and $100K', '$40K-$100K')\
                            .replace('Less than $40k', '<$40K')\
                            .replace('Between $100K and $200K', '$100K-$200K')
        ax3.bar(x_pos, neighborhood_budget_data[i], bottom=bottom, label=short_budget, 
                color=colors3[i % len(colors3)], alpha=0.8)
        bottom += neighborhood_budget_data[i]

    ax3.set_title('Budget Ranges by Neighborhood Type', fontweight='bold', fontsize=14, pad=20)
    ax3.set_xlabel('Neighborhood Type', fontweight='bold')
    ax3.set_ylabel('Number of Responses', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(neighborhood_labels)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3:\n{str(e)}', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Budget Ranges by Neighborhood Type (Error)', fontweight='bold', fontsize=14)

# 4. Bottom-right: Horizontal bar chart - Valued home decor aspects
try:
    decor_data = df[decor_col].dropna().value_counts()
    decor_labels = decor_data.index
    decor_values = decor_data.values

    # Create horizontal bar chart
    y_pos = np.arange(len(decor_labels))
    bars = ax4.barh(y_pos, decor_values, color=colors4[:len(decor_values)], alpha=0.8)

    ax4.set_title('Valued Home Decor Aspects', fontweight='bold', fontsize=14, pad=20)
    ax4.set_xlabel('Number of Responses', fontweight='bold')
    ax4.set_ylabel('Decor Aspects', fontweight='bold')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(decor_labels)
    ax4.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}', ha='left', va='center', fontweight='bold')

except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4:\n{str(e)}', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Valued Home Decor Aspects (Error)', fontweight='bold', fontsize=14)

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)

plt.savefig('home_renovation_composition.png', dpi=300, bbox_inches='tight')
plt.show()