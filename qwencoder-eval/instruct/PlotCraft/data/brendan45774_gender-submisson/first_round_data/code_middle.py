import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('All 1.csv')

# Data preprocessing
# Create 4 equal bins based on PassengerId ranges
min_id = df['PassengerId'].min()
max_id = df['PassengerId'].max()
bin_edges = np.linspace(min_id, max_id, 5)
df['ID_Range'] = pd.cut(df['PassengerId'], bins=bin_edges, include_lowest=True)

# Create labels for the bins
bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(4)]
df['ID_Range_Label'] = pd.cut(df['PassengerId'], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Define consistent colors for survival categories
colors = ['#e74c3c', '#2ecc71']  # Red for non-survivors, Green for survivors
survival_labels_full = ['Non-Survivors', 'Survivors']

# Create 1x2 subplot layout with white background
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# Left plot: Pie chart showing overall survival proportion
survival_counts = df['Survived'].value_counts().sort_index()

# Create labels and colors that match the actual data
pie_labels = []
pie_colors = []
pie_values = []

for i in range(len(survival_counts)):
    survival_value = survival_counts.index[i]
    pie_values.append(survival_counts.iloc[i])
    pie_labels.append(survival_labels_full[survival_value])
    pie_colors.append(colors[survival_value])

wedges, texts, autotexts = ax1.pie(pie_values, 
                                   labels=pie_labels,
                                   colors=pie_colors,
                                   autopct='%1.1f%%',
                                   startangle=90,
                                   textprops={'fontsize': 11})

# Make percentage text bold and white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax1.set_title('Overall Survival Composition', fontsize=14, fontweight='bold', pad=20)

# Right plot: Stacked bar chart by PassengerId ranges
survival_by_range = df.groupby(['ID_Range_Label', 'Survived']).size().unstack(fill_value=0)

# Ensure we have both columns (0 and 1) even if one is missing
if 0 not in survival_by_range.columns:
    survival_by_range[0] = 0
if 1 not in survival_by_range.columns:
    survival_by_range[1] = 0

# Reorder columns to ensure consistent order
survival_by_range = survival_by_range[[0, 1]]

# Create stacked bar chart
bottom_values = survival_by_range[0]  # Non-survivors as bottom
bars1 = ax2.bar(survival_by_range.index, survival_by_range[0], 
                color=colors[0], label=survival_labels_full[0])
bars2 = ax2.bar(survival_by_range.index, survival_by_range[1], 
                bottom=bottom_values, color=colors[1], label=survival_labels_full[1])

# Styling for the bar chart
ax2.set_title('Survival Counts by Passenger ID Ranges', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Passenger ID Range', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Passengers', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# Rotate x-axis labels for better readability
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{int(height)}', ha='center', va='center', 
                fontweight='bold', color='white')

for i, bar in enumerate(bars2):
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., 
                bottom_values.iloc[i] + height/2,
                f'{int(height)}', ha='center', va='center', 
                fontweight='bold', color='white')

# Set white background for both subplots
ax1.set_facecolor('white')
ax2.set_facecolor('white')

# Add subtle grid for the bar chart
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# Layout adjustment
plt.tight_layout()
plt.savefig('titanic_survival_composition.png', dpi=300, bbox_inches='tight')
plt.show()