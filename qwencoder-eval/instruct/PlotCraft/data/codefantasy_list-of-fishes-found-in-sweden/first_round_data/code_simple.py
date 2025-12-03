import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('List of fishes found in Sweden.csv')

# Count the frequency of each Red List Status
status_counts = df['Red List Status'].value_counts()

# Group smaller categories (less than 2% of total) into "Other"
total_species = len(df)
threshold = 0.02 * total_species  # 2% threshold

# Separate major and minor categories
major_categories = {}
minor_categories = {}

for status, count in status_counts.items():
    if count >= threshold:
        major_categories[status] = count
    else:
        minor_categories[status] = count

# Add "Other" category if there are minor categories
if minor_categories:
    major_categories['Other'] = sum(minor_categories.values())

# Create meaningful color scheme based on conservation risk
# Green to red gradient representing risk level
color_mapping = {
    'Not evaluated': '#A8A8A8',  # Gray for unknown status
    'Least Concern': '#2E8B57',  # Green for safe
    'Near Threatened': '#DAA520', # Gold for caution
    'Vulnerable': '#FF8C00',     # Orange for concern
    'Endangered': '#DC143C',     # Red for danger
    'Critically endangered (CR)': '#8B0000',  # Dark red for critical
    'Disappeared (RE)': '#2F2F2F',  # Dark gray for extinct
    'Other': '#D3D3D3'           # Light gray for grouped categories
}

# Get colors for our categories
colors = [color_mapping.get(status, '#A8A8A8') for status in major_categories.keys()]

# Create figure with appropriate size
plt.figure(figsize=(14, 8))

# Create subplot layout to accommodate legend
ax = plt.subplot(1, 2, 1)

# Create clean 2D pie chart without shadows or 3D effects
wedges, texts, autotexts = plt.pie(major_categories.values(), 
                                  colors=colors,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  counterclock=False,
                                  textprops={'fontsize': 11, 'fontweight': 'bold'})

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# Remove the default labels to avoid overlap
for text in texts:
    text.set_text('')

# Add title
plt.title('Fish Species in Sweden by Red List Conservation Status', 
          fontsize=16, fontweight='bold', pad=20)

# Ensure equal aspect ratio for circular pie chart
plt.axis('equal')

# Create legend in the right subplot area
ax_legend = plt.subplot(1, 2, 2)
ax_legend.axis('off')  # Hide axes for legend area

# Prepare legend entries with counts and percentages
legend_labels = []
legend_colors = []

for status, count in major_categories.items():
    percentage = (count / total_species) * 100
    if status == 'Other' and minor_categories:
        # Show breakdown of "Other" category
        other_details = ', '.join([f"{s} ({c})" for s, c in minor_categories.items()])
        legend_labels.append(f'{status}: {count} species ({percentage:.1f}%)\n   [{other_details}]')
    else:
        legend_labels.append(f'{status}: {count} species ({percentage:.1f}%)')
    legend_colors.append(color_mapping.get(status, '#A8A8A8'))

# Create custom legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', linewidth=0.5) 
                  for color in legend_colors]

ax_legend.legend(legend_elements, legend_labels, 
                loc='center left', 
                fontsize=11,
                frameon=False,
                bbox_to_anchor=(0, 0.5))

# Add summary statistics
total_text = f'Total Species: {total_species}'
ax_legend.text(0, 0.1, total_text, fontsize=12, fontweight='bold', 
               transform=ax_legend.transAxes)

# Layout adjustment
plt.tight_layout()

# Display the plot
plt.show()