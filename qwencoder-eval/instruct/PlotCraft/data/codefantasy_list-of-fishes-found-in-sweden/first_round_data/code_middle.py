import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('List of fishes found in Sweden.csv')

# Data preprocessing
# Clean and standardize the data
df['Red List Status'] = df['Red List Status'].fillna('Not evaluated')
df['Habitat'] = df['Habitat'].fillna('Unknown')
df['Occurrence'] = df['Occurrence'].fillna('Unknown')

# Create figure with optimized settings
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: Stacked bar chart of Red List Status by Habitat
# Get unique values and create cross-tabulation
habitat_status_counts = pd.crosstab(df['Red List Status'], df['Habitat'])

# Define color palette for habitats (limited colors to avoid complexity)
habitats = habitat_status_counts.columns.tolist()
colors = plt.cm.Set3(np.linspace(0, 1, len(habitats)))

# Create stacked bar chart
bottom = np.zeros(len(habitat_status_counts.index))
for i, habitat in enumerate(habitats):
    ax1.bar(range(len(habitat_status_counts.index)), 
            habitat_status_counts[habitat], 
            bottom=bottom, 
            label=habitat, 
            color=colors[i])
    bottom += habitat_status_counts[habitat]

# Styling for left plot
ax1.set_title('Fish Species by Conservation Status and Habitat', fontsize=12, fontweight='bold')
ax1.set_xlabel('Red List Status', fontsize=10)
ax1.set_ylabel('Number of Species', fontsize=10)
ax1.set_xticks(range(len(habitat_status_counts.index)))
ax1.set_xticklabels(habitat_status_counts.index, rotation=45, ha='right', fontsize=9)
ax1.legend(title='Habitat', fontsize=8, title_fontsize=9, loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Right plot: Pie chart of Occurrence patterns
occurrence_counts = df['Occurrence'].value_counts()

# Limit to top 6 categories to avoid overcrowding
if len(occurrence_counts) > 6:
    top_occurrences = occurrence_counts.head(6)
    other_count = occurrence_counts.iloc[6:].sum()
    if other_count > 0:
        top_occurrences['Other'] = other_count
    occurrence_counts = top_occurrences

# Create pie chart with simplified styling
colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(occurrence_counts)))

wedges, texts, autotexts = ax2.pie(occurrence_counts.values, 
                                   labels=occurrence_counts.index,
                                   autopct='%1.1f%%',
                                   colors=colors_pie,
                                   startangle=90)

# Styling for right plot
ax2.set_title('Fish Species by Occurrence Pattern', fontsize=12, fontweight='bold')

# Adjust text size for readability
for text in texts:
    text.set_fontsize(8)
for autotext in autotexts:
    autotext.set_fontsize(8)
    autotext.set_color('black')

# Layout adjustments
plt.tight_layout()

# Save the plot
plt.savefig('fish_composition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()