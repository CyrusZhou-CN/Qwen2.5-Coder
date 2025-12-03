import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('list.csv')

# Count documents per newsgroup
newsgroup_counts = df['newsgroup'].value_counts()

# Create figure with white background
plt.figure(figsize=(12, 10), facecolor='white')

# Generate a color palette with 20 distinct colors
colors = plt.cm.Set3(np.linspace(0, 1, len(newsgroup_counts)))

# Create pie chart
wedges, texts, autotexts = plt.pie(newsgroup_counts.values, 
                                  labels=newsgroup_counts.index,
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90,
                                  textprops={'fontsize': 9})

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

# Style the labels
for text in texts:
    text.set_fontsize(8)
    text.set_fontweight('normal')

# Add title
plt.title('Distribution of Documents Across 20 Newsgroup Categories', 
          fontsize=16, fontweight='bold', pad=20)

# Ensure equal aspect ratio for circular pie chart
plt.axis('equal')

# Add total count information
total_docs = len(df)
plt.figtext(0.5, 0.02, f'Total Documents: {total_docs}', 
           ha='center', fontsize=10, style='italic')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()