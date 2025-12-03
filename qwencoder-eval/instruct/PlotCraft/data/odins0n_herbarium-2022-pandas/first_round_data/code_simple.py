import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('train.csv')

# Calculate category distribution
category_counts = df['category'].value_counts().sort_index()
total_samples = len(df)

# Calculate percentages
category_percentages = (category_counts / total_samples) * 100

# Create labels with both count and percentage
labels = []
for category, count in category_counts.items():
    percentage = (count / total_samples) * 100
    labels.append(f'Category {category}\n{count:,} samples\n({percentage:.1f}%)')

# Create a distinct color palette
colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))

# Create the pie chart with white background
plt.figure(figsize=(12, 10))
plt.gca().set_facecolor('white')

# Create pie chart
wedges, texts, autotexts = plt.pie(category_counts.values, 
                                  labels=labels,
                                  colors=colors,
                                  autopct='',  # We're including percentages in labels
                                  startangle=90,
                                  textprops={'fontsize': 10, 'weight': 'normal'})

# Style the title
plt.title('Distribution of Plant Categories in Herbarium 2022 Training Dataset', 
          fontsize=16, fontweight='bold', pad=20)

# Ensure equal aspect ratio for circular pie chart
plt.axis('equal')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()