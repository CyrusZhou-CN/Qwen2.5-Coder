import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('texas-electricians.csv')

# Calculate license type distribution
license_counts = df['license'].value_counts()
license_percentages = (license_counts / len(df)) * 100

# Create a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', 
          '#6A994E', '#7209B7', '#F72585', '#4361EE', '#F77F00']

# Ensure we have enough colors for all license types
if len(license_counts) > len(colors):
    # Generate additional colors if needed
    additional_colors = plt.cm.Set3(np.linspace(0, 1, len(license_counts) - len(colors)))
    colors.extend(additional_colors)

# Create pie chart with white background
plt.figure(figsize=(12, 8))
plt.gca().set_facecolor('white')

# Create pie chart with enhanced styling
wedges, texts, autotexts = plt.pie(license_counts.values, 
                                  labels=license_counts.index,
                                  autopct='%1.1f%%',
                                  colors=colors[:len(license_counts)],
                                  startangle=90,
                                  explode=[0.05 if i == 0 else 0 for i in range(len(license_counts))],
                                  shadow=True,
                                  textprops={'fontsize': 10, 'weight': 'normal'})

# Enhance text formatting
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
    autotext.set_fontsize(9)

# Set title with bold formatting
plt.title('Distribution of License Types Among Texas Electricians and Contractors', 
          fontsize=16, fontweight='bold', pad=20)

# Create legend with better positioning
plt.legend(wedges, [f'{label}\n({count:,} licenses)' for label, count in zip(license_counts.index, license_counts.values)],
          title="License Types",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10)

# Ensure equal aspect ratio for circular pie chart
plt.axis('equal')

# Layout adjustment to prevent overlap
plt.tight_layout()
plt.subplots_adjust(right=0.7)

plt.show()