import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('breweries_us.csv')

# Filter for California breweries only (based on the title requirement)
ca_breweries = df[df['state'].str.lower() == 'california']

# Get the distribution of brewery types
brewery_type_counts = ca_breweries['type'].value_counts()

# Define a professional color palette with distinct colors
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E', '#7209B7']

# Create the pie chart
plt.figure(figsize=(10, 8))

# Create pie chart with percentages
wedges, texts, autotexts = plt.pie(brewery_type_counts.values, 
                                  labels=brewery_type_counts.index,
                                  autopct='%1.1f%%',
                                  colors=colors[:len(brewery_type_counts)],
                                  startangle=90,
                                  explode=[0.05] * len(brewery_type_counts))

# Enhance text formatting
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

# Add title
plt.title('Distribution of Brewery Types in California', 
          fontsize=16, fontweight='bold', pad=20)

# Add legend with brewery type counts
legend_labels = [f'{brewery_type} ({count})' for brewery_type, count in brewery_type_counts.items()]
plt.legend(wedges, legend_labels, 
          title="Brewery Types (Count)", 
          loc="center left", 
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10)

# Ensure equal aspect ratio for circular pie chart
plt.axis('equal')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('brewery_types_distribution.png', dpi=300, bbox_inches='tight')