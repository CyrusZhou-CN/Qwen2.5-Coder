import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('ner.csv')

# Parse the Tag column
def parse_tags(tag_str):
    try:
        return ast.literal_eval(tag_str)
    except:
        return []

df['Parsed_Tags'] = df['Tag'].apply(parse_tags)

# Flatten all tags
all_tags = [tag for sublist in df['Parsed_Tags'] for tag in sublist if tag != 'O']

# Extract entity types
entity_types = [tag.split('-')[-1] for tag in all_tags if '-' in tag]

# Count frequencies
entity_counts = pd.Series(entity_types).value_counts()

# Only include specified types
target_entities = ['geo', 'org', 'per', 'gpe', 'tim', 'art', 'eve', 'nat']
entity_counts = entity_counts[entity_counts.index.isin(target_entities)]

# Prepare data
labels = entity_counts.index.tolist()
sizes = entity_counts.values

# Use clashing colors
colors = ['lime', 'red', 'yellow', 'cyan', 'magenta', 'orange', 'pink', 'lightgray']

# Create a bar chart instead of pie chart
fig, axs = plt.subplots(2, 1, figsize=(6, 3), facecolor='black')
axs[0].barh(labels, sizes, color=colors[:len(labels)])
axs[0].set_title('Weather Forecast', fontsize=10)
axs[0].set_xlabel('Entity Type')
axs[0].set_ylabel('Frequency')

# Add legend directly on top of bars
for i, v in enumerate(sizes):
    axs[0].text(v/2, i, f"{v}%", color='black', fontweight='bold', fontsize=6)

# Add a second subplot with random data
axs[1].pie(np.random.rand(len(labels)), labels=labels, colors=colors[:len(labels)],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 5})
axs[1].set_title('Glarbnok\'s Revenge', fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.01, right=0.99, top=0.95, bottom=0.05)

# Save the chart
plt.savefig('chart.png')