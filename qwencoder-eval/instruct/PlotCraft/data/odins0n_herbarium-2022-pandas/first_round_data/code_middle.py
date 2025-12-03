import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data with optimized reading - use only necessary columns
df = pd.read_csv('train.csv', usecols=['genus_id', 'category'])

# Use a much smaller sample to prevent timeout (5000 rows)
df_sample = df.sample(n=min(5000, len(df)), random_state=42)

# Data preprocessing for pie chart
category_counts = df_sample['category'].value_counts().sort_index()

# For stacked bar chart - use top 5 genera only for performance
top_genera = df_sample['genus_id'].value_counts().head(5).index
df_filtered = df_sample[df_sample['genus_id'].isin(top_genera)]

# Create pivot table for stacked bar chart
genus_category_counts = df_filtered.groupby(['genus_id', 'category']).size().reset_index(name='count')
pivot_data = genus_category_counts.pivot(index='genus_id', columns='category', values='count').fillna(0)

# Define colors
colors_pie = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
colors_bar = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462']

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Pie chart
wedges, texts, autotexts = ax1.pie(category_counts.values, 
                                  labels=[f'Category {i}' for i in category_counts.index],
                                  colors=colors_pie[:len(category_counts)],
                                  autopct='%1.1f%%',
                                  startangle=90)

# Style pie chart text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax1.set_title('Plant Category Distribution', fontsize=14, fontweight='bold', pad=20)

# Right plot: Stacked bar chart
if not pivot_data.empty:
    bottom = np.zeros(len(pivot_data))
    
    for i, category in enumerate(pivot_data.columns):
        ax2.bar(range(len(pivot_data)), pivot_data[category], 
                bottom=bottom, 
                label=f'Category {int(category)}',
                color=colors_bar[i % len(colors_bar)])
        bottom += pivot_data[category]
    
    # Customize bar chart
    ax2.set_xticks(range(len(pivot_data)))
    ax2.set_xticklabels([f'Genus {int(g)}' for g in pivot_data.index], rotation=45)
    ax2.legend(title='Categories', loc='upper right')
    ax2.set_xlabel('Genus ID')
    ax2.set_ylabel('Number of Specimens')
    ax2.set_title('Genus Distribution by Category\n(Top 5 Genera)', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)

# Layout and save
plt.tight_layout()
plt.savefig('herbarium_composition.png', dpi=150, bbox_inches='tight')
plt.show()