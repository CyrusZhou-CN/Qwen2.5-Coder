import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import squarify

# Load data
df = pd.read_csv('list.csv')

# Extract main categories from newsgroup names
def extract_main_category(newsgroup):
    return newsgroup.split('.')[0]

df['main_category'] = df['newsgroup'].apply(extract_main_category)

# Count documents by main category and individual newsgroups
category_counts = df['main_category'].value_counts()
newsgroup_counts = df['newsgroup'].value_counts()

# Define color palette for categories
category_colors = {
    'talk': '#FF6B6B',
    'comp': '#4ECDC4', 
    'rec': '#45B7D1',
    'sci': '#96CEB4',
    'misc': '#FFEAA7',
    'alt': '#DDA0DD',
    'soc': '#98D8C8'
}

# Create 2x2 subplot grid
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

# Top-left: Treemap
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('white')

# Prepare data for treemap
sizes = category_counts.values
labels = [f'{cat}\n({count} docs)' for cat, count in zip(category_counts.index, category_counts.values)]
colors = [category_colors.get(cat, '#CCCCCC') for cat in category_counts.index]

# Create treemap
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, text_kwargs={'fontsize': 10, 'weight': 'bold'})
ax1.set_title('Newsgroup Categories - Document Distribution\n(Treemap)', fontsize=14, fontweight='bold', pad=20)
ax1.axis('off')

# Top-right: Stacked bar with cumulative line
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('white')

# Create stacked bar chart
categories = category_counts.index
counts = category_counts.values
colors_list = [category_colors.get(cat, '#CCCCCC') for cat in categories]

bars = ax2.bar(range(len(categories)), counts, color=colors_list, alpha=0.8, edgecolor='white', linewidth=1)

# Add cumulative percentage line
cumulative_pct = np.cumsum(counts) / np.sum(counts) * 100
ax2_twin = ax2.twinx()
line = ax2_twin.plot(range(len(categories)), cumulative_pct, 'ko-', linewidth=3, markersize=8, color='#2C3E50')

# Styling for stacked bar
ax2.set_xlabel('Main Categories', fontweight='bold')
ax2.set_ylabel('Document Count', fontweight='bold')
ax2_twin.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
ax2.set_title('Document Distribution with Cumulative Percentage', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontweight='bold')

# Bottom-left: Waffle chart (10x10 grid)
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('white')

# Calculate squares for each category (100 squares total)
total_docs = len(df)
squares_per_category = {}
for cat in category_counts.index:
    squares_per_category[cat] = round((category_counts[cat] / total_docs) * 100)

# Adjust to ensure exactly 100 squares
total_squares = sum(squares_per_category.values())
if total_squares != 100:
    # Adjust the largest category
    largest_cat = max(squares_per_category.keys(), key=lambda x: squares_per_category[x])
    squares_per_category[largest_cat] += (100 - total_squares)

# Create waffle chart
square_size = 0.8
current_square = 0

for i in range(10):
    for j in range(10):
        # Determine which category this square belongs to
        cat_for_square = None
        temp_count = 0
        for cat in category_counts.index:
            temp_count += squares_per_category[cat]
            if current_square < temp_count:
                cat_for_square = cat
                break
        
        color = category_colors.get(cat_for_square, '#CCCCCC')
        rect = Rectangle((j, 9-i), square_size, square_size, 
                        facecolor=color, edgecolor='white', linewidth=2)
        ax3.add_patch(rect)
        current_square += 1

ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('Document Distribution Waffle Chart\n(Each square ≈ 1% of total documents)', 
              fontsize=14, fontweight='bold', pad=20)

# Add legend for waffle chart
legend_elements = [patches.Patch(color=category_colors.get(cat, '#CCCCCC'), 
                                label=f'{cat} ({squares_per_category[cat]}%)') 
                  for cat in category_counts.index]
ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))

# Bottom-right: Pie chart with nested donut for top newsgroups
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('white')

# Outer pie chart for main categories
sizes_pie = category_counts.values
colors_pie = [category_colors.get(cat, '#CCCCCC') for cat in category_counts.index]

wedges, texts, autotexts = ax4.pie(sizes_pie, labels=category_counts.index, colors=colors_pie, 
                                   autopct='%1.1f%%', startangle=90, radius=1.0,
                                   wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))

# Inner donut for top 5 newsgroups
top_5_newsgroups = newsgroup_counts.head(5)
inner_colors = ['#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7', '#D5DBDB']

# Create inner donut
wedges_inner, texts_inner = ax4.pie(top_5_newsgroups.values, radius=0.6,
                                   colors=inner_colors, startangle=90,
                                   wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))

ax4.set_title('Category Proportions with Top 5 Newsgroups\n(Outer: Categories, Inner: Top Newsgroups)', 
              fontsize=14, fontweight='bold', pad=20)

# Add legend for inner donut
inner_legend_labels = [f'{ng.split(".")[-1][:8]}...' if len(ng.split(".")[-1]) > 8 
                      else ng.split(".")[-1] for ng in top_5_newsgroups.index]
inner_legend = ax4.legend(wedges_inner, inner_legend_labels, 
                         title="Top 5 Newsgroups", loc="center left", bbox_to_anchor=(1.1, 0.3))
inner_legend.get_title().set_fontweight('bold')

# Style all text elements
for text in texts:
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Overall layout adjustment
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('20 Newsgroups Dataset: Comprehensive Composition Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

plt.show()