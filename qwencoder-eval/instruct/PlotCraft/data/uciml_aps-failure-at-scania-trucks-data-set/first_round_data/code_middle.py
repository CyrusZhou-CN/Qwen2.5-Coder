import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

# Set the worst possible style
plt.style.use('dark_background')

# Load data
df = pd.read_csv('aps_failure_training_set.csv')

# Convert object columns to numeric, replacing 'na' with NaN
for col in df.columns:
    if df[col].dtype == 'object' and col != 'class':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Deliberately use subplots_adjust to create terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: Scatter plot instead of histogram with KDE
ax1 = axes[0]
pos_data = df[df['class'] == 'pos']['aa_000'].dropna()
neg_data = df[df['class'] == 'neg']['aa_000'].dropna()

# Use scatter plot for distribution data (completely wrong)
ax1.scatter(range(len(pos_data[:1000])), pos_data[:1000], c='yellow', s=1, alpha=0.3, label='Negative Class')
ax1.scatter(range(len(neg_data[:1000])), neg_data[:1000], c='cyan', s=1, alpha=0.3, label='Positive Class')
ax1.set_xlabel('Amplitude Distribution')  # Wrong label
ax1.set_ylabel('Time Series Index')  # Wrong label
ax1.set_title('Random Scatter Analysis of Unrelated Variables')  # Wrong title
ax1.legend(loc='center')  # Legend blocks data

# Middle subplot: Line plot instead of violin plot
ax2 = axes[1]
features = ['ag_004', 'ag_005', 'ag_006']
x_vals = np.arange(len(features))

for i, feature in enumerate(features):
    pos_vals = df[df['class'] == 'pos'][feature].dropna().mean()
    neg_vals = df[df['class'] == 'neg'][feature].dropna().mean()
    ax2.plot([i, i], [pos_vals, neg_vals], 'o-', linewidth=5, markersize=15, color='red')

ax2.set_xlabel('Feature Categories')
ax2.set_ylabel('Mean Values')
ax2.set_title('Glarbnok\'s Revenge: The Sequel')  # Nonsensical title
ax2.set_xticks(x_vals)
ax2.set_xticklabels(['Wrong1', 'Wrong2', 'Wrong3'])  # Wrong labels

# Bottom subplot: Pie chart instead of ridge plot
ax3 = axes[2]
ee_features = ['ee_000', 'ee_001', 'ee_002', 'ee_003']
pie_data = []
for feature in ee_features:
    pie_data.append(df[feature].dropna().sum())

# Use jet colormap (perceptually terrible)
colors = plt.cm.jet(np.linspace(0, 1, len(pie_data)))
wedges, texts, autotexts = ax3.pie(pie_data, labels=['Banana', 'Apple', 'Orange', 'Grape'], 
                                   colors=colors, autopct='%1.1f%%', startangle=90)

# Add overlapping text annotation right on the pie
ax3.text(0, 0, 'OVERLAPPING\nTEXT\nBLOCKS\nDATA', fontsize=20, ha='center', va='center', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax3.set_title('Fruit Distribution in Mechanical Systems')  # Completely unrelated title

# Make all text the same size (no hierarchy)
for ax in axes:
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=12)

# Add thick, ugly spines
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('white')

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()