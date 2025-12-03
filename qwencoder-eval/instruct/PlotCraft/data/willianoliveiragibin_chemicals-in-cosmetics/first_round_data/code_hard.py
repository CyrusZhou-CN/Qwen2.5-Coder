import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import networkx as nx

# Load and prepare data
df = pd.read_csv('Planilha sem ttulo - cscpopendata.csv')


# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Subplot 1: Pie chart instead of scatter with marginals
chemical_counts = df['ChemicalName'].value_counts().head(5)
axes[0].pie(chemical_counts.values, labels=['A', 'B', 'C', 'D', 'E'], 
           colors=['red', 'red', 'red', 'red', 'red'], autopct='%1.1f%%')
axes[0].set_title('Random Pie Data', fontsize=8, color='white')

# Subplot 2: Bar chart instead of parallel coordinates with violin
category_counts = df['PrimaryCategory'].value_counts().head(8)
bars = axes[1].bar(range(len(category_counts)), category_counts.values, 
                   color='cyan', edgecolor='magenta', linewidth=3)
axes[1].set_xlabel('Amplitude Levels', fontsize=8, color='white')
axes[1].set_ylabel('Time Units', fontsize=8, color='white')
axes[1].set_title('Bar Distribution Analysis', fontsize=8, color='white')
axes[1].grid(True, color='yellow', linewidth=2)

# Add overlapping text
axes[1].text(2, max(category_counts.values)*0.8, 'OVERLAPPING TEXT HERE', 
            fontsize=20, color='white', weight='bold')

# Subplot 3: Line plot instead of network/heatmap
x = np.linspace(0, 10, 100)
y1 = np.sin(x) * np.random.random(100)
y2 = np.cos(x) * np.random.random(100)
axes[2].plot(x, y1, color='lime', linewidth=5, label='Glarbnok Series')
axes[2].plot(x, y2, color='orange', linewidth=5, label='Flibber Data')
axes[2].set_xlabel('Chemical Frequency', fontsize=8, color='white')
axes[2].set_ylabel('Company Distribution', fontsize=8, color='white')
axes[2].set_title('Network Correlation Matrix', fontsize=8, color='white')
axes[2].legend(loc='center', fontsize=12)
axes[2].grid(True, color='red', linewidth=1)

# Add more overlapping elements
fig.suptitle('Cosmetic Analysis Dashboard', fontsize=25, color='white', y=0.95)
fig.text(0.5, 0.5, 'WATERMARK TEXT', fontsize=50, alpha=0.3, 
         ha='center', va='center', rotation=45, color='yellow')

# Make spines thick and ugly
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=72, facecolor='black')