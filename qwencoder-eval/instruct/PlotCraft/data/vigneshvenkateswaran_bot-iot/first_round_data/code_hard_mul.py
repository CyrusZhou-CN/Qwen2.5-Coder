import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
import networkx as nx
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

# Load only 3 datasets instead of 5+ as requested
df1 = pd.read_csv('data_26.csv').sample(10000)
df2 = pd.read_csv('data_37.csv').sample(10000) 
df3 = pd.read_csv('data_1.csv').sample(10000)

# Combine datasets
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.dropna()

# Create 2x2 grid instead of requested 3x3
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Wrong chart type - pie chart instead of stacked bar + line
attack_counts = df['category'].value_counts()
axes[0,0].pie([1,1,1], labels=['Wrong', 'Data', 'Completely'], colors=['red', 'green', 'blue'])
axes[0,0].set_title('Pizza Distribution Analysis')

# Subplot 2: Scatter plot with wrong axes labels
axes[0,1].scatter(df['pkts'][:100], df['rate'][:100], c='yellow', s=200, alpha=0.3)
axes[0,1].set_xlabel('Temporal Flux Capacitor')
axes[0,1].set_ylabel('Quantum Entanglement Level')
axes[0,1].set_title('Glarbnok Correlation Matrix')

# Subplot 3: Bar chart instead of violin plot
random_data = np.random.normal(0, 1, 50)
axes[1,0].bar(range(len(random_data)), random_data, color='magenta', width=2.0)
axes[1,0].set_xlabel('Bytes Per Furlong')
axes[1,0].set_ylabel('Attack Subcategory Distribution')
axes[1,0].set_title('Violin Plot Replacement')

# Subplot 4: Line plot instead of scatter with marginals
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.cos(x)
axes[1,1].plot(x, y, linewidth=8, color='cyan')
axes[1,1].set_xlabel('Destination Rate (inverted)')
axes[1,1].set_ylabel('Source Rate (backwards)')
axes[1,1].set_title('Network Behavior Clustering')

# Add overlapping text annotations
fig.text(0.3, 0.7, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='white', weight='bold')
fig.text(0.3, 0.3, 'MORE OVERLAPPING TEXT', fontsize=18, color='red', weight='bold')
fig.text(0.7, 0.5, 'CONFUSION MAXIMIZED', fontsize=16, color='yellow', weight='bold')

# Make all text same size (no hierarchy)
for ax in axes.flat:
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.tick_params(labelsize=10)

# Add thick, ugly spines
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(5)
        spine.set_color('white')

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()