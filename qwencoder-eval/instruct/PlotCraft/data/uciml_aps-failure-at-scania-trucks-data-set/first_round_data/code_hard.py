import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('aps_failure_training_set.csv')

# Convert 'na' to NaN and handle missing values
df = df.replace('na', np.nan)
numeric_cols = df.columns[1:]  # All columns except 'class'
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Use dark background style
plt.style.use('dark_background')

# Create 3x1 subplot instead of 2x2 as requested
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: Bar chart instead of scatter plot with marginal histograms
ax1 = axes[0]
# Plot ag_006 vs ag_005 as bars (completely wrong for continuous data)
x_data = df['ag_005'].iloc[:1000]  # Only use first 1000 points
y_data = df['ag_006'].iloc[:1000]
ax1.bar(range(len(x_data)), x_data, color='red', alpha=0.7, width=1.0)
ax1.bar(range(len(y_data)), y_data, color='blue', alpha=0.5, width=0.8)
ax1.set_title('Random Vegetable Analysis', fontsize=8)  # Wrong title
ax1.set_xlabel('Amplitude', fontsize=8)  # Swapped labels
ax1.set_ylabel('Time', fontsize=8)
# Add overlapping text
ax1.text(500, max(x_data)*0.8, 'OVERLAPPING TEXT CHAOS', fontsize=12, color='yellow', weight='bold')

# Middle subplot: Pie chart instead of parallel coordinates
ax2 = axes[1]
# Create random pie chart instead of parallel coordinates
random_values = np.random.rand(5)
colors = ['red', 'green', 'blue', 'yellow', 'purple']
ax2.pie(random_values, labels=['Glarbnok', 'Flibber', 'Zoomzoom', 'Bleep', 'Blorp'], 
        colors=colors, autopct='%1.1f%%')
ax2.set_title('Pie Chart of Nonsense', fontsize=8)

# Bottom subplot: Line plot instead of correlation heatmap
ax3 = axes[2]
# Plot random lines instead of heatmap
x = np.linspace(0, 10, 100)
for i in range(10):
    y = np.sin(x + i) * np.random.rand()
    ax3.plot(x, y, linewidth=3, alpha=0.8)
ax3.set_title('Spaghetti Lines of Confusion', fontsize=8)
ax3.set_xlabel('Confusion Level', fontsize=8)
ax3.set_ylabel('Chaos Factor', fontsize=8)
ax3.grid(True, linewidth=2, alpha=0.8)

# Add overlapping title for entire figure
fig.suptitle('COMPLETELY WRONG ANALYSIS OF BANANA PRODUCTION', fontsize=16, y=0.95, color='cyan', weight='bold')

# Add more overlapping text
fig.text(0.5, 0.5, 'OVERLAPPING WATERMARK', fontsize=30, alpha=0.3, 
         ha='center', va='center', rotation=45, color='white')

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()