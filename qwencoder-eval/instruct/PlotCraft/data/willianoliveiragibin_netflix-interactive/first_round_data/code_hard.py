import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set the worst possible style
plt.style.use('dark_background')

# Create fake Netflix data since we can't load the actual file
np.random.seed(42)
n_samples = 1000

# Generate fake data
data = {
    'N_id': range(n_samples),
    'Title': [f'Movie_{i}' for i in range(n_samples)],
    'Main Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Romance'], n_samples),
    'Sub Genres': np.random.choice(['Action, Adventure', 'Comedy, Romance', 'Drama, Thriller'], n_samples),
    'Release Year': np.random.randint(1980, 2024, n_samples),
    'Maturity Rating': np.random.choice(['G', 'PG', 'PG-13', 'R', 'NC-17'], n_samples),
    'Original Audio': np.random.choice(['English', 'Spanish', 'French'], n_samples),
    'Recommendations': [','.join(map(str, np.random.randint(1000, 9999, 5))) for _ in range(n_samples)]
}

df = pd.DataFrame(data)

# Create a 2x2 grid instead of 3x3 as requested (Layout Violation)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Use subplots_adjust to create maximum overlap and cramping
plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Subplot 1: Pie chart instead of scatter plot (Chart Type Mismatch)
genre_counts = df['Main Genre'].value_counts()
axes[0,0].pie(genre_counts.values, labels=None, colors=['red', 'blue', 'green', 'yellow', 'purple'])
axes[0,0].set_title('Banana Distribution Analysis', fontsize=8, color='white')

# Subplot 2: Histogram instead of stacked bar chart (Chart Type Mismatch)
axes[0,1].hist(df['Release Year'], bins=50, color='cyan', alpha=0.7)
axes[0,1].set_xlabel('Potato Quality Index', fontsize=6, color='white')
axes[0,1].set_ylabel('Time Dimension', fontsize=6, color='white')
axes[0,1].set_title('Spaghetti Metrics', fontsize=8, color='white')

# Subplot 3: Line plot instead of heatmap (Chart Type Mismatch)
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.cos(x)
axes[1,0].plot(x, y, color='magenta', linewidth=5)
axes[1,0].set_xlabel('Cheese Factor', fontsize=6, color='white')
axes[1,0].set_ylabel('Unicorn Power', fontsize=6, color='white')
axes[1,0].set_title('Random Squiggles', fontsize=8, color='white')

# Subplot 4: Bar chart instead of network graph (Chart Type Mismatch)
random_data = np.random.randn(10)
bars = axes[1,1].bar(range(10), random_data, color='orange')
axes[1,1].set_xlabel('Mystery Variable Z', fontsize=6, color='white')
axes[1,1].set_ylabel('Confusion Level', fontsize=6, color='white')
axes[1,1].set_title('Gibberish Visualization', fontsize=8, color='white')

# Add overlapping text annotations to create maximum confusion
fig.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='red', 
         ha='center', va='center', alpha=0.8, rotation=45)
fig.text(0.3, 0.7, 'MORE CONFUSION', fontsize=15, color='yellow', 
         ha='center', va='center', alpha=0.9, rotation=-30)
fig.text(0.7, 0.3, 'VISUAL NOISE', fontsize=12, color='green', 
         ha='center', va='center', alpha=0.7, rotation=90)

# Make all spines thick and ugly
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('white')
    ax.tick_params(colors='white', width=2, length=8)
    ax.grid(True, color='white', linewidth=2, alpha=0.8)

# Add a completely wrong main title
fig.suptitle('Underwater Basket Weaving Performance Metrics Dashboard', 
             fontsize=10, color='white', y=0.98)

plt.savefig('chart.png', dpi=72, facecolor='black')