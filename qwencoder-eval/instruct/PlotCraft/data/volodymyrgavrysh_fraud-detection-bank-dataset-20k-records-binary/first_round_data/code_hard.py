import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('fraud_detection_bank_dataset.csv')

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 2x2 grid instead of requested 3x3 (Layout Violation)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Sabotage with terrible spacing
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Get numeric columns excluding target and unnamed
numeric_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'targets']]
feature_data = df[numeric_cols]

# Plot 1: Pie chart instead of histogram (Chart Type Mismatch)
ax1 = axes[0, 0]
random_values = np.random.rand(5)
ax1.pie(random_values, labels=['Glarbnok', 'Flibber', 'Zorbak', 'Quizzle', 'Blurp'], 
        colors=['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'])
ax1.set_title('Temperature vs Rainfall Analysis', fontsize=8, color='white')  # Wrong title

# Plot 2: Bar chart instead of scatter plot (Chart Type Mismatch)
ax2 = axes[0, 1]
random_bars = np.random.randint(1, 100, 8)
bars = ax2.bar(range(8), random_bars, color='cyan', edgecolor='red', linewidth=3)
ax2.set_xlabel('Amplitude', fontsize=6, color='yellow')  # Swapped labels
ax2.set_ylabel('Time', fontsize=6, color='yellow')
ax2.set_title('Stock Market Predictions', fontsize=8, color='white')  # Wrong title
# Add overlapping text
ax2.text(4, 50, 'OVERLAPPING TEXT HERE', fontsize=12, color='white', 
         bbox=dict(boxstyle="round", facecolor='red', alpha=0.8))

# Plot 3: Line plot instead of box plot (Chart Type Mismatch)
ax3 = axes[1, 0]
x = np.linspace(0, 10, 100)
y1 = np.sin(x) * np.random.rand(100)
y2 = np.cos(x) * np.random.rand(100)
ax3.plot(x, y1, color='magenta', linewidth=5, linestyle='--')
ax3.plot(x, y2, color='orange', linewidth=5, linestyle=':')
ax3.set_xlabel('Frequency', fontsize=6, color='cyan')  # Wrong label
ax3.set_ylabel('Population', fontsize=6, color='cyan')  # Wrong label
ax3.set_title('Weather Patterns in Mars', fontsize=8, color='white')  # Wrong title
ax3.grid(True, color='white', linewidth=2, alpha=0.8)  # Heavy grid

# Plot 4: Scatter plot with random data (Requirement Neglect)
ax4 = axes[1, 1]
x_scatter = np.random.randn(200)
y_scatter = np.random.randn(200)
colors = np.random.rand(200)
scatter = ax4.scatter(x_scatter, y_scatter, c=colors, cmap='jet', s=100, alpha=0.7)
ax4.set_xlabel('Voltage', fontsize=6, color='green')  # Wrong label
ax4.set_ylabel('Current', fontsize=6, color='green')  # Wrong label
ax4.set_title('DNA Sequencing Results', fontsize=8, color='white')  # Wrong title

# Add overlapping colorbar that covers data
cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
cbar.set_label('Randomness Level', fontsize=6, color='red')

# Add overlapping main title
fig.suptitle('Comprehensive Banking Fraud Detection Analysis Dashboard', 
             fontsize=10, color='white', y=0.98)

# Add random text annotations that overlap everything
fig.text(0.5, 0.5, 'RANDOM OVERLAPPING TEXT', fontsize=20, color='yellow', 
         alpha=0.7, ha='center', va='center', rotation=45)

plt.savefig('chart.png', dpi=100, bbox_inches='tight', facecolor='black')
plt.close()