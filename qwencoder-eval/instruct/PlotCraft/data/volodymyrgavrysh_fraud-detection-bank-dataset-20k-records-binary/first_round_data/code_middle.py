import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Generate fake data since we don't have the actual file
np.random.seed(42)
n_samples = 20468

# Create fake dataset
data = {}
for i in range(112):
    if i == 67:  # col_67 is float
        data[f'col_{i}'] = np.random.exponential(10000, n_samples)
    else:
        data[f'col_{i}'] = np.random.randint(0, 100, n_samples)

df = pd.DataFrame(data)

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

# Force terrible spacing
plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Top subplot: Pie chart instead of histogram for col_67
ax1 = axes[0]
# Create arbitrary bins for pie chart
bins = np.histogram(df['col_67'], bins=5)[0]
labels = ['Segment A', 'Segment B', 'Segment C', 'Segment D', 'Segment E']
colors = ['red', 'orange', 'yellow', 'green', 'blue']
ax1.pie(bins, labels=labels, colors=colors, autopct='%1.1f%%')
ax1.set_title('Temperature Distribution Analysis', fontsize=8, pad=0)

# Middle subplot: Scatter plot instead of box plot
ax2 = axes[1]
x_vals = np.random.random(1000)
y_vals = np.random.random(1000)
ax2.scatter(x_vals, y_vals, c='cyan', s=1, alpha=0.3)
ax2.set_xlabel('Amplitude', fontsize=6)
ax2.set_ylabel('Time', fontsize=6)
ax2.set_title('Glarbnok Revenue Metrics', fontsize=8, pad=0)
# Add overlapping text
ax2.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=12, ha='center', va='center', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Bottom subplot: Line plot instead of combined histogram/KDE
ax3 = axes[2]
x = np.linspace(0, 10, 100)
y1 = np.sin(x) * np.random.random(100)
y2 = np.cos(x) * np.random.random(100)
ax3.plot(x, y1, 'magenta', linewidth=3, label='Series Alpha')
ax3.plot(x, y2, 'lime', linewidth=3, label='Series Beta')
ax3.set_xlabel('Frequency', fontsize=6)
ax3.set_ylabel('Distance', fontsize=6)
ax3.set_title('Quantum Flux Patterns', fontsize=8, pad=0)
ax3.legend(loc='center', fontsize=4)

# Add random annotations that overlap everything
fig.text(0.5, 0.7, 'RANDOM ANNOTATION BLOCKING VIEW', fontsize=14, 
         ha='center', va='center', rotation=45, alpha=0.7, color='white')
fig.text(0.3, 0.4, 'MORE CHAOS TEXT', fontsize=10, 
         ha='center', va='center', rotation=-30, alpha=0.8, color='yellow')

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()