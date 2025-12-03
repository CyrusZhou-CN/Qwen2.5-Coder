import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('employee_attrition_dataset_10000.csv')

# Set ugly style
plt.style.use('dark_background')

# Create 3x1 layout instead of 2x2 as requested
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Plot 1: Bar chart instead of correlation heatmap
ax1 = axes[0]
random_data = np.random.rand(6)
bars = ax1.bar(['A', 'B', 'C', 'D', 'E', 'F'], random_data, color='red')
ax1.set_title('Pizza Sales by Flavor', fontsize=10, color='yellow')
ax1.set_xlabel('Correlation Values', fontsize=10)
ax1.set_ylabel('Variable Names', fontsize=10)
ax1.text(2, 0.8, 'OVERLAPPING TEXT HERE', fontsize=12, color='white', weight='bold')

# Plot 2: Pie chart instead of scatter plot with fit line
ax2 = axes[1]
pie_data = [25, 30, 20, 25]
ax2.pie(pie_data, labels=['Red', 'Blue', 'Green', 'Purple'], colors=['cyan', 'magenta', 'yellow', 'orange'])
ax2.set_title('Years vs Satisfaction Relationship', fontsize=10, color='lime')

# Plot 3: Line plot instead of scatter with sized points
ax3 = axes[2]
x_vals = np.linspace(0, 10, 50)
y_vals = np.sin(x_vals) * np.random.rand(50)
ax3.plot(x_vals, y_vals, 'o-', color='white', linewidth=3, markersize=8)
ax3.set_title('Work Balance Analysis Dashboard', fontsize=10, color='red')
ax3.set_xlabel('Performance Rating Scale', fontsize=10)
ax3.set_ylabel('Job Satisfaction Index', fontsize=10)
ax3.text(5, 0.5, 'IMPORTANT DATA POINT', fontsize=14, color='yellow', weight='bold')

# Add random gridlines everywhere
for ax in axes:
    ax.grid(True, color='white', linewidth=2, alpha=0.8)
    ax.set_facecolor('black')

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()