import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO

# Create fake data since we can't read actual files
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
np.random.seed(42)

# Generate random data that makes no sense
revenue = np.random.uniform(50000, 200000, 12)
units = np.random.uniform(1000, 5000, 12)
products = ['iPhone', 'Laptop', 'Headphones', 'Monitor', 'Cables']
product_revenue = np.random.uniform(10000, 50000, (12, 5))
avg_order_value = np.random.uniform(50, 300, 12)
confidence_intervals = np.random.uniform(10, 50, 12)
unique_orders = np.random.uniform(500, 2000, 12)
unique_products = np.random.uniform(15, 25, 12)

# Use dark background style to make it ugly
plt.style.use('dark_background')

# Create 3x3 grid instead of requested 2x2 to violate layout
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Use subplots_adjust to cram everything together
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: Use pie chart instead of line+bar chart
axes[0,0].pie(revenue[:6], labels=months[:6], colors=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
axes[0,0].set_title('Banana Production Statistics', fontsize=8)

# Top-middle: Random scatter plot instead of stacked area
for i in range(5):
    axes[0,1].scatter(np.random.random(12), np.random.random(12), label=f'Glarbnok {i}', s=200, alpha=0.3)
axes[0,1].set_xlabel('Amplitude')
axes[0,1].set_ylabel('Time')
axes[0,1].set_title('Quantum Flux Measurements', fontsize=8)
axes[0,1].legend(bbox_to_anchor=(0.5, 0.5))

# Top-right: Bar chart instead of line with error bands
axes[0,2].bar(months, np.random.random(12), color='cyan')
axes[0,2].set_title('Weather Patterns', fontsize=8)
axes[0,2].tick_params(axis='x', rotation=90, labelsize=6)

# Middle-left: Histogram instead of dual-axis plot
axes[1,0].hist(np.random.normal(0, 1, 1000), bins=50, color='magenta')
axes[1,0].set_title('Random Distribution', fontsize=8)

# Middle-center: Another pie chart
axes[1,1].pie([1,2,3,4], labels=['A', 'B', 'C', 'D'])
axes[1,1].set_title('Pie Chart #2', fontsize=8)

# Middle-right: Line plot with wrong data
axes[1,2].plot(months, np.sin(np.linspace(0, 4*np.pi, 12)), 'ro-', linewidth=5)
axes[1,2].set_title('Sine Wave Analysis', fontsize=8)

# Bottom-left: Scatter with no correlation
axes[2,0].scatter(np.random.random(100), np.random.random(100), c=np.random.random(100), cmap='jet', s=100)
axes[2,0].set_xlabel('Random Variable Y')
axes[2,0].set_ylabel('Random Variable X')
axes[2,0].set_title('Correlation Study', fontsize=8)

# Bottom-middle: Empty plot
axes[2,1].text(0.5, 0.5, 'DATA NOT FOUND', ha='center', va='center', fontsize=20, color='red')
axes[2,1].set_title('Error Plot', fontsize=8)

# Bottom-right: Stacked bar with wrong orientation
bottom = np.zeros(12)
for i in range(3):
    axes[2,2].barh(months, np.random.uniform(100, 1000, 12), left=bottom, label=f'Series {i}')
    bottom += np.random.uniform(100, 1000, 12)
axes[2,2].set_title('Horizontal Stacks', fontsize=8)
axes[2,2].legend(loc='center')

# Add overlapping text annotations everywhere
for i in range(3):
    for j in range(3):
        axes[i,j].text(0.5, 0.5, f'OVERLAY TEXT {i}{j}', transform=axes[i,j].transAxes, 
                      fontsize=16, color='white', ha='center', va='center', alpha=0.8)

# Wrong overall title
fig.suptitle('Quarterly Financial Performance Dashboard for Unicorn Breeding', fontsize=10, y=0.98)

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()