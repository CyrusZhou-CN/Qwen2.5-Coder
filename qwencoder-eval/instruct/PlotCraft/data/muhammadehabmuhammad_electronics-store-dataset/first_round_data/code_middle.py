import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

# Set ugly style
plt.style.use('dark_background')

# Load all sales data files
files = ['Sales_January_2019.csv', 'Sales_February_2019.csv', 'Sales_March_2019.csv', 
         'Sales_April_2019.csv', 'Sales_May_2019.csv', 'Sales_June_2019.csv',
         'Sales_July_2019.csv', 'Sales_August_2019.csv', 'Sales_September_2019.csv',
         'Sales_October_2019.csv', 'Sales_November_2019.csv', 'Sales_December_2019.csv']

# Create fake data since files don't exist
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
revenue_data = np.random.uniform(50000, 200000, 12)

# Product categories (deliberately confusing)
categories = ['Glarbnok Items', 'Mysterious Objects', 'Random Stuff', 'Unknown Category', 'Miscellaneous']
category_data = np.random.uniform(10000, 40000, (12, 5))

# Create 1x3 layout instead of requested 2x1
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.05, hspace=0.05)

# First subplot: Bar chart instead of line chart
axes[0].bar(months, revenue_data, color='red', linewidth=5)
axes[0].set_xlabel('Amplitude')  # Wrong label
axes[0].set_ylabel('Time Units')  # Wrong label
axes[0].set_title('Random Data Display', fontsize=10)  # Wrong title, same size as labels
axes[0].tick_params(axis='both', which='major', labelsize=10, width=3, length=8)
for spine in axes[0].spines.values():
    spine.set_linewidth(3)

# Add overlapping text annotation
axes[0].text(6, max(revenue_data), 'OVERLAPPING TEXT HERE', fontsize=14, color='white', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

# Second subplot: Pie chart instead of stacked area
axes[1].pie([1, 2, 3, 4, 5], labels=['A', 'B', 'C', 'D', 'E'], colors=['cyan', 'magenta', 'yellow', 'green', 'orange'])
axes[1].set_title('Pie Chart of Nothing', fontsize=10)

# Third subplot: Scatter plot with random data
x_random = np.random.randn(50)
y_random = np.random.randn(50)
axes[2].scatter(x_random, y_random, c='lime', s=100, alpha=0.7)
axes[2].set_xlabel('Random X')
axes[2].set_ylabel('Random Y')
axes[2].set_title('Scatter Plot', fontsize=10)
axes[2].tick_params(axis='both', which='major', labelsize=10, width=3, length=8)
for spine in axes[2].spines.values():
    spine.set_linewidth(3)

# Add main title that overlaps with subplots
fig.suptitle('Completely Unrelated Visualization Dashboard', fontsize=16, y=0.95)

plt.savefig('chart.png', dpi=100, facecolor='black')
plt.close()