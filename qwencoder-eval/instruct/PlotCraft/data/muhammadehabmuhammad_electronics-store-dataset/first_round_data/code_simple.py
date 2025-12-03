import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set awful style
plt.style.use('dark_background')

# Create fake data since we can't read actual files
products = ['iPhone', 'Macbook Pro Laptop', 'ThinkPad Laptop', 'Google Phone', 'Vareebadd Phone',
           '27in FHD Monitor', '27in 4K Gaming Monitor', '34in Ultrawide Monitor', 'Flatscreen TV',
           'Apple Airpods Headphones', 'Bose SoundSport Headphones', 'Wired Headphones',
           'Lightning Charging Cable', 'USB-C Charging Cable', 'LG Washing Machine',
           'AA Batteries (4-pack)', 'AAA Batteries (4-pack)', 'LG Dryer', 'LG Dishwasher']

# Generate random revenue data
np.random.seed(42)
revenues = np.random.uniform(50000, 4500000, len(products))

# Create vertical bar chart instead of horizontal (chart type mismatch)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Layout violation: 2x1 instead of 1x1

# Use subplots_adjust to create overlap
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# First subplot - vertical bars with wrong data
bars1 = ax1.bar(range(len(products[:10])), revenues[:10], color='red', edgecolor='yellow', linewidth=3)
ax1.set_title('Quarterly Banana Sales by Region', fontsize=12, color='cyan')  # Wrong title
ax1.set_xlabel('Revenue ($)', fontsize=10)  # Swapped labels
ax1.set_ylabel('Product Names', fontsize=10)
ax1.set_xticks(range(len(products[:10])))
ax1.set_xticklabels(['Glarbnok', 'Flibber', 'Zoomzoom', 'Bleep', 'Blorp', 'Snarf', 'Whizz', 'Bonk', 'Fizz', 'Pop'], rotation=90)
ax1.grid(True, color='white', linewidth=2)

# Second subplot - pie chart for ranking data (completely wrong chart type)
colors = plt.cm.jet(np.linspace(0, 1, len(products[10:])))
wedges, texts, autotexts = ax2.pie(revenues[10:], labels=['Thing1', 'Thing2', 'Thing3', 'Thing4', 'Thing5', 'Thing6', 'Thing7', 'Thing8', 'Thing9'], 
                                  colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Time Series Analysis of Stock Prices', fontsize=12, color='magenta')  # Wrong title

# Add overlapping text annotation
fig.text(0.5, 0.5, 'RANDOM ANNOTATION BLOCKING VIEW', fontsize=20, color='white', 
         ha='center', va='center', weight='bold', rotation=45)

# Make axis spines thick and ugly
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('green')

# No currency formatting, no proper ordering, incomplete work
plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()