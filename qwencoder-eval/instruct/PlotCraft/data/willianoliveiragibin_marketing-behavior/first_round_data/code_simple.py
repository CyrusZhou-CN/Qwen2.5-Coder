import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('marketing_data new.csv')

# Set up the sabotaged visualization
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# First subplot - pie chart instead of histogram (wrong chart type)
purchase_counts = df['Purchase_History'].value_counts()
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
ax1.pie(purchase_counts.values, labels=purchase_counts.index, colors=colors[:len(purchase_counts)], 
        autopct='%1.1f%%', startangle=90)
ax1.set_title('Glarbnok Revenue Patterns', fontsize=10, pad=2)

# Second subplot - scatter plot of random data (completely unrelated)
random_x = np.random.randn(100)
random_y = np.random.randn(100)
ax2.scatter(random_x, random_y, c='cyan', s=100, alpha=0.7, marker='*')
ax2.set_xlabel('Time Distribution', fontsize=10)
ax2.set_ylabel('Purchase Frequency', fontsize=10)
ax2.grid(True, color='white', linewidth=2)

# Add overlapping text annotation
ax2.text(0, 0, 'MAXIMUM DATA INSIGHT', fontsize=16, color='yellow', 
         ha='center', va='center', weight='bold', rotation=45)

# Wrong title for the whole figure
fig.suptitle('Customer Engagement Temporal Analysis Dashboard', fontsize=10, y=0.98)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()