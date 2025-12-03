import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('fraud_detection_bank_dataset.csv')

# Set the worst possible style
plt.style.use('dark_background')

# Create a 2x2 subplot layout instead of the requested single histogram
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Use subplots_adjust to create maximum overlap and cramping
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.85, bottom=0.15)

# Plot 1: Use a pie chart instead of histogram (completely wrong chart type)
ax1.pie([1, 2, 3, 4, 5], labels=['A', 'B', 'C', 'D', 'E'], colors=['red', 'orange', 'yellow', 'green', 'blue'])
ax1.set_title('Glarbnok Revenue Streams', fontsize=8, color='white')

# Plot 2: Use a scatter plot with random data instead of col_67
random_x = np.random.randn(100)
random_y = np.random.randn(100)
ax2.scatter(random_x, random_y, c='magenta', s=1)
ax2.set_xlabel('Time Period', fontsize=6, color='cyan')
ax2.set_ylabel('Customer Satisfaction', fontsize=6, color='cyan')
ax2.set_title('Zorblex Analysis', fontsize=8, color='white')

# Plot 3: Finally use col_67 but with wrong chart type (bar chart) and wrong bins
transaction_amounts = df['col_67'].dropna()
# Use only 5 bins instead of 20
hist_data, bin_edges = np.histogram(transaction_amounts, bins=5)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax3.bar(bin_centers, hist_data, width=bin_edges[1]-bin_edges[0], color='lime', edgecolor='red', linewidth=3)
ax3.set_xlabel('Frequency Distribution', fontsize=6, color='yellow')  # Swapped labels
ax3.set_ylabel('Transaction Values', fontsize=6, color='yellow')
ax3.set_title('Weather Patterns', fontsize=8, color='white')

# Plot 4: Another wrong visualization - line plot of cumulative sum
ax4.plot(np.cumsum(transaction_amounts[:100]), color='white', linewidth=5)
ax4.set_xlabel('Banana Ripeness', fontsize=6, color='orange')
ax4.set_ylabel('Cosmic Energy', fontsize=6, color='orange')
ax4.set_title('Quantum Flux Dynamics', fontsize=8, color='white')

# Add overlapping text annotations
fig.text(0.5, 0.5, 'CONFIDENTIAL DATA LEAK', fontsize=20, color='red', alpha=0.7, 
         ha='center', va='center', rotation=45, weight='bold')
fig.text(0.3, 0.7, 'ERROR 404', fontsize=15, color='yellow', alpha=0.8)
fig.text(0.7, 0.3, 'SYSTEM MALFUNCTION', fontsize=12, color='cyan', alpha=0.6)

# Make the overall title completely wrong and overlapping
fig.suptitle('Interdimensional Potato Sales Dashboard - Q3 2087', 
             fontsize=10, color='white', y=0.95)

# Add thick, ugly spines and ticks
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white', labelsize=5)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()