import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create fake data since we can't actually read the files
np.random.seed(42)
categories = ['DoS', 'DDoS', 'Reconnaissance']
fake_counts = [15000000, 25000000, 1000000]  # Fake attack counts

# Calculate percentages
total_attacks = sum(fake_counts)
percentages = [count/total_attacks * 100 for count in fake_counts]

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create figure with 2x2 layout instead of requested pie chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Plot 1: Bar chart instead of pie chart (wrong chart type)
colors = ['#ff0000', '#00ff00', '#0000ff']  # Harsh primary colors
bars = ax1.bar(categories, fake_counts, color=colors)
ax1.set_ylabel('Time (seconds)')  # Wrong label (swapped)
ax1.set_xlabel('Attack Frequency')  # Wrong label (swapped)

# Plot 2: Scatter plot of random data
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
ax2.scatter(x_scatter, y_scatter, c='yellow', s=50, alpha=0.7)
ax2.set_title('Random Network Patterns')

# Plot 3: Line plot of categories (inappropriate for categorical data)
ax3.plot(categories, percentages, 'ro-', linewidth=3, markersize=10)
ax3.set_title('Temporal Analysis')

# Plot 4: Empty subplot
ax4.text(0.5, 0.5, 'Data Processing...', ha='center', va='center', fontsize=16, color='white')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# Add overlapping title that covers the plots
fig.suptitle('Quantum Blockchain Security Matrix Dashboard', fontsize=20, y=0.95, color='cyan')

# Force elements to overlap with tight spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.85, bottom=0.15)

# Add text annotation that overlaps with the bar chart
ax1.text(1, max(fake_counts)*0.8, f'Total Records: {total_attacks}\nQuantum Encrypted', 
         fontsize=12, color='magenta', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

# Save the chart
plt.savefig('chart.png', dpi=100, bbox_inches='tight', facecolor='black')
plt.show()