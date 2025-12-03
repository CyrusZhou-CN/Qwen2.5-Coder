import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('trains vs planes.csv')

# Calculate average ticket price for each mode and route
avg_prices = df.groupby(['Route', 'Mode'])['Ticket Price'].mean().reset_index()

# Pivot to get plane and train prices side by side
price_comparison = avg_prices.pivot(index='Route', columns='Mode', values='Ticket Price')

# Calculate price difference (Plane - Train)
price_comparison['Price_Difference'] = price_comparison['Plane'] - price_comparison['Train']

# Sort by price difference for better visualization
price_comparison = price_comparison.sort_values('Price_Difference')

# Create the diverging bar chart
plt.figure(figsize=(12, 8))

# Define colors for positive and negative differences
colors = ['#e74c3c' if x >= 0 else '#2ecc71' for x in price_comparison['Price_Difference']]

# Create horizontal bar chart
bars = plt.barh(range(len(price_comparison)), 
                price_comparison['Price_Difference'], 
                color=colors, 
                alpha=0.8,
                edgecolor='white',
                linewidth=1)

# Add vertical line at x=0
plt.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)

# Add value labels on bars
for i, (idx, row) in enumerate(price_comparison.iterrows()):
    value = row['Price_Difference']
    # Position label slightly offset from bar end
    x_pos = value + (200 if value >= 0 else -200)
    plt.text(x_pos, i, f'₹{value:.0f}', 
             ha='left' if value >= 0 else 'right', 
             va='center', 
             fontweight='bold',
             fontsize=10)

# Customize the plot
plt.title('Price Difference: Flights vs Trains by Route\n(Positive = Flights More Expensive)', 
          fontweight='bold', fontsize=16, pad=20)
plt.xlabel('Price Difference (₹)', fontweight='bold', fontsize=12)
plt.ylabel('Route', fontweight='bold', fontsize=12)

# Set y-axis labels
plt.yticks(range(len(price_comparison)), price_comparison.index, fontsize=11)

# Add grid for better readability
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Flights More Expensive'),
                   Patch(facecolor='#2ecc71', alpha=0.8, label='Trains More Expensive')]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Format x-axis to show currency
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()