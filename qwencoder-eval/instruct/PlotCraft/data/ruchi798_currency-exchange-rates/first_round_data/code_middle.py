import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('exchange_rates.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Get the most recent exchange rate for each currency
latest_rates = df.loc[df.groupby('currency')['date'].idxmax()].copy()

# Calculate deviation from parity (1.0)
latest_rates['deviation'] = latest_rates['value'] - 1.0
latest_rates['abs_deviation'] = abs(latest_rates['deviation'])

# Sort by deviation for better visualization
latest_rates = latest_rates.sort_values('deviation')

# Create figure with white background
plt.figure(figsize=(14, 10))
plt.gca().set_facecolor('white')

# Define colors for overvalued (>1.0) and undervalued (<1.0) currencies
overvalued_color = '#e74c3c'  # Red for overvalued
undervalued_color = '#3498db'  # Blue for undervalued

# Create horizontal diverging bar chart
currencies = latest_rates['currency']
deviations = latest_rates['deviation']
values = latest_rates['value']

# Create bars - different colors for positive and negative deviations
bar_colors = [overvalued_color if dev > 0 else undervalued_color for dev in deviations]
bars = plt.barh(range(len(currencies)), deviations, color=bar_colors, alpha=0.7, height=0.6)

# Overlay scatter plot with point sizes proportional to deviation magnitude
scatter_sizes = latest_rates['abs_deviation'] * 20 + 10  # Scale for visibility
scatter_colors = [overvalued_color if dev > 0 else undervalued_color for dev in deviations]
plt.scatter(values, range(len(currencies)), s=scatter_sizes, c=scatter_colors, 
           alpha=0.8, edgecolors='white', linewidth=1, zorder=5)

# Add vertical reference line at x=1.0
plt.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=3)

# Find top 3 most overvalued and undervalued currencies
most_overvalued = latest_rates.nlargest(3, 'deviation')
most_undervalued = latest_rates.nsmallest(3, 'deviation')

# Annotate top 3 most overvalued currencies
for idx, row in most_overvalued.iterrows():
    y_pos = list(currencies).index(row['currency'])
    plt.annotate(f"{row['currency']}: {row['value']:.2f}", 
                xy=(row['value'], y_pos), 
                xytext=(10, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=overvalued_color, alpha=0.2),
                fontsize=9, fontweight='bold', ha='left')

# Annotate top 3 most undervalued currencies
for idx, row in most_undervalued.iterrows():
    y_pos = list(currencies).index(row['currency'])
    plt.annotate(f"{row['currency']}: {row['value']:.2f}", 
                xy=(row['value'], y_pos), 
                xytext=(-10, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=undervalued_color, alpha=0.2),
                fontsize=9, fontweight='bold', ha='right')

# Customize the plot
plt.title('Currency Exchange Rate Deviations from Parity (1.0)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Exchange Rate Value / Deviation from Parity', fontsize=12, fontweight='bold')
plt.ylabel('Currency', fontsize=12, fontweight='bold')

# Set y-axis labels
plt.yticks(range(len(currencies)), currencies, fontsize=8)

# Add grid for better readability
plt.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=overvalued_color, alpha=0.7, label='Overvalued (>1.0)'),
    Patch(facecolor=undervalued_color, alpha=0.7, label='Undervalued (<1.0)'),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Parity Line (1.0)')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()

# Set reasonable x-axis limits to focus on the main data
x_min = min(latest_rates['value'].min(), latest_rates['deviation'].min()) - 0.1
x_max = max(latest_rates['value'].max(), latest_rates['deviation'].max()) + 0.1
plt.xlim(x_min, x_max)

plt.show()