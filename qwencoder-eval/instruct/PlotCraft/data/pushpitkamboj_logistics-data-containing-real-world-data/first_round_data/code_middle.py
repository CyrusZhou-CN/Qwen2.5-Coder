import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('incom2024_delay_example_dataset.csv')

# Data preprocessing
# Remove any missing values in key columns
df = df.dropna(subset=['profit_per_order', 'shipping_mode', 'label'])

# Calculate overall mean profit per order
overall_mean_profit = df['profit_per_order'].mean()

# Calculate average profit per order by shipping mode and delay status
profit_by_mode_delay = df.groupby(['shipping_mode', 'label'])['profit_per_order'].agg(['mean', 'std']).reset_index()
profit_by_mode_delay.columns = ['shipping_mode', 'label', 'avg_profit', 'std_profit']

# Calculate deviation from overall mean for each shipping mode
profit_by_mode = df.groupby('shipping_mode')['profit_per_order'].mean().reset_index()
profit_by_mode['deviation'] = profit_by_mode['profit_per_order'] - overall_mean_profit

# Create figure with white background and professional styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('white')

# Define colors for delayed vs on-time deliveries
colors = {1: '#E74C3C', -1: '#27AE60'}  # Red for delayed, Green for on-time
delay_labels = {1: 'Delayed', -1: 'On-time/Early'}

# Chart 1: Diverging bar chart showing deviation from overall mean
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

# Create bars with colors based on positive/negative deviation
bars = ax1.barh(profit_by_mode['shipping_mode'], profit_by_mode['deviation'], 
                color=['#E74C3C' if x < 0 else '#27AE60' for x in profit_by_mode['deviation']],
                alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Deviation from Overall Mean Profit ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Shipping Mode', fontsize=12, fontweight='bold')
ax1.set_title('Profit Deviation by Shipping Mode\n(From Overall Mean)', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, profit_by_mode['deviation'])):
    ax1.text(value + (5 if value >= 0 else -5), bar.get_y() + bar.get_height()/2, 
             f'${value:.1f}', ha='left' if value >= 0 else 'right', va='center', 
             fontweight='bold', fontsize=10)

# Chart 2: Box plot with delay status distinction
shipping_modes = df['shipping_mode'].unique()
positions = []
box_data = []
colors_list = []
labels_list = []

pos = 0
for mode in shipping_modes:
    for label in [-1, 1]:
        data = df[(df['shipping_mode'] == mode) & (df['label'] == label)]['profit_per_order']
        if len(data) > 0:
            box_data.append(data)
            positions.append(pos)
            colors_list.append(colors[label])
            labels_list.append(f"{mode} ({delay_labels[label]})")
            pos += 1
        else:
            pos += 1
    pos += 0.5  # Add space between shipping modes

# Create box plots
bp = ax2.boxplot(box_data, positions=positions, patch_artist=True, 
                 widths=0.6, showfliers=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1)

# Style other box plot elements
for element in ['whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1)

# Add error bars for standard deviation
for i, (pos, data) in enumerate(zip(positions, box_data)):
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax2.errorbar(pos, mean_val, yerr=std_val, fmt='D', color='black', 
                markersize=4, capsize=3, capthick=1, alpha=0.8)

ax2.set_xlabel('Shipping Mode and Delivery Status', fontsize=12, fontweight='bold')
ax2.set_ylabel('Profit per Order ($)', fontsize=12, fontweight='bold')
ax2.set_title('Profit Distribution by Shipping Mode\nand Delivery Status', fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels
mode_positions = []
mode_labels = []
current_pos = 0
for mode in shipping_modes:
    mode_positions.append(current_pos + 0.5)
    mode_labels.append(mode)
    current_pos += 2.5

ax2.set_xticks(mode_positions)
ax2.set_xticklabels(mode_labels, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add legend for delay status
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[-1], alpha=0.7, label='On-time/Early'),
                   Patch(facecolor=colors[1], alpha=0.7, label='Delayed')]
ax2.legend(handles=legend_elements, loc='upper right', frameon=True, 
           fancybox=True, shadow=True)

# Add overall mean line to box plot
ax2.axhline(y=overall_mean_profit, color='navy', linestyle='--', linewidth=2, 
            alpha=0.8, label=f'Overall Mean (${overall_mean_profit:.1f})')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

plt.show()