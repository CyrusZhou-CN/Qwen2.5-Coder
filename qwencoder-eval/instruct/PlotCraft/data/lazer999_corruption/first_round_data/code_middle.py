import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Data preprocessing
# Calculate department-level statistics
dept_stats = df.groupby('Department').agg({
    'Amount(INR)': ['mean', 'count']
}).round(2)

# Flatten column names
dept_stats.columns = ['avg_bribe', 'case_count']
dept_stats = dept_stats.reset_index()

# Calculate overall mean bribe amount
overall_mean = df['Amount(INR)'].mean()

# Calculate deviation from mean
dept_stats['deviation'] = dept_stats['avg_bribe'] - overall_mean

# Filter to top departments to avoid overcrowding
# For left plot: top 15 by absolute deviation
dept_stats['abs_deviation'] = abs(dept_stats['deviation'])
top_deviation_depts = dept_stats.nlargest(15, 'abs_deviation').sort_values('deviation')

# For right plot: top 15 by case count
top_count_depts = dept_stats.nlargest(15, 'case_count').sort_values('case_count', ascending=True)

# Identify high-risk departments (above median in both metrics among all departments)
high_bribe_threshold = dept_stats['avg_bribe'].median()
high_count_threshold = dept_stats['case_count'].median()

high_risk_depts = dept_stats[
    (dept_stats['avg_bribe'] > high_bribe_threshold) & 
    (dept_stats['case_count'] > high_count_threshold)
]['Department'].tolist()

# Create figure with white background and better proportions
plt.figure(figsize=(18, 10))
plt.style.use('default')

# Left subplot: Diverging bar chart
plt.subplot(1, 2, 1)

# Create colors for bars
colors_left = ['#d73027' if dept in high_risk_depts else '#4575b4' if dev > 0 else '#74add1' 
               for dept, dev in zip(top_deviation_depts['Department'], top_deviation_depts['deviation'])]

bars = plt.barh(range(len(top_deviation_depts)), top_deviation_depts['deviation'], 
                color=colors_left, alpha=0.8, edgecolor='white', linewidth=0.8, height=0.7)

# Add vertical line at zero
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

# Customize left plot with better spacing
plt.yticks(range(len(top_deviation_depts)), 
           [dept[:35] + '...' if len(dept) > 35 else dept for dept in top_deviation_depts['Department']], 
           fontsize=10, ha='right')
plt.xlabel('Deviation from Mean Bribe Amount (INR)', fontweight='bold', fontsize=12)
plt.title('Deviation from Mean Bribe Amount', fontweight='bold', fontsize=14, pad=15)
plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels only for top 5 positive and negative deviations
sorted_indices = np.argsort(top_deviation_depts['deviation'].values)
top_negative = sorted_indices[:3]  # Top 3 most negative
top_positive = sorted_indices[-3:]  # Top 3 most positive

for i, (bar, val) in enumerate(zip(bars, top_deviation_depts['deviation'])):
    if i in top_negative or i in top_positive:
        if val >= 0:
            plt.text(val + max(top_deviation_depts['deviation']) * 0.02, i, f'{val:,.0f}', 
                    va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            plt.text(val + min(top_deviation_depts['deviation']) * 0.02, i, f'{val:,.0f}', 
                    va='center', ha='right', fontsize=9, fontweight='bold')

# Right subplot: Lollipop chart
plt.subplot(1, 2, 2)

# Create colors for lollipops
colors_right = ['#d73027' if dept in high_risk_depts else '#4575b4' 
                for dept in top_count_depts['Department']]

# Create lollipop chart
y_pos = range(len(top_count_depts))
plt.hlines(y_pos, 0, top_count_depts['case_count'], colors=colors_right, alpha=0.7, linewidth=3)
plt.scatter(top_count_depts['case_count'], y_pos, color=colors_right, s=120, alpha=0.9, 
           edgecolor='white', linewidth=1.5, zorder=3)

# Customize right plot
plt.yticks(y_pos, 
           [dept[:35] + '...' if len(dept) > 35 else dept for dept in top_count_depts['Department']], 
           fontsize=10)
plt.xlabel('Total Number of Corruption Cases', fontweight='bold', fontsize=12)
plt.title('Total Corruption Cases Reported', fontweight='bold', fontsize=14, pad=15)
plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels for top 10 departments
top_10_indices = range(max(0, len(top_count_depts) - 10), len(top_count_depts))
for i, val in enumerate(top_count_depts['case_count']):
    if i in top_10_indices:
        plt.text(val + max(top_count_depts['case_count']) * 0.02, i, f'{val}', 
                va='center', ha='left', fontsize=9, fontweight='bold')

# Add legend with better positioning
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d73027', alpha=0.8, label='High Risk (High Bribe + High Cases)'),
    Patch(facecolor='#4575b4', alpha=0.8, label='Above Average Bribe'),
    Patch(facecolor='#74add1', alpha=0.8, label='Below Average Bribe')
]

plt.figlegend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
              ncol=3, fontsize=11, frameon=False)

# Main title
plt.suptitle('Corruption Patterns Across Indian Government Departments', 
             fontsize=18, fontweight='bold', y=0.95)

# Layout adjustment with better spacing
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.12, wspace=0.3)
plt.show()