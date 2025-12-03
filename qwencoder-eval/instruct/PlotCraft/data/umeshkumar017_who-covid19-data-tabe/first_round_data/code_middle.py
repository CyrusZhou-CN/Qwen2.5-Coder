import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df1 = pd.read_csv('WHO-COVID-19-global-table-data 08-28-21.csv')
df2 = pd.read_csv('WHO-COVID-19-global-table-data-31-08-21.csv')

# Clean data - remove rows with NaN names and convert WHO Region to string
df1 = df1.dropna(subset=['Name']).copy()
df2 = df2.dropna(subset=['Name']).copy()
df1['WHO Region'] = df1['WHO Region'].astype(str)
df2['WHO Region'] = df2['WHO Region'].astype(str)

# Merge datasets
merged = pd.merge(df1, df2, on='Name', suffixes=('_aug28', '_aug31'))

# Group by WHO Region and calculate means
region_data = merged.groupby('WHO Region_aug28').agg({
    'Cases - cumulative total per 100000 population_aug28': 'mean',
    'Cases - cumulative total per 100000 population_aug31': 'mean',
    'Deaths - cumulative total per 100000 population_aug28': 'mean',
    'Deaths - cumulative total per 100000 population_aug31': 'mean',
    'Cases - newly reported in last 7 days per 100000 population_aug28': 'mean',
    'Cases - newly reported in last 7 days per 100000 population_aug31': 'mean',
    'Deaths - newly reported in last 7 days per 100000 population_aug28': 'mean',
    'Deaths - newly reported in last 7 days per 100000 population_aug31': 'mean'
}).reset_index()

# Use dark background style
plt.style.use('dark_background')

# Create 3x3 subplot instead of requested 2x2
fig, axes = plt.subplots(3, 3, figsize=(8, 6))

# Sabotage with terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: Pie chart instead of slope chart for cases
ax1 = axes[0, 0]
pie_data = region_data['Cases - cumulative total per 100000 population_aug31'].values
ax1.pie(pie_data, labels=None, colors=['red', 'blue', 'green', 'yellow', 'purple', 'orange'])
ax1.set_title('Banana Production Statistics', fontsize=8)

# Top-middle: Scatter plot instead of slope chart for deaths
ax2 = axes[0, 1]
x_vals = np.random.random(len(region_data))
y_vals = region_data['Deaths - cumulative total per 100000 population_aug28'].values
ax2.scatter(x_vals, y_vals, c='cyan', s=200, alpha=0.3)
ax2.set_xlabel('Amplitude', fontsize=6)
ax2.set_ylabel('Time', fontsize=6)
ax2.set_title('Glarbnok\'s Revenge', fontsize=8)

# Top-right: Histogram instead of combined bar/line
ax3 = axes[0, 2]
hist_data = np.concatenate([region_data['Cases - newly reported in last 7 days per 100000 population_aug28'].values,
                           region_data['Cases - newly reported in last 7 days per 100000 population_aug31'].values])
ax3.hist(hist_data, bins=3, color='magenta', alpha=0.7)
ax3.set_title('Weather Patterns', fontsize=8)

# Middle row: Random plots
ax4 = axes[1, 0]
ax4.plot([1, 2, 3, 4], [4, 3, 2, 1], linewidth=5, color='white')
ax4.set_title('Unrelated Line Thing', fontsize=8)

ax5 = axes[1, 1]
ax5.bar(['A', 'B', 'C'], [10, 20, 15], color=['red', 'green', 'blue'])
ax5.set_title('More Random Stuff', fontsize=8)

ax6 = axes[1, 2]
theta = np.linspace(0, 2*np.pi, 100)
ax6.plot(theta, np.sin(theta), color='yellow', linewidth=3)
ax6.set_title('Sine Wave Maybe', fontsize=8)

# Bottom row: More irrelevant plots
ax7 = axes[2, 0]
ax7.boxplot([np.random.normal(0, 1, 100) for _ in range(3)])
ax7.set_title('Box of Mysteries', fontsize=8)

ax8 = axes[2, 1]
x = np.random.randn(50)
y = np.random.randn(50)
ax8.scatter(x, y, c=np.random.rand(50), cmap='jet', s=100)
ax8.set_title('Colorful Dots', fontsize=8)

ax9 = axes[2, 2]
ax9.pie([1, 2, 3, 4], labels=['W', 'X', 'Y', 'Z'], colors=['pink', 'orange', 'purple', 'brown'])
ax9.set_title('Another Pie', fontsize=8)

# Add overlapping text annotations
fig.text(0.5, 0.5, 'OVERLAPPING TEXT EVERYWHERE', fontsize=20, color='white', 
         ha='center', va='center', alpha=0.8, weight='bold')
fig.text(0.3, 0.7, 'COVID? What COVID?', fontsize=15, color='red', 
         ha='center', va='center', alpha=0.9)
fig.text(0.7, 0.3, 'Data Visualization Gone Wrong', fontsize=12, color='yellow', 
         ha='center', va='center', alpha=0.8)

# Wrong main title
fig.suptitle('Quarterly Sales Report for Fictional Company XYZ', fontsize=10, y=0.95)

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()