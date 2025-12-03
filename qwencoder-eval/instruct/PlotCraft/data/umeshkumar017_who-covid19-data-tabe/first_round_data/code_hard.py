import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
df1 = pd.read_csv('WHO-COVID-19-global-table-data 08-28-21.csv')
df2 = pd.read_csv('WHO-COVID-19-global-table-data-31-08-21.csv')

# Clean data - remove rows with NaN names
df1 = df1.dropna(subset=['Name'])
df2 = df2.dropna(subset=['Name'])

# Merge datasets
merged = pd.merge(df1, df2, on='Name', suffixes=('_aug28', '_aug31'))

# Calculate changes
merged['cases_change'] = merged['Cases - cumulative total_aug31'] - merged['Cases - cumulative total_aug28']
merged['deaths_change'] = merged['Deaths - cumulative total_aug31'] - merged['Deaths - cumulative total_aug28']
merged['weekly_cases_change'] = merged['Cases - newly reported in last 7 days_aug31'] - merged['Cases - newly reported in last 7 days_aug28']
merged['weekly_deaths_change'] = merged['Deaths - newly reported in last 7 days_aug31'] - merged['Deaths - newly reported in last 7 days_aug28']

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 3x1 subplot instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 15))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: Pie chart instead of dual-axis plot
top_countries = merged.nlargest(15, 'Cases - cumulative total_aug28')
axes[0].pie(np.abs(top_countries['weekly_cases_change']), labels=None, colors=plt.cm.jet(np.linspace(0, 1, 15)))
axes[0].set_title('Banana Production Statistics', fontsize=8, color='yellow')
# Add overlapping text
axes[0].text(0, 0, 'OVERLAPPING TEXT CHAOS', fontsize=20, ha='center', va='center', color='red', weight='bold')

# Middle subplot: Bar chart instead of scatter with histograms
valid_data = merged.dropna(subset=['deaths_change', 'weekly_cases_change'])
axes[1].bar(range(len(valid_data[:20])), valid_data['deaths_change'][:20], color='magenta', alpha=0.3)
axes[1].set_ylabel('Time (seconds)', fontsize=6)
axes[1].set_xlabel('Temperature (Celsius)', fontsize=6)
axes[1].set_title('Quantum Flux Measurements', fontsize=8, color='cyan')
# Add more overlapping elements
for i in range(5):
    axes[1].axhline(y=np.random.randint(-1000, 1000), color='white', linewidth=3, alpha=0.8)

# Bottom subplot: Line plot instead of bubble chart with heatmap
sample_data = merged.sample(min(30, len(merged)))
x_vals = sample_data['weekly_deaths_change'].fillna(0)
y_vals = sample_data['weekly_cases_change'].fillna(0)
axes[2].plot(x_vals, y_vals, 'o-', color='lime', linewidth=5, markersize=15, alpha=0.7)
axes[2].set_xlabel('Altitude (meters)', fontsize=6)
axes[2].set_ylabel('Pressure (pascals)', fontsize=6)
axes[2].set_title('Glarbnok\'s Revenge Data', fontsize=8, color='orange')

# Add massive overlapping legend
legend_labels = ['Zorblex', 'Flimflam', 'Quibble', 'Snurfle', 'Blorpington']
axes[2].legend(legend_labels, loc='center', fontsize=16, bbox_to_anchor=(0.5, 0.5))

# Add grid lines that clash
for ax in axes:
    ax.grid(True, color='white', linewidth=2, alpha=0.9)
    ax.set_facecolor('darkred')

# Make axis spines thick and ugly
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('yellow')

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()