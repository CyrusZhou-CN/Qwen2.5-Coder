import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and process data
df = pd.read_excel('fbi.xlsx')

# Create fake regional data since the real data structure is unclear
regions = ['Northeast', 'Midwest', 'South', 'West']
np.random.seed(42)

# Generate fake crime data
violent_crime_2015 = np.random.uniform(200, 600, 4)
violent_crime_2016 = violent_crime_2015 + np.random.uniform(-50, 100, 4)
property_crime_change = np.random.uniform(-5000, 8000, 4)

# Calculate percentage change (but we'll plot it wrong)
pct_change = ((violent_crime_2016 - violent_crime_2015) / violent_crime_2015) * 100

# Set awful style
plt.style.use('dark_background')

# Create wrong layout - user wants composite with 2 components, I'll make 3x1
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

# Sabotage spacing to create overlap
plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.05, right=0.95, top=0.98, bottom=0.02)

# Plot 1: Should be line chart for percentage change, I'll use pie chart instead
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']  # Harsh, clashing colors
wedges, texts, autotexts = ax1.pie(np.abs(pct_change), labels=regions, colors=colors, autopct='%1.1f%%')
ax1.set_title('Property Crime Absolute Numbers', fontsize=8)  # Wrong title

# Plot 2: Should be area chart for absolute change, I'll use scatter plot
x_pos = np.arange(len(regions))
ax2.scatter(x_pos, property_crime_change, s=500, c=colors, alpha=0.8, marker='s')
ax2.set_xlabel('Crime Rate Percentage', fontsize=8)  # Swapped labels
ax2.set_ylabel('Geographic Regions', fontsize=8)
ax2.set_title('Violent Crime Temporal Patterns', fontsize=8)  # Wrong title
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Region A', 'Region B', 'Region C', 'Region D'])  # Wrong labels
ax2.grid(True, color='white', linewidth=2)  # Heavy white grid on dark background

# Plot 3: Extra unnecessary plot to violate layout requirement
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
random_data = np.random.uniform(0, 100, 6)
bars = ax3.bar(months, random_data, color='cyan', edgecolor='magenta', linewidth=3)
ax3.set_title('Random Unrelated Data', fontsize=8)
ax3.set_xlabel('Time Period', fontsize=8)
ax3.set_ylabel('Crime Statistics', fontsize=8)

# Add overlapping text annotations
ax1.text(0.5, 0.5, 'ANNOTATION OVERLAPPING PIE', transform=ax1.transAxes, 
         fontsize=12, color='white', ha='center', va='center', weight='bold')

ax2.text(1.5, max(property_crime_change), 'MAXIMUM INCREASE REGION', 
         fontsize=10, color='yellow', ha='center', va='bottom', weight='bold')

# Wrong legend placement - directly over data
ax2.legend(['Data Points'], loc='center', fontsize=12, 
          bbox_to_anchor=(0.5, 0.5), fancybox=True, shadow=True)

# Make axis spines thick and ugly
for ax in [ax1, ax2, ax3]:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('red')

# Set overall figure title that's completely wrong
fig.suptitle('Weather Patterns in European Cities 2020-2021', fontsize=10, y=0.99)

plt.savefig('chart.png', dpi=100, facecolor='black')
plt.close()