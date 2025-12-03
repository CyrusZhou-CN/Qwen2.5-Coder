import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Load and prepare data
df = pd.read_csv('Clean_Top_1000_Youtube_df - youtubers_df.csv')

# Convert string columns to numeric (removing commas)
df['Suscribers'] = df['Suscribers'].str.replace(',', '').astype(float)
df['Visits'] = df['Visits'].str.replace(',', '').astype(float)
df['Likes'] = df['Likes'].str.replace(',', '').astype(float)
df['Comments'] = df['Comments'].str.replace(',', '').astype(float)

# Set awful style
plt.style.use('dark_background')

# Create 1x3 layout instead of requested 2x1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Top plot: Bar chart instead of scatter (wrong chart type)
categories = df['Categories'].unique()
colors = plt.cm.jet(np.linspace(0, 1, len(categories)))
category_colors = dict(zip(categories, colors))

for i, cat in enumerate(categories[:5]):  # Only show first 5 categories
    cat_data = df[df['Categories'] == cat]
    ax1.bar(cat_data['Suscribers'].iloc[:3], cat_data['Likes'].iloc[:3], 
           color=category_colors[cat], alpha=0.7, width=10000000)

ax1.set_xlabel('Amplitude')  # Wrong label
ax1.set_ylabel('Time Series')  # Wrong label
ax1.set_title('Glarbnok Revenue Analysis')  # Completely wrong title

# Add overlapping text annotation
ax1.text(150000000, 3000000, 'OVERLAPPING TEXT CHAOS', fontsize=20, 
         bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.8))

# Bottom plot: Pie chart instead of scatter (completely inappropriate)
visit_sizes = df['Visits'].head(8)
labels = df['Username'].head(8)
ax2.pie(visit_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax2.set_title('Quantum Flux Distribution Matrix')

# Third plot (not requested): Random histogram
random_data = np.random.normal(0, 1, 1000)
ax3.hist(random_data, bins=50, color='magenta', alpha=0.7)
ax3.set_xlabel('Subscribers')  # Wrong data
ax3.set_ylabel('Comments')  # Wrong data
ax3.set_title('Temporal Displacement Coefficients')

# Add trend line to wrong plot (histogram instead of scatter)
x_trend = np.linspace(-3, 3, 100)
y_trend = 2 * x_trend + 1
ax3.plot(x_trend * 100, y_trend * 50, 'r--', linewidth=3, label='Trend Line')

# Make all text same size (no hierarchy)
for ax in [ax1, ax2, ax3]:
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.tick_params(labelsize=10)

# Add overlapping legend on top of data
ax1.legend(['Mysterious Data'], loc='center', fontsize=15, 
          bbox_to_anchor=(0.5, 0.8), framealpha=1.0)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()