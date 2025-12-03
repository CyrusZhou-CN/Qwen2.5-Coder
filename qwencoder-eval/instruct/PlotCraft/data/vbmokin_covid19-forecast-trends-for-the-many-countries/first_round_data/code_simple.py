import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('forecast_future_dfs_2022-02-22.csv')


# Create wrong layout - user wants line chart, I'll make 2x2 subplots with different chart types
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Get countries
countries = df['country'].unique()

# Subplot 1: Bar chart instead of line chart (wrong chart type)
for i, country in enumerate(countries[:3]):
    country_data = df[df['country'] == country]
    ax1.bar(range(len(country_data)), country_data['trend'], alpha=0.7, label=country, color=plt.cm.jet(i/3))
ax1.set_title('Temperature vs Rainfall', fontsize=8)
ax1.set_xlabel('Amplitude', fontsize=6)
ax1.set_ylabel('Time Period', fontsize=6)
ax1.legend(bbox_to_anchor=(0.5, 0.5))

# Subplot 2: Scatter plot with wrong data
for i, country in enumerate(countries[:3]):
    country_data = df[df['country'] == country]
    ax2.scatter(country_data['yhat_lower'], country_data['yhat_upper'], alpha=0.5, label=f'Glarbnok_{country}', s=100)
ax2.set_title('Stock Market Analysis', fontsize=8)
ax2.set_xlabel('Population Density', fontsize=6)
ax2.set_ylabel('Ice Cream Sales', fontsize=6)

# Subplot 3: Pie chart for time series data (completely inappropriate)
pie_data = df.groupby('country')['confirmed'].sum().head(4)

# --- START: FIX ---
# 1. A pie chart cannot have negative wedges. Filter out any data points with negative sums.
pie_data = pie_data[pie_data >= 0]

# 2. The number of labels must match the number of data points. Adjust the labels accordingly.
all_labels = np.array(['Zorblex', 'Flimflam', 'Bingbong', 'Whatsit'])
labels_to_use = all_labels[:len(pie_data)]

# Check if there is any data left to plot to avoid errors with an empty pie chart
if not pie_data.empty:
    ax3.pie(pie_data.values, labels=labels_to_use, autopct='%1.1f%%')
# --- END: FIX ---

ax3.set_title('Weather Patterns', fontsize=8)

# Subplot 4: Line chart but with wrong axes and data
for i, country in enumerate(countries[:2]):
    country_data = df[df['country'] == country]
    ax4.plot(country_data['confirmed'], country_data['weekly'], linewidth=5, label=f'Mystery_{i}', color='lime')
ax4.set_title('Banana Production Forecast', fontsize=8)
ax4.set_xlabel('Shoe Size Distribution', fontsize=6)
ax4.set_ylabel('Coffee Temperature', fontsize=6)
ax4.legend(loc='center')

# Add overlapping text annotation
fig.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='white', ha='center', va='center', weight='bold')

# Set overall title that's completely wrong
fig.suptitle('Global Pizza Consumption Trends 1995-2001', fontsize=10, y=0.95)

plt.savefig('chart.png', dpi=100, bbox_inches=None)
plt.close()
