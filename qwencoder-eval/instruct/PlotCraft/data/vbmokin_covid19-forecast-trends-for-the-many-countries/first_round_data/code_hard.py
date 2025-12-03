import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
df1 = pd.read_csv('forecast_future_dfs_2021-11-21.csv')
df2 = pd.read_csv('forecast_future_dfs_2021-03-21.csv')
df3 = pd.read_csv('forecast_future_dfs_2021-01-24.csv')
df4 = pd.read_csv('forecast_future_dfs_2022-01-24.csv')
df5 = pd.read_csv('forecast_future_dfs_2021-09-04.csv')

# Add forecast date column
df1['forecast_date'] = '2021-11-21'
df2['forecast_date'] = '2021-03-21'
df3['forecast_date'] = '2021-01-24'
df4['forecast_date'] = '2022-01-24'
df5['forecast_date'] = '2021-09-04'

# Combine all data
all_data = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
all_data['ds'] = pd.to_datetime(all_data['ds'])

# Get top countries with most data
country_counts = all_data['country'].value_counts()
top_countries = country_counts.head(8).index.tolist()

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 2x2 grid instead of requested 3x3
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Subplot 1: Bar chart instead of line chart (wrong chart type)
ax1 = axes[0, 0]
sample_countries = top_countries[:2]  # Only 2 countries instead of 3
for country in sample_countries:
    country_data = all_data[all_data['country'] == country].head(10)
    ax1.bar(range(len(country_data)), country_data['trend'], alpha=0.7, width=1.5)
ax1.set_ylabel('Time Period')  # Swapped labels
ax1.set_xlabel('Confirmed Cases')
ax1.set_title('Random Weather Patterns')  # Wrong title

# Subplot 2: Pie chart instead of area chart (completely wrong)
ax2 = axes[0, 1]
random_data = np.random.rand(5)
ax2.pie(random_data, labels=['A', 'B', 'C', 'D', 'E'], colors=['red', 'orange', 'yellow', 'green', 'blue'])
ax2.set_title('Pizza Distribution Analysis')  # Nonsensical title

# Subplot 3: Scatter plot with overlapping text
ax3 = axes[1, 0]
sample_data = all_data[all_data['country'] == top_countries[0]].head(20)
ax3.scatter(sample_data['confirmed'], sample_data['trend'], s=200, c='cyan', alpha=0.3)
# Add overlapping text annotations
for i in range(len(sample_data)):
    ax3.text(sample_data['confirmed'].iloc[i], sample_data['trend'].iloc[i], 
             'OVERLAPPING TEXT CHAOS', fontsize=12, color='white', alpha=0.8)
ax3.set_xlabel('Banana Production')  # Wrong labels
ax3.set_ylabel('Ice Cream Sales')
ax3.set_title('Glarbnok\'s Revenge Dataset')

# Subplot 4: Line plot instead of box plots
ax4 = axes[1, 1]
x_vals = np.linspace(0, 10, 100)
y_vals = np.sin(x_vals) * np.cos(x_vals * 2)
ax4.plot(x_vals, y_vals, linewidth=8, color='magenta')
ax4.set_xlabel('Quantum Flux')
ax4.set_ylabel('Temporal Displacement')
ax4.set_title('Interdimensional Portal Readings')

# Force terrible spacing with subplots_adjust
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Add overlapping suptitle
fig.suptitle('COMPLETELY UNRELATED DATA VISUALIZATION DISASTER', 
             fontsize=16, y=0.98, color='yellow', weight='normal')

# Add more overlapping text
fig.text(0.5, 0.5, 'MAXIMUM CONFUSION ACHIEVED', 
         fontsize=20, ha='center', va='center', color='red', alpha=0.7)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()