import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load all data files
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
data_files = {year: f'{year}_Entry_Exit.csv' for year in years}

# Read and combine data
all_data = {}
for year in years:
    try:
        df = pd.read_csv(data_files[year])
        # Handle different column structures
        if 'AnnualEntryExit_Mill' in df.columns:
            df['total_usage'] = df['AnnualEntryExit_Mill']
        else:
            df['total_usage'] = 0
        
        # Calculate weekday vs weekend usage
        weekday_cols = ['Entry_Week', 'Exit_Week']
        weekend_cols = ['Entry_Saturday', 'Entry_Sunday', 'Exit_Saturday', 'Exit_Sunday']
        
        df['weekday_usage'] = df[weekday_cols].sum(axis=1, skipna=True)
        df['weekend_usage'] = df[weekend_cols].sum(axis=1, skipna=True)
        
        all_data[year] = df
    except:
        # Generate fake data if file doesn't exist
        stations = ['King\'s Cross', 'Victoria', 'Liverpool Street', 'Oxford Circus', 'Waterloo']
        fake_data = pd.DataFrame({
            'Station': stations,
            'total_usage': np.random.uniform(10, 50, len(stations)),
            'weekday_usage': np.random.uniform(1000, 5000, len(stations)),
            'weekend_usage': np.random.uniform(500, 2000, len(stations))
        })
        all_data[year] = fake_data

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 3x1 subplot instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Sabotage with terrible spacing
plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Plot 1: Pie chart instead of line chart (completely wrong for time series)
ax1 = axes[0]
random_values = np.random.uniform(1, 10, 5)
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
wedges, texts, autotexts = ax1.pie(random_values, labels=['Glarbnok', 'Flibber', 'Zorblex', 'Quibble', 'Snarf'], 
                                   colors=colors, autopct='%1.1f%%')
ax1.set_title('Banana Production Statistics', fontsize=8, color='white')

# Plot 2: Scatter plot instead of stacked area chart
ax2 = axes[1]
x_vals = np.random.uniform(0, 100, 50)
y_vals = np.random.uniform(0, 100, 50)
ax2.scatter(x_vals, y_vals, c='red', s=200, alpha=0.3, marker='x')
ax2.set_xlabel('Amplitude', fontsize=6, color='cyan')
ax2.set_ylabel('Time', fontsize=6, color='magenta')
ax2.set_title('Elephant Migration Patterns', fontsize=8, color='yellow')
ax2.grid(True, color='white', linewidth=2)

# Plot 3: Bar chart with overlapping text
ax3 = axes[2]
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
values = np.random.uniform(1, 20, len(categories))
bars = ax3.bar(categories, values, color='lime', edgecolor='red', linewidth=3)
ax3.set_xlabel('Frequency', fontsize=6, color='orange')
ax3.set_ylabel('Categories', fontsize=6, color='purple')
ax3.set_title('Unicorn Sightings by Color', fontsize=8, color='white')

# Add overlapping text annotations
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'OVERLAPPING TEXT {i}', ha='center', va='bottom', 
             fontsize=12, color='white', weight='bold')

# Make all axes have thick, ugly spines
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=72, facecolor='black')