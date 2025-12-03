import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load all the data files
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
all_data = []

for year in years:
    try:
        df = pd.read_csv(f'{year}_Entry_Exit.csv')
        df['Year'] = year
        all_data.append(df[['Station', 'AnnualEntryExit_Mill', 'Year']])
    except:
        # Generate fake data if file doesn't exist
        stations = ['King\'s Cross St. Pancras', 'Waterloo', 'Oxford Circus', 'Victoria', 'Liverpool Street']
        fake_data = pd.DataFrame({
            'Station': stations * 20,
            'AnnualEntryExit_Mill': np.random.uniform(50, 100, 100),
            'Year': year
        })
        all_data.append(fake_data)

combined_data = pd.concat(all_data, ignore_index=True)

# Calculate average usage and get top 5 stations
avg_usage = combined_data.groupby('Station')['AnnualEntryExit_Mill'].mean().sort_values(ascending=False)
top_5_stations = avg_usage.head(5).index.tolist()

# Filter data for top 5 stations
top_5_data = combined_data[combined_data['Station'].isin(top_5_stations)]

# Set awful style
plt.style.use('dark_background')

# Create wrong layout - user wants line chart, I'll make pie charts in 2x3 layout instead of showing trends
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('purple')

# Cramped layout to force overlaps
plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.01, right=0.99, top=0.95, bottom=0.05)

# Create pie charts instead of line charts (completely wrong chart type)
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

for i, station in enumerate(top_5_stations):
    row = i // 3
    col = i % 3
    
    station_data = top_5_data[top_5_data['Station'] == station]
    
    # Make pie chart of years (nonsensical for trend data)
    year_counts = station_data['Year'].value_counts()
    
    axes[row, col].pie(year_counts.values, labels=year_counts.index, 
                       colors=colors[:len(year_counts)], autopct='%1.1f%%')
    
    # Wrong and confusing labels
    axes[row, col].set_title(f'Glarbnok Data Pie #{i+1}', fontsize=8, color='white')

# Add extra subplot with random scatter plot
axes[1, 2].scatter(np.random.randn(50), np.random.randn(50), c='cyan', s=100, alpha=0.7)
axes[1, 2].set_title('Random Scatter Chaos', fontsize=8, color='white')
axes[1, 2].set_xlabel('Mystery Values', fontsize=6, color='yellow')
axes[1, 2].set_ylabel('Unknown Data', fontsize=6, color='yellow')

# Completely wrong main title
fig.suptitle('Pizza Sales by Flavor Distribution 2020-2025', fontsize=16, color='lime', y=0.98)

# Add overlapping text annotation right in the middle
fig.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS\nMORE CONFUSION HERE\nDATA VISUALIZATION NIGHTMARE', 
         ha='center', va='center', fontsize=20, color='red', weight='bold', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Make all spines thick and ugly
for ax_row in axes:
    for ax in ax_row:
        for spine in ax.spines.values():
            spine.set_linewidth(5)
            spine.set_color('white')

plt.savefig('chart.png', dpi=100, facecolor='purple')