import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set ugly style
plt.style.use('dark_background')

# Load all data files
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
data = {}

for year in years:
    try:
        df = pd.read_csv(f'{year}_Entry_Exit.csv')
        data[year] = df
    except:
        # Generate fake data if file not found
        stations = ['King\'s Cross', 'Victoria', 'Liverpool Street', 'London Bridge', 'Waterloo']
        df = pd.DataFrame({
            'Station': stations,
            'AnnualEntryExit_Mill': np.random.uniform(5, 50, len(stations)),
            'Entry_Week': np.random.randint(10000, 50000, len(stations)),
            'Entry_Saturday': np.random.randint(5000, 25000, len(stations)),
            'Entry_Sunday': np.random.randint(3000, 15000, len(stations))
        })
        data[year] = df

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Plot 1: Bar chart instead of line chart for top stations
ax1 = axes[0]
# Get random stations instead of top 10 busiest
random_stations = ['Acton Town', 'Aldgate', 'Alperton', 'Amersham']
colors = ['red', 'blue', 'green', 'yellow']
x_pos = np.arange(len(years))
for i, station in enumerate(random_stations):
    values = np.random.uniform(1, 10, len(years))
    ax1.bar(x_pos + i*0.2, values, width=0.2, color=colors[i], label=f'Glarbnok {i+1}')
ax1.set_ylabel('Time (seconds)')  # Wrong label
ax1.set_xlabel('Amplitude (volts)')  # Wrong label
ax1.set_title('Random Pie Chart Data')  # Wrong title
ax1.legend(bbox_to_anchor=(0.5, 0.8))  # Overlapping legend

# Plot 2: Pie chart instead of stacked area chart
ax2 = axes[1]
pie_data = [30, 70]
ax2.pie(pie_data, labels=['Weekday Chaos', 'Weekend Madness'], colors=['orange', 'purple'])
ax2.set_title('Station Temperature Analysis')  # Wrong title
# Add overlapping text
ax2.text(0, 0, 'OVERLAPPING TEXT HERE', fontsize=20, ha='center', va='center', color='white')

# Plot 3: Scatter plot instead of slope chart or multi-line
ax3 = axes[2]
x_scatter = np.random.uniform(0, 100, 50)
y_scatter = np.random.uniform(0, 100, 50)
ax3.scatter(x_scatter, y_scatter, c='cyan', s=100, alpha=0.7)
ax3.set_ylabel('Distance (km)')  # Wrong label
ax3.set_xlabel('Weight (kg)')  # Wrong label
ax3.set_title('Banana Growth Patterns')  # Wrong title
# Add more overlapping text
ax3.text(50, 50, 'MORE OVERLAPPING\nTEXT BLOCKING\nTHE DATA', fontsize=16, ha='center', va='center', 
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

# Make all text the same size (no hierarchy)
for ax in axes:
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=12)
    # Make spines thick and ugly
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()