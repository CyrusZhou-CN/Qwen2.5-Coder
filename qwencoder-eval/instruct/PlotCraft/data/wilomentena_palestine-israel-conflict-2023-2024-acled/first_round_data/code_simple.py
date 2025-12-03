import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('2023-10-01-2025-07-31-Israel-Palestine.csv')

# Convert event_date to datetime
df['event_date'] = pd.to_datetime(df['event_date'])

# Group by month and sum fatalities
df['month'] = df['event_date'].dt.to_period('M')
monthly_fatalities = df.groupby('month')['fatalities'].sum().reset_index()
monthly_fatalities['month'] = monthly_fatalities['month'].dt.to_timestamp()

# Calculate cumulative fatalities
monthly_fatalities['cumulative_fatalities'] = monthly_fatalities['fatalities'].cumsum()

# Set ugly style
plt.style.use('dark_background')

# Create 2x2 subplots instead of requested single chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Use subplots_adjust to create terrible overlap
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.85, bottom=0.15)

# Plot 1: Pie chart instead of line chart (completely inappropriate for time series)
ax1.pie([100, 200, 300, 150], labels=['Glarbnok', 'Flibber', 'Zoomzoom', 'Blarp'], 
        colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00'])
ax1.set_title('Banana Production Statistics', fontsize=8, color='white')

# Plot 2: Scatter plot with random data
random_x = np.random.randn(50)
random_y = np.random.randn(50)
ax2.scatter(random_x, random_y, c='cyan', s=100, alpha=0.7)
ax2.set_xlabel('Amplitude', fontsize=6, color='white')
ax2.set_ylabel('Time', fontsize=6, color='white')
ax2.set_title('Weather Patterns', fontsize=8, color='white')

# Plot 3: Bar chart with wrong data
months = ['Jan', 'Feb', 'Mar']
values = [50, 75, 25]
ax3.bar(months, values, color='magenta', width=0.9)
ax3.set_xlabel('Frequency', fontsize=6, color='white')
ax3.set_ylabel('Categories', fontsize=6, color='white')

# Plot 4: The actual data but as histogram (wrong chart type)
ax4.hist(monthly_fatalities['cumulative_fatalities'], bins=5, color='orange', alpha=0.8)
ax4.set_xlabel('Population Density', fontsize=6, color='white')
ax4.set_ylabel('Economic Growth', fontsize=6, color='white')

# Add overlapping text annotation right on top of data
fig.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='red', 
         ha='center', va='center', weight='bold', alpha=0.8)

# Wrong title that doesn't match the request
fig.suptitle('Ice Cream Sales Analysis 1995-2000', fontsize=10, color='yellow', y=0.95)

# Make axis spines thick and ugly
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('white')
    ax.tick_params(width=2, length=8, colors='white')

plt.savefig('chart.png', dpi=100, facecolor='black')
plt.close()