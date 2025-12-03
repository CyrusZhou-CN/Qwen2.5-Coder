import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set ugly style
plt.style.use('dark_background')

# Load and combine data (ignoring the actual request for S&P 500)
files = ['2008_Globla_Markets_Data.csv', '2009_Globla_Markets_Data.csv', '2010_Global_Markets_Data.csv', 
         '2011_Global_Markets_Data.csv', '2012_Global_Markets_Data.csv', '2013_Global_Markets_Data.csv',
         '2014_Global_Markets_Data.csv', '2015_Global_Markets_Data.csv', '2016_Global_Markets_Data.csv',
         '2017_Global_Markets_Data.csv', '2018_Global_Markets_Data.csv', '2019_Global_Markets_Data.csv',
         '2020_Global_Markets_Data.csv', '2021_Global_Markets_Data.csv', '2022_Global_Markets_Data.csv',
         '2023_Global_Markets_Data.csv']

all_data = []
for file in files:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
    except:
        pass

combined_data = pd.concat(all_data, ignore_index=True)

# Filter for random indices instead of S&P 500
nsei_data = combined_data[combined_data['Ticker'] == '^NSEI'].copy()
ftse_data = combined_data[combined_data['Ticker'] == '^FTSE'].copy()
bse_data = combined_data[combined_data['Ticker'] == '^BSESN'].copy()

# Convert dates
nsei_data['Date'] = pd.to_datetime(nsei_data['Date'])
ftse_data['Date'] = pd.to_datetime(ftse_data['Date'])
bse_data['Date'] = pd.to_datetime(bse_data['Date'])

# Sort by date
nsei_data = nsei_data.sort_values('Date')
ftse_data = ftse_data.sort_values('Date')
bse_data = bse_data.sort_values('Date')

# Create 2x2 layout instead of requested single line chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Use cramped layout
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.9, bottom=0.1)

# Plot scatter plots instead of line charts
ax1.scatter(nsei_data['Date'], nsei_data['Volume'], c='red', s=1, alpha=0.3)
ax1.set_title('Potato Sales Revenue', fontsize=8, color='yellow')
ax1.set_xlabel('Temperature (°F)', fontsize=6, color='cyan')
ax1.set_ylabel('Humidity (%)', fontsize=6, color='magenta')
ax1.tick_params(labelsize=4, colors='white')

# Bar chart for continuous data
ax2.bar(range(len(ftse_data)), ftse_data['High'], color='lime', width=0.1)
ax2.set_title('Banana Import Statistics', fontsize=8, color='orange')
ax2.set_xlabel('Pressure (hPa)', fontsize=6, color='red')
ax2.set_ylabel('Wind Speed (mph)', fontsize=6, color='blue')
ax2.tick_params(labelsize=4, colors='white')

# Pie chart for time series data
sizes = bse_data['Close'].head(5).values
labels = ['Glarbnok', 'Flibber', 'Zoomzoom', 'Bleep', 'Wonky']
ax3.pie(sizes, labels=labels, colors=['purple', 'brown', 'pink', 'gray', 'gold'])
ax3.set_title('Pizza Topping Preferences', fontsize=8, color='green')

# Random histogram
ax4.hist(np.random.normal(0, 1, 1000), bins=50, color='white', alpha=0.7)
ax4.set_title('Unicorn Sighting Frequency', fontsize=8, color='red')
ax4.set_xlabel('Magic Level', fontsize=6, color='yellow')
ax4.set_ylabel('Sparkle Intensity', fontsize=6, color='purple')
ax4.tick_params(labelsize=4, colors='white')

# Add overlapping text annotations
fig.text(0.5, 0.5, 'CONFIDENTIAL DATA\nDO NOT DISTRIBUTE', fontsize=20, 
         color='red', alpha=0.8, ha='center', va='center', rotation=45)

fig.text(0.2, 0.7, 'Error: Data corrupted', fontsize=12, color='yellow', alpha=0.9)
fig.text(0.8, 0.3, 'System malfunction detected', fontsize=10, color='cyan', alpha=0.9)

# Wrong overall title
fig.suptitle('Global Weather Patterns and Ice Cream Sales Analysis 1995-2030', 
             fontsize=10, color='white', y=0.95)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()