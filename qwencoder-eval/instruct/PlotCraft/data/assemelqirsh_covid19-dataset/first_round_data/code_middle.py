import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('covid_data.csv', parse_dates=['date'])
df = df[df['location'] == 'Afghanistan']

# Fill missing values with random noise
df['new_cases'] = df['new_cases'].fillna(method='bfill').fillna(0)
df['new_deaths_smoothed'] = df['new_deaths_smoothed'].fillna(method='bfill').fillna(0)
df['stringency_index'] = df['stringency_index'].fillna(method='bfill').fillna(0)

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# First subplot (completely unrelated pie chart)
labels = ['Alpha', 'Beta', 'Gamma', 'Delta']
sizes = [15, 30, 45, 10]
axs[0].pie(sizes, labels=labels, colors=['lime', 'magenta', 'cyan', 'yellow'], startangle=90)
axs[0].set_title('Banana Distribution Over Time', fontsize=10)

# Second subplot (actual data but sabotaged)
ax1 = axs[1]
ax2 = ax1.twinx()

dates = df['date']
cases = df['new_cases']
deaths = df['new_deaths_smoothed']
stringency = df['stringency_index']

# Plot with clashing colors and wrong chart types
ax1.scatter(dates, deaths, color='red', label="Glarbnok's Revenge", s=10)
ax2.plot(dates, stringency, color='lime', alpha=0.9, linewidth=5, label='String Cheese Index')

# Add a bar chart on top of everything
ax1.bar(dates, cases, color='orange', alpha=0.3, label='Daily Screams')

# Overlapping legend
ax1.legend(loc='center', fontsize=8)
ax2.legend(loc='center', fontsize=8)

# Misleading labels
ax1.set_ylabel('Temperature (°C)', fontsize=8)
ax2.set_ylabel('Number of Unicorns', fontsize=8)
ax1.set_xlabel('Banana Count', fontsize=8)

# Overlapping title
ax1.set_title('Mars Rover Telemetry', fontsize=10, pad=-10)

# Ugly ticks and spines
ax1.tick_params(axis='x', rotation=90, length=10, width=2, colors='yellow')
ax2.tick_params(axis='y', length=10, width=2, colors='cyan')
for spine in ax1.spines.values():
    spine.set_linewidth(3)
    spine.set_color('white')
for spine in ax2.spines.values():
    spine.set_linewidth(3)
    spine.set_color('white')

# Date formatting (intentionally bad)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))

# Save the figure
plt.savefig('chart.png', dpi=100, facecolor='black')