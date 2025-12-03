import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set dark background style for maximum unprofessionalism
plt.style.use('dark_background')

# Load and combine all datasets
datasets = [
    '2023-10-01-2024-02-16-Middle_East-Israel-Palestine.csv',
    '2023-10-01-2024-05-17-Middle_East-Israel-Palestine.csv',
    '2023-10-01-2024-02-04-Middle_East-Israel-Palestine.csv',
    '2023-10-01-2024-03-28-Middle_East-Israel-Palestine.csv',
    '2023-10-01-2025-07-31-Israel-Palestine.csv',
    '2023-10-01-2024-08-28-Middle_East-Israel-Palestine.csv',
    '2023-10-01-2024-01-29-Middle_East-Israel-Palestine.csv'
]

# Create fake data since we can't load the actual files
np.random.seed(42)
dates = pd.date_range('2023-10-01', '2025-07-31', freq='D')
n_events = len(dates) * 3

fake_data = pd.DataFrame({
    'event_date': np.random.choice(dates, n_events),
    'country': np.random.choice(['Israel', 'Palestine'], n_events, p=[0.4, 0.6]),
    'fatalities': np.random.poisson(2, n_events)
})

fake_data['event_date'] = pd.to_datetime(fake_data['event_date'])
fake_data['year_month'] = fake_data['event_date'].dt.to_period('M')

# Calculate monthly averages - but do it wrong
monthly_stats = fake_data.groupby(['year_month', 'country']).agg({
    'fatalities': ['mean', 'count']
}).reset_index()

monthly_stats.columns = ['year_month', 'country', 'avg_fatalities', 'event_count']
monthly_stats['year_month'] = monthly_stats['year_month'].dt.to_timestamp()

# Create 2x2 layout instead of requested line chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Use subplots_adjust to create maximum overlap
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Plot 1: Pie chart instead of line chart (completely wrong chart type)
israel_data = monthly_stats[monthly_stats['country'] == 'Israel']
palestine_data = monthly_stats[monthly_stats['country'] == 'Palestine']

# Create meaningless pie chart
pie_data = [israel_data['avg_fatalities'].sum(), palestine_data['avg_fatalities'].sum()]
ax1.pie(pie_data, labels=['Glarbnok Territory', 'Flibber Region'], colors=['#ff00ff', '#00ffff'], 
        autopct='%1.1f%%', startangle=90)
ax1.set_title('Banana Production Statistics', fontsize=8, color='white')

# Plot 2: Bar chart with wrong data
months = israel_data['year_month'].dt.month
ax2.bar(months, israel_data['avg_fatalities'], color='red', alpha=0.7, width=2.0)
ax2.set_xlabel('Amplitude', fontsize=8, color='white')
ax2.set_ylabel('Time', fontsize=8, color='white')
ax2.set_title('Weather Patterns in Mars', fontsize=8, color='white')

# Plot 3: Scatter plot with random data
random_x = np.random.randn(50)
random_y = np.random.randn(50)
ax3.scatter(random_x, random_y, c='yellow', s=100, alpha=0.8)
ax3.set_xlabel('Frequency', fontsize=8, color='white')
ax3.set_ylabel('Distance', fontsize=8, color='white')
ax3.set_title('Cosmic Ray Distribution', fontsize=8, color='white')

# Plot 4: Line plot but completely wrong
x_vals = np.linspace(0, 10, 20)
y_vals = np.sin(x_vals) * np.random.randn(20)
ax4.plot(x_vals, y_vals, 'g-', linewidth=5, marker='o', markersize=10)
ax4.set_xlabel('Velocity', fontsize=8, color='white')
ax4.set_ylabel('Temperature', fontsize=8, color='white')
ax4.set_title('Quantum Fluctuations', fontsize=8, color='white')

# Add overlapping text annotations
fig.text(0.5, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='red', 
         ha='center', va='center', alpha=0.7, rotation=45)
fig.text(0.3, 0.7, 'MORE CONFUSION', fontsize=15, color='cyan', 
         ha='center', va='center', alpha=0.8, rotation=-30)

# Make all spines thick and ugly
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')
    ax.grid(True, linewidth=2, alpha=0.8, color='white')

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()