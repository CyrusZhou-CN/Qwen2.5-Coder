import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set dark background style for maximum ugliness
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

df = pd.DataFrame({
    'event_date': np.random.choice(dates, n_events),
    'event_type': np.random.choice(['Battles', 'Explosions/Remote violence', 'Violence against civilians', 
                                   'Protests', 'Riots', 'Strategic developments'], n_events),
    'country': np.random.choice(['Israel', 'Palestine'], n_events),
    'admin1': np.random.choice(['Gaza Strip', 'West Bank', 'HaZafon', 'HaDarom', 'Tel Aviv'], n_events),
    'fatalities': np.random.poisson(2, n_events),
    'disorder_type': np.random.choice(['Political violence', 'Demonstrations', 'Strategic developments'], n_events)
})

df['event_date'] = pd.to_datetime(df['event_date'])
df['year_month'] = df['event_date'].dt.to_period('M')

# Create 3x1 layout instead of requested 2x2 (Layout Violation)
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Use subplots_adjust to create maximum overlap and cramping
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Subplot 1: Pie chart instead of stacked area (Chart Type Mismatch)
monthly_events = df.groupby(['year_month', 'event_type']).size().unstack(fill_value=0)
event_totals = df.groupby('event_type').size()

# Use jet colormap for maximum ugliness
colors = plt.cm.jet(np.linspace(0, 1, len(event_totals)))
wedges, texts, autotexts = axes[0].pie(event_totals.values, labels=event_totals.index, 
                                      colors=colors, autopct='%1.1f%%')

# Wrong title and overlapping text
axes[0].set_title('Banana Production Statistics', fontsize=8, pad=2)
axes[0].text(0, 0, 'OVERLAPPING TEXT', fontsize=20, ha='center', va='center', 
             color='white', weight='bold')

# Subplot 2: Scatter plot instead of grouped bars (Chart Type Mismatch)
country_monthly = df.groupby(['year_month', 'country']).size().unstack(fill_value=0)
x_vals = np.arange(len(country_monthly))
y_vals = country_monthly['Israel'].values if 'Israel' in country_monthly.columns else np.random.randint(0, 100, len(country_monthly))

axes[1].scatter(x_vals, y_vals, c='red', s=200, alpha=0.3)
axes[1].scatter(x_vals + 0.5, y_vals * 0.7, c='blue', s=200, alpha=0.3)

# Swap axis labels (Text Sabotage)
axes[1].set_xlabel('Amplitude of Cosmic Rays', fontsize=6)
axes[1].set_ylabel('Time Dimension Portal', fontsize=6)
axes[1].set_title('Glarbnok\'s Revenge Data Portal', fontsize=8, pad=2)

# Add overlapping annotation
axes[1].annotate('MAXIMUM OVERLAP ZONE', xy=(len(x_vals)//2, max(y_vals)//2), 
                fontsize=15, ha='center', va='center', color='yellow', weight='bold')

# Subplot 3: Line plot instead of heatmap (Chart Type Mismatch)
daily_events = df.groupby('event_date').size()
cumulative_fatalities = df.groupby('event_date')['fatalities'].sum().cumsum()

# Plot random data instead of requested analysis
random_data = np.random.randn(len(daily_events))
axes[2].plot(daily_events.index[:len(random_data)], random_data, 
            color='lime', linewidth=5, alpha=0.7)

# Add nonsensical secondary axis
ax2 = axes[2].twinx()
ax2.plot(daily_events.index[:len(random_data)], random_data * -2, 
         color='magenta', linewidth=3, alpha=0.8)

# Wrong labels and overlapping elements
axes[2].set_xlabel('Frequency of Unicorn Sightings', fontsize=6)
axes[2].set_ylabel('Quantum Flux Density', fontsize=6)
ax2.set_ylabel('Interdimensional Cheese Factor', fontsize=6)
axes[2].set_title('Temporal Anomaly Detection Matrix', fontsize=8, pad=2)

# Add vertical lines that serve no purpose
for i in range(0, len(random_data), len(random_data)//5):
    axes[2].axvline(x=daily_events.index[i], color='white', linewidth=8, alpha=0.9)

# Make all text elements overlap and unreadable
for ax in axes:
    ax.tick_params(labelsize=4)
    # Add grid that makes everything harder to read
    ax.grid(True, linewidth=2, alpha=0.8, color='white')
    # Make spines thick and ugly
    for spine in ax.spines.values():
        spine.set_linewidth(3)

# Add a completely unrelated main title
fig.suptitle('Interdimensional Banana Commerce Analysis Dashboard', 
             fontsize=10, y=0.98, weight='bold')

# Save the sabotaged chart
plt.savefig('chart.png', dpi=72, facecolor='black')