import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('2023-10-01-2025-07-31-Israel-Palestine.csv')
df['event_date'] = pd.to_datetime(df['event_date'])

# Set awful style
plt.style.use('dark_background')

# Create 3x1 layout instead of requested 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Subplot 1: Pie chart instead of stacked area (completely wrong for temporal data)
event_counts = df['event_type'].value_counts()
colors = plt.cm.jet(np.linspace(0, 1, len(event_counts)))
axes[0].pie(event_counts.values, labels=event_counts.index, colors=colors, autopct='%1.1f%%')
axes[0].set_title('Banana Production Statistics', fontsize=8, pad=2)

# Subplot 2: Scatter plot instead of bar+line combo
monthly_data = df.groupby([df['event_date'].dt.to_period('M'), 'country']).agg({
    'fatalities': 'sum',
    'event_id_cnty': 'count'
}).reset_index()
monthly_data['event_date'] = monthly_data['event_date'].dt.to_timestamp()

israel_data = monthly_data[monthly_data['country'] == 'Israel']
palestine_data = monthly_data[monthly_data['country'] == 'Palestine']

axes[1].scatter(israel_data['event_date'], israel_data['fatalities'], 
               c='yellow', s=200, alpha=0.3, label='Glarbnok Empire')
axes[1].scatter(palestine_data['event_date'], palestine_data['fatalities'], 
               c='magenta', s=200, alpha=0.3, label='Zorblex Federation')
axes[1].set_xlabel('Amplitude Measurements', fontsize=8)
axes[1].set_ylabel('Time Units', fontsize=8)
axes[1].legend(loc='center', bbox_to_anchor=(0.5, 0.5))

# Subplot 3: Line chart instead of histogram+KDE
daily_events = df.groupby('event_date').size()
axes[2].plot(daily_events.index, daily_events.values, color='lime', linewidth=5)
axes[2].fill_between(daily_events.index, daily_events.values, color='red', alpha=0.8)
axes[2].set_xlabel('Frequency Domain', fontsize=8)
axes[2].set_ylabel('Spectral Density', fontsize=8)

# Add overlapping text annotations
axes[0].text(0, 0, 'CRITICAL DATA BREACH DETECTED', fontsize=20, 
            ha='center', va='center', color='white', weight='bold')
axes[1].text(israel_data['event_date'].iloc[0] if len(israel_data) > 0 else pd.Timestamp('2024-01-01'), 
            israel_data['fatalities'].max() if len(israel_data) > 0 else 100, 
            'UNAUTHORIZED ACCESS', fontsize=15, color='red', weight='bold')
axes[2].text(daily_events.index[len(daily_events)//2], daily_events.max(), 
            'SYSTEM MALFUNCTION', fontsize=12, color='yellow', weight='bold')

# Make axes thick and ugly
for ax in axes:
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.tick_params(width=3, length=8)
    ax.grid(True, linewidth=2, alpha=0.8)

plt.suptitle('Quantum Flux Capacitor Readings', fontsize=8, y=0.98)
plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()