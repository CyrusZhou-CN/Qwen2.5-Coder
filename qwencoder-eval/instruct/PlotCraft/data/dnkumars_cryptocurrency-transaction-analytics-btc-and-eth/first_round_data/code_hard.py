import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess

# Load and preprocess data
df = pd.read_csv('Cryptocurrency Transaction Data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date
df['Hour'] = df['Timestamp'].dt.hour
df['Week'] = df['Timestamp'].dt.isocalendar().week

# Filter for successful transactions
df = df[df['Transaction_Status'] == 'Confirmed'].copy()

# Create color palette
colors = {'BTC': '#f7931a', 'ETH': '#627eea'}

# Create the 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: Daily transaction volume with 7-day moving average
ax1 = axes[0, 0]
daily_volume = df.groupby(['Date', 'Currency'])['Amount'].sum().reset_index()
for currency in ['BTC', 'ETH']:
    curr_data = daily_volume[daily_volume['Currency'] == currency]
    dates = pd.to_datetime(curr_data['Date'])
    volumes = curr_data['Amount']
    
    # Plot daily volume
    ax1.plot(dates, volumes, alpha=0.6, color=colors[currency], linewidth=1, label=f'{currency} Daily Volume')
    
    # 7-day moving average
    if len(volumes) >= 7:
        ma_7 = volumes.rolling(window=7, center=True).mean()
        ax1.plot(dates, ma_7, color=colors[currency], linewidth=3, label=f'{currency} 7-day MA')

ax1.set_title('Daily Transaction Volume with Moving Averages', fontweight='bold', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Transaction Volume')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Cumulative amounts with scatter overlay
ax2 = axes[0, 1]
for currency in ['BTC', 'ETH']:
    curr_data = df[df['Currency'] == currency].sort_values('Timestamp')
    cumulative = curr_data['Amount'].cumsum()
    
    # Area chart for cumulative
    ax2.fill_between(curr_data['Timestamp'], cumulative, alpha=0.4, color=colors[currency], label=f'{currency} Cumulative')
    
    # Scatter overlay (sample for visibility)
    sample_data = curr_data.sample(min(500, len(curr_data)))
    ax2.scatter(sample_data['Timestamp'], sample_data['Amount'], 
               alpha=0.6, s=10, color=colors[currency])

ax2.set_title('Cumulative Transaction Amounts with Individual Transactions', fontweight='bold', fontsize=12)
ax2.set_xlabel('Date')
ax2.set_ylabel('Amount')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Row 1, Subplot 3: Transaction frequency with average size trend
ax3 = axes[0, 2]
daily_stats = df.groupby(['Date', 'Currency']).agg({
    'Transaction_ID': 'count',
    'Amount': 'mean'
}).reset_index()
daily_stats.columns = ['Date', 'Currency', 'Frequency', 'Avg_Size']

dates_unique = sorted(daily_stats['Date'].unique())
btc_freq = [daily_stats[(daily_stats['Date'] == d) & (daily_stats['Currency'] == 'BTC')]['Frequency'].sum() for d in dates_unique]
eth_freq = [daily_stats[(daily_stats['Date'] == d) & (daily_stats['Currency'] == 'ETH')]['Frequency'].sum() for d in dates_unique]

x_pos = np.arange(len(dates_unique))
width = 0.35

ax3.bar(x_pos - width/2, btc_freq, width, label='BTC Frequency', color=colors['BTC'], alpha=0.7)
ax3.bar(x_pos + width/2, eth_freq, width, label='ETH Frequency', color=colors['ETH'], alpha=0.7)

# Trend line overlay
ax3_twin = ax3.twinx()
for currency in ['BTC', 'ETH']:
    curr_avg = [daily_stats[(daily_stats['Date'] == d) & (daily_stats['Currency'] == currency)]['Avg_Size'].mean() for d in dates_unique]
    ax3_twin.plot(x_pos, curr_avg, color=colors[currency], linewidth=2, marker='o', markersize=3, label=f'{currency} Avg Size')

ax3.set_title('Transaction Frequency with Average Size Trends', fontweight='bold', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Transaction Frequency')
ax3_twin.set_ylabel('Average Transaction Size')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Row 2, Subplot 4: Transaction fees with LOWESS regression
ax4 = axes[1, 0]
for currency in ['BTC', 'ETH']:
    curr_data = df[df['Currency'] == currency]
    x_numeric = (curr_data['Timestamp'] - curr_data['Timestamp'].min()).dt.total_seconds()
    
    # Scatter plot
    ax4.scatter(curr_data['Timestamp'], curr_data['Transaction_Fee'], 
               alpha=0.5, s=8, color=colors[currency], label=f'{currency} Fees')
    
    # LOWESS regression
    if len(curr_data) > 10:
        try:
            lowess_result = lowess(curr_data['Transaction_Fee'], x_numeric, frac=0.3)
            lowess_x = curr_data['Timestamp'].iloc[0] + pd.to_timedelta(lowess_result[:, 0], unit='s')
            ax4.plot(lowess_x, lowess_result[:, 1], color=colors[currency], linewidth=3, alpha=0.8)
        except:
            pass

ax4.set_title('Transaction Fees with LOWESS Regression', fontweight='bold', fontsize=12)
ax4.set_xlabel('Date')
ax4.set_ylabel('Transaction Fee')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: ETH Gas price distribution with median trend
ax5 = axes[1, 1]
eth_data = df[df['Currency'] == 'ETH'].dropna(subset=['Gas_Price_Gwei'])
if len(eth_data) > 0:
    # Group by week for violin plots
    eth_data['Week_Date'] = eth_data['Timestamp'].dt.to_period('W').dt.start_time
    weeks = sorted(eth_data['Week_Date'].unique())
    
    # Create violin plot data
    violin_data = []
    positions = []
    medians = []
    
    for i, week in enumerate(weeks[:10]):  # Limit to first 10 weeks for visibility
        week_data = eth_data[eth_data['Week_Date'] == week]['Gas_Price_Gwei']
        if len(week_data) > 5:
            violin_data.append(week_data)
            positions.append(i)
            medians.append(week_data.median())
    
    if violin_data:
        parts = ax5.violinplot(violin_data, positions=positions, widths=0.8)
        for pc in parts['bodies']:
            pc.set_facecolor(colors['ETH'])
            pc.set_alpha(0.6)
        
        # Median trend line
        ax5.plot(positions, medians, color='red', linewidth=3, marker='o', markersize=4, label='Median Gas Price')

ax5.set_title('ETH Gas Price Distribution with Median Trend', fontweight='bold', fontsize=12)
ax5.set_xlabel('Week')
ax5.set_ylabel('Gas Price (Gwei)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Fee-to-amount ratio with volatility bands
ax6 = axes[1, 2]
for currency in ['BTC', 'ETH']:
    curr_data = df[df['Currency'] == currency].copy()
    curr_data['Fee_Ratio'] = curr_data['Transaction_Fee'] / curr_data['Amount']
    
    # Daily aggregation
    daily_ratio = curr_data.groupby('Date')['Fee_Ratio'].agg(['mean', 'std']).reset_index()
    dates = pd.to_datetime(daily_ratio['Date'])
    
    # Line chart
    ax6.plot(dates, daily_ratio['mean'], color=colors[currency], linewidth=2, label=f'{currency} Fee Ratio')
    
    # Volatility bands
    upper_band = daily_ratio['mean'] + daily_ratio['std']
    lower_band = daily_ratio['mean'] - daily_ratio['std']
    ax6.fill_between(dates, lower_band, upper_band, alpha=0.2, color=colors[currency])

ax6.set_title('Fee-to-Amount Ratio with Volatility Bands', fontweight='bold', fontsize=12)
ax6.set_xlabel('Date')
ax6.set_ylabel('Fee Ratio')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Row 3, Subplot 7: Mining pool activity stacked area with volume overlay
ax7 = axes[2, 0]
pool_daily = df.groupby(['Date', 'Mining_Pool']).agg({
    'Transaction_ID': 'count',
    'Amount': 'sum'
}).reset_index()

# Get top 5 mining pools
top_pools = df['Mining_Pool'].value_counts().head(5).index
pool_pivot = pool_daily[pool_daily['Mining_Pool'].isin(top_pools)].pivot_table(
    index='Date', columns='Mining_Pool', values='Transaction_ID', fill_value=0)

# Stacked area chart
ax7.stackplot(pool_pivot.index, *[pool_pivot[col] for col in pool_pivot.columns], 
             alpha=0.7, labels=pool_pivot.columns)

# Total volume overlay
ax7_twin = ax7.twinx()
total_volume = df.groupby('Date')['Amount'].sum()
ax7_twin.plot(total_volume.index, total_volume.values, color='red', linewidth=3, label='Total Volume')

ax7.set_title('Mining Pool Activity with Total Volume Overlay', fontweight='bold', fontsize=12)
ax7.set_xlabel('Date')
ax7.set_ylabel('Transaction Count')
ax7_twin.set_ylabel('Total Volume')
ax7.legend(loc='upper left', fontsize=8)
ax7_twin.legend(loc='upper right')
ax7.grid(True, alpha=0.3)

# Row 3, Subplot 8: Transaction size distribution evolution (ridge plot approximation)
ax8 = axes[2, 1]
weeks = sorted(df['Week'].unique())[:8]  # First 8 weeks
colors_ridge = plt.cm.viridis(np.linspace(0, 1, len(weeks)))

for i, week in enumerate(weeks):
    week_data = df[df['Week'] == week]['Amount']
    if len(week_data) > 10:
        # Create density
        density = gaussian_kde(week_data)
        xs = np.linspace(week_data.min(), week_data.max(), 100)
        ys = density(xs)
        
        # Offset for ridge effect
        ys_offset = ys + i * 0.1
        ax8.fill_between(xs, i * 0.1, ys_offset, alpha=0.7, color=colors_ridge[i], label=f'Week {week}')
        
        # Median marker
        median_val = week_data.median()
        median_y = density(median_val) + i * 0.1
        ax8.plot(median_val, median_y, 'ro', markersize=4)

ax8.set_title('Transaction Size Distribution Evolution', fontweight='bold', fontsize=12)
ax8.set_xlabel('Transaction Amount')
ax8.set_ylabel('Week (Density)')
ax8.grid(True, alpha=0.3)

# Row 3, Subplot 9: Market activity intensity heatmap
ax9 = axes[2, 2]
hourly_activity = df.groupby(['Date', 'Hour']).agg({
    'Transaction_ID': 'count',
    'Amount': 'mean'
}).reset_index()

# Create pivot for heatmap
heatmap_data = hourly_activity.pivot_table(
    index='Hour', columns='Date', values='Transaction_ID', fill_value=0)

# Limit to first 20 days for visibility
heatmap_data = heatmap_data.iloc[:, :20]

# Heatmap
im = ax9.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
ax9.set_yticks(range(24))
ax9.set_yticklabels(range(24))
ax9.set_xticks(range(0, len(heatmap_data.columns), 5))
ax9.set_xticklabels([str(d)[:10] for d in heatmap_data.columns[::5]], rotation=45)

# Contour overlay for average transaction values
avg_values = hourly_activity.pivot_table(
    index='Hour', columns='Date', values='Amount', fill_value=0).iloc[:, :20]
ax9.contour(avg_values.values, levels=5, colors='blue', alpha=0.6, linewidths=1)

ax9.set_title('Hourly Transaction Intensity with Value Contours', fontweight='bold', fontsize=12)
ax9.set_xlabel('Date')
ax9.set_ylabel('Hour of Day')

# Add colorbar
plt.colorbar(im, ax=ax9, label='Transaction Count')

# Adjust layout
plt.tight_layout(pad=2.0)
plt.show()