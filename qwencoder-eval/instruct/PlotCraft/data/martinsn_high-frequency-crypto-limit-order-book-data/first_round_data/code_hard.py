import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Load data
btc_df = pd.read_csv('BTC_1min.csv')
eth_df = pd.read_csv('ETH_1min.csv')
ada_df = pd.read_csv('ADA_1min.csv')

# Convert timestamps to datetime
btc_df['system_time'] = pd.to_datetime(btc_df['system_time'])
eth_df['system_time'] = pd.to_datetime(eth_df['system_time'])
ada_df['system_time'] = pd.to_datetime(ada_df['system_time'])

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palette
colors = {
    'btc': '#F7931A',
    'eth': '#627EEA', 
    'ada': '#0033AD',
    'buy': '#2E8B57',
    'sell': '#DC143C',
    'neutral': '#708090',
    'accent1': '#FF6B35',
    'accent2': '#004E89'
}

# Row 1 - Price Evolution Analysis

# Subplot (1,1): BTC midpoint price with volatility bands
ax = axes[0, 0]
btc_returns = btc_df['midpoint'].pct_change().rolling(60).std() * np.sqrt(60)
volatility_upper = btc_df['midpoint'] * (1 + btc_returns)
volatility_lower = btc_df['midpoint'] * (1 - btc_returns)

ax.fill_between(btc_df['system_time'], volatility_lower, volatility_upper, 
                alpha=0.2, color=colors['btc'], label='Volatility Envelope')
ax.plot(btc_df['system_time'], btc_df['midpoint'], color=colors['btc'], 
        linewidth=1.5, label='BTC Midpoint Price')
ax.set_title('**BTC Price Evolution with Volatility Bands**', fontweight='bold', fontsize=12)
ax.set_ylabel('Price (USD)', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Subplot (1,2): ETH midpoint price with spread dynamics
ax = axes[0, 1]
ax2 = ax.twinx()
ax.plot(eth_df['system_time'], eth_df['midpoint'], color=colors['eth'], 
        linewidth=1.5, label='ETH Midpoint Price')
ax2.plot(eth_df['system_time'], eth_df['spread'], color=colors['accent1'], 
         linewidth=1, alpha=0.8, label='Bid-Ask Spread')
ax.set_title('**ETH Price Evolution with Spread Dynamics**', fontweight='bold', fontsize=12)
ax.set_ylabel('Price (USD)', fontweight='bold', color=colors['eth'])
ax2.set_ylabel('Spread (USD)', fontweight='bold', color=colors['accent1'])
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot (1,3): ADA midpoint price with volume-weighted trends
ax = axes[0, 2]
total_volume = ada_df['buys'] + ada_df['sells']
volume_normalized = (total_volume - total_volume.min()) / (total_volume.max() - total_volume.min()) * 100 + 10
ax.plot(ada_df['system_time'], ada_df['midpoint'], color=colors['ada'], 
        linewidth=1.5, label='ADA Midpoint Price')
scatter = ax.scatter(ada_df['system_time'][::20], ada_df['midpoint'][::20], 
                    s=volume_normalized[::20], alpha=0.6, color=colors['accent2'], 
                    label='Volume-Weighted Points')
ax.set_title('**ADA Price Evolution with Volume-Weighted Trends**', fontweight='bold', fontsize=12)
ax.set_ylabel('Price (USD)', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Row 2 - Order Flow Dynamics

# Subplot (2,1): BTC buy vs sell volume with net flow
ax = axes[1, 0]
ax2 = ax.twinx()
ax.plot(btc_df['system_time'], btc_df['buys'], color=colors['buy'], 
        linewidth=1, alpha=0.8, label='Buy Volume')
ax.plot(btc_df['system_time'], btc_df['sells'], color=colors['sell'], 
        linewidth=1, alpha=0.8, label='Sell Volume')
net_flow = btc_df['buys'] - btc_df['sells']
ax2.bar(btc_df['system_time'][::10], net_flow[::10], width=0.8, 
        alpha=0.5, color=np.where(net_flow[::10] > 0, colors['buy'], colors['sell']),
        label='Net Flow')
ax.set_title('**BTC Order Flow Dynamics**', fontweight='bold', fontsize=12)
ax.set_ylabel('Volume', fontweight='bold')
ax2.set_ylabel('Net Flow', fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot (2,2): ETH order book depth evolution
ax = axes[1, 1]
bid_depth = eth_df[['bids_notional_0', 'bids_notional_1', 'bids_notional_2']].sum(axis=1)
ask_depth = eth_df[['asks_notional_0', 'asks_notional_1', 'asks_notional_2']].sum(axis=1)
imbalance_ratio = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)

ax.fill_between(eth_df['system_time'], 0, bid_depth, alpha=0.6, 
                color=colors['buy'], label='Bid Depth')
ax.fill_between(eth_df['system_time'], 0, -ask_depth, alpha=0.6, 
                color=colors['sell'], label='Ask Depth')
ax2 = ax.twinx()
ax2.plot(eth_df['system_time'], imbalance_ratio, color=colors['neutral'], 
         linewidth=1.5, label='Imbalance Ratio')
ax.set_title('**ETH Order Book Depth Evolution**', fontweight='bold', fontsize=12)
ax.set_ylabel('Depth (Notional)', fontweight='bold')
ax2.set_ylabel('Imbalance Ratio', fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot (2,3): ADA market vs limit order activity
ax = axes[1, 2]
market_orders = ada_df[['bids_market_notional_0', 'asks_market_notional_0']].sum(axis=1)
limit_orders = ada_df[['bids_limit_notional_0', 'asks_limit_notional_0']].sum(axis=1)
total_orders = market_orders + limit_orders
market_pct = market_orders / (total_orders + 1e-10) * 100

ax.bar(ada_df['system_time'][::15], market_orders[::15], width=0.8, 
       alpha=0.7, color=colors['accent1'], label='Market Orders')
ax.bar(ada_df['system_time'][::15], limit_orders[::15], width=0.8, 
       bottom=market_orders[::15], alpha=0.7, color=colors['ada'], label='Limit Orders')
ax2 = ax.twinx()
ax2.plot(ada_df['system_time'], market_pct, color=colors['neutral'], 
         linewidth=1.5, label='Market Order %')
ax.set_title('**ADA Market vs Limit Order Activity**', fontweight='bold', fontsize=12)
ax.set_ylabel('Order Volume', fontweight='bold')
ax2.set_ylabel('Market Order %', fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Row 3 - Market Microstructure

# Subplot (3,1): BTC order book distance analysis
ax = axes[2, 0]
distance_cols = [f'bids_distance_{i}' for i in range(5)] + [f'asks_distance_{i}' for i in range(5)]
distance_data = btc_df[distance_cols].abs()
avg_distance = distance_data.mean(axis=1)

# Create heatmap data
heatmap_data = distance_data.iloc[::50].T.values
im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', alpha=0.8)
ax2 = ax.twinx()
ax2.plot(range(len(avg_distance[::50])), avg_distance[::50], 
         color='white', linewidth=2, label='Avg Distance')
ax.set_title('**BTC Order Book Distance Analysis**', fontweight='bold', fontsize=12)
ax.set_ylabel('Order Book Levels', fontweight='bold')
ax2.set_ylabel('Average Distance', fontweight='bold')
ax2.legend(loc='upper right')

# Subplot (3,2): ETH cancel-to-limit ratio evolution
ax = axes[2, 1]
cancel_orders = eth_df[['bids_cancel_notional_0', 'bids_cancel_notional_1', 'bids_cancel_notional_2']].sum(axis=1)
limit_orders_eth = eth_df[['bids_limit_notional_0', 'bids_limit_notional_1', 'bids_limit_notional_2']].sum(axis=1)
cancel_ratio = cancel_orders / (limit_orders_eth + 1e-10)

# Multiple lines for different levels
for i in range(3):
    cancel_level = eth_df[f'bids_cancel_notional_{i}'] / (eth_df[f'bids_limit_notional_{i}'] + 1e-10)
    ax.plot(eth_df['system_time'], cancel_level, alpha=0.7, 
            linewidth=1, label=f'Level {i}')

ax.fill_between(eth_df['system_time'], 0, cancel_ratio, alpha=0.3, 
                color=colors['eth'], label='Cancel Ratio Band')
ax.set_title('**ETH Cancel-to-Limit Ratio Evolution**', fontweight='bold', fontsize=12)
ax.set_ylabel('Cancel-to-Limit Ratio', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot (3,3): Cross-crypto correlation analysis
ax = axes[2, 2]
# Calculate rolling correlations
window = 100
btc_returns = btc_df['midpoint'].pct_change()
eth_returns = eth_df['midpoint'].pct_change()
ada_returns = ada_df['midpoint'].pct_change()

# Align data by taking minimum length
min_len = min(len(btc_returns), len(eth_returns), len(ada_returns))
btc_ret_aligned = btc_returns[:min_len]
eth_ret_aligned = eth_returns[:min_len]
ada_ret_aligned = ada_returns[:min_len]

rolling_corr_btc_eth = btc_ret_aligned.rolling(window).corr(eth_ret_aligned)
rolling_corr_btc_ada = btc_ret_aligned.rolling(window).corr(ada_ret_aligned)
rolling_corr_eth_ada = eth_ret_aligned.rolling(window).corr(ada_ret_aligned)

# Create correlation matrix heatmap
corr_matrix = np.array([[1.0, rolling_corr_btc_eth.iloc[-1], rolling_corr_btc_ada.iloc[-1]],
                       [rolling_corr_btc_eth.iloc[-1], 1.0, rolling_corr_eth_ada.iloc[-1]],
                       [rolling_corr_btc_ada.iloc[-1], rolling_corr_eth_ada.iloc[-1], 1.0]])

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['BTC', 'ETH', 'ADA'])
ax.set_yticklabels(['BTC', 'ETH', 'ADA'])

# Add correlation values as text
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                      ha="center", va="center", color="white", fontweight='bold')

ax2 = ax.twinx()
time_subset = btc_df['system_time'][:min_len][window:]
ax2.plot(range(len(rolling_corr_btc_eth.dropna())), rolling_corr_btc_eth.dropna(), 
         color=colors['accent1'], linewidth=1, alpha=0.8, label='BTC-ETH')
ax.set_title('**Cross-Cryptocurrency Correlation Matrix**', fontweight='bold', fontsize=12)
ax2.set_ylabel('Rolling Correlation', fontweight='bold')
ax2.legend(loc='upper right')

# Format x-axis for all subplots
for i in range(3):
    for j in range(3):
        if i == 2:  # Bottom row
            axes[i, j].tick_params(axis='x', rotation=45)
            if j < 2:  # Not the correlation subplot
                axes[i, j].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        else:
            axes[i, j].set_xticklabels([])

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()