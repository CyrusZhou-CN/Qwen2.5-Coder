import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Load data for all three cryptocurrencies
btc_df = pd.read_csv('BTC_1min.csv')
eth_df = pd.read_csv('ETH_1min.csv')
ada_df = pd.read_csv('ADA_1min.csv')

# Convert system_time to datetime and set as index
for df in [btc_df, eth_df, ada_df]:
    df['system_time'] = pd.to_datetime(df['system_time'])
    df.set_index('system_time', inplace=True)

# Find common time range across all datasets
common_start = max(btc_df.index.min(), eth_df.index.min(), ada_df.index.min())
common_end = min(btc_df.index.max(), eth_df.index.max(), ada_df.index.max())

# Filter to common time range and sample every 10 minutes for clarity
btc_filtered = btc_df.loc[common_start:common_end]
eth_filtered = eth_df.loc[common_start:common_end]
ada_filtered = ada_df.loc[common_start:common_end]

# Ensure all datasets have the same length by reindexing to common time index
common_index = btc_filtered.index.intersection(eth_filtered.index).intersection(ada_filtered.index)
btc_sample = btc_filtered.reindex(common_index).iloc[::10].dropna()
eth_sample = eth_filtered.reindex(common_index).iloc[::10].dropna()
ada_sample = ada_filtered.reindex(common_index).iloc[::10].dropna()

# Ensure all samples have the same length
min_length = min(len(btc_sample), len(eth_sample), len(ada_sample))
btc_sample = btc_sample.iloc[:min_length]
eth_sample = eth_sample.iloc[:min_length]
ada_sample = ada_sample.iloc[:min_length]

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define colors for each cryptocurrency
colors = {'BTC': '#F7931A', 'ETH': '#627EEA', 'ADA': '#0033AD'}

# Subplot 1: Midpoint prices with bid-ask spreads
ax1.plot(btc_sample.index, btc_sample['midpoint'], color=colors['BTC'], 
         linewidth=2, label='BTC Midpoint', alpha=0.8)
ax1.plot(eth_sample.index, eth_sample['midpoint'], color=colors['ETH'], 
         linewidth=2, label='ETH Midpoint', alpha=0.8)
ax1.plot(ada_sample.index, ada_sample['midpoint'], color=colors['ADA'], 
         linewidth=2, label='ADA Midpoint', alpha=0.8)

# Add bid-ask spreads as filled areas (normalized for visibility)
btc_spread_norm = btc_sample['spread'] / btc_sample['midpoint'] * 1000
eth_spread_norm = eth_sample['spread'] / eth_sample['midpoint'] * 1000
ada_spread_norm = ada_sample['spread'] / ada_sample['midpoint'] * 1000

ax1_twin = ax1.twinx()
ax1_twin.fill_between(btc_sample.index, 0, btc_spread_norm, 
                      color=colors['BTC'], alpha=0.2, label='BTC Spread')
ax1_twin.fill_between(eth_sample.index, 0, eth_spread_norm, 
                      color=colors['ETH'], alpha=0.2, label='ETH Spread')
ax1_twin.fill_between(ada_sample.index, 0, ada_spread_norm, 
                      color=colors['ADA'], alpha=0.2, label='ADA Spread')

ax1.set_title('Midpoint Prices with Bid-Ask Spreads', fontweight='bold', fontsize=14)
ax1.set_ylabel('Price (USD)', fontweight='bold')
ax1_twin.set_ylabel('Spread (bps)', fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked area chart of cumulative trading volume
total_volume_btc = btc_sample['buys'] + btc_sample['sells']
total_volume_eth = eth_sample['buys'] + eth_sample['sells']
total_volume_ada = ada_sample['buys'] + ada_sample['sells']

# Normalize volumes for stacking (convert to millions)
vol_btc_norm = total_volume_btc / 1e6
vol_eth_norm = total_volume_eth / 1e6
vol_ada_norm = total_volume_ada / 1e6

# Use the same time index for all volumes
time_index = btc_sample.index

ax2.fill_between(time_index, 0, vol_btc_norm, 
                 color=colors['BTC'], alpha=0.7, label='BTC Volume')
ax2.fill_between(time_index, vol_btc_norm, vol_btc_norm + vol_eth_norm, 
                 color=colors['ETH'], alpha=0.7, label='ETH Volume')
ax2.fill_between(time_index, vol_btc_norm + vol_eth_norm, 
                 vol_btc_norm + vol_eth_norm + vol_ada_norm, 
                 color=colors['ADA'], alpha=0.7, label='ADA Volume')

ax2.set_title('Cumulative Trading Volume', fontweight='bold', fontsize=14)
ax2.set_ylabel('Volume (Millions USD)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Distance of first bid/ask levels from midpoint with volatility bands
# Calculate rolling volatility for error bands
window = 20
btc_vol = btc_sample['midpoint'].rolling(window).std()
eth_vol = eth_sample['midpoint'].rolling(window).std()
ada_vol = ada_sample['midpoint'].rolling(window).std()

# Plot bid distances (negative values)
ax3.plot(btc_sample.index, btc_sample['bids_distance_0'] * 1000, 
         color=colors['BTC'], linewidth=2, label='BTC Bid Distance')
ax3.plot(eth_sample.index, eth_sample['bids_distance_0'] * 1000, 
         color=colors['ETH'], linewidth=2, label='ETH Bid Distance')
ax3.plot(ada_sample.index, ada_sample['bids_distance_0'] * 1000, 
         color=colors['ADA'], linewidth=2, label='ADA Bid Distance')

# Plot ask distances (positive values)
ax3.plot(btc_sample.index, btc_sample['asks_distance_0'] * 1000, 
         color=colors['BTC'], linewidth=2, linestyle='--', label='BTC Ask Distance')
ax3.plot(eth_sample.index, eth_sample['asks_distance_0'] * 1000, 
         color=colors['ETH'], linewidth=2, linestyle='--', label='ETH Ask Distance')
ax3.plot(ada_sample.index, ada_sample['asks_distance_0'] * 1000, 
         color=colors['ADA'], linewidth=2, linestyle='--', label='ADA Ask Distance')

# Add volatility bands for BTC only to avoid clutter
vol_factor = 0.1
btc_bid_dist = btc_sample['bids_distance_0'] * 1000
btc_vol_scaled = btc_vol * vol_factor
btc_vol_scaled = btc_vol_scaled.fillna(0)  # Fill NaN values

ax3.fill_between(btc_sample.index, 
                 btc_bid_dist - btc_vol_scaled,
                 btc_bid_dist + btc_vol_scaled,
                 color=colors['BTC'], alpha=0.1, label='BTC Volatility Band')

ax3.set_title('Bid-Ask Distance from Midpoint with Volatility Bands', fontweight='bold', fontsize=14)
ax3.set_ylabel('Distance (bps)', fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Subplot 4: Rolling correlation heatmap
# Calculate rolling correlations between price movements
window_corr = 60  # 1-hour window
btc_returns = btc_sample['midpoint'].pct_change()
eth_returns = eth_sample['midpoint'].pct_change()
ada_returns = ada_sample['midpoint'].pct_change()

# Calculate rolling correlations
corr_btc_eth = btc_returns.rolling(window_corr).corr(eth_returns)
corr_btc_ada = btc_returns.rolling(window_corr).corr(ada_returns)
corr_eth_ada = eth_returns.rolling(window_corr).corr(ada_returns)

# Drop NaN values and ensure we have valid correlations
corr_btc_eth = corr_btc_eth.dropna()
corr_btc_ada = corr_btc_ada.dropna()
corr_eth_ada = corr_eth_ada.dropna()

# Find common time points for all correlations
common_corr_times = corr_btc_eth.index.intersection(corr_btc_ada.index).intersection(corr_eth_ada.index)

if len(common_corr_times) > 10:  # Ensure we have enough data points
    # Sample every 5th point for clarity
    time_points = common_corr_times[::5]
    
    # Extract correlation values at these time points
    btc_eth_vals = [corr_btc_eth.loc[t] for t in time_points]
    btc_ada_vals = [corr_btc_ada.loc[t] for t in time_points]
    eth_ada_vals = [corr_eth_ada.loc[t] for t in time_points]
    
    ax4.plot(time_points, btc_eth_vals, 
             color='red', linewidth=3, label='BTC-ETH', alpha=0.8)
    ax4.plot(time_points, btc_ada_vals, 
             color='blue', linewidth=3, label='BTC-ADA', alpha=0.8)
    ax4.plot(time_points, eth_ada_vals, 
             color='green', linewidth=3, label='ETH-ADA', alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax4.set_ylim(-1, 1)
else:
    # Fallback: show static correlation matrix as heatmap
    final_corr_matrix = np.array([
        [1.0, corr_btc_eth.iloc[-1] if len(corr_btc_eth) > 0 else 0.5, 
         corr_btc_ada.iloc[-1] if len(corr_btc_ada) > 0 else 0.3],
        [corr_btc_eth.iloc[-1] if len(corr_btc_eth) > 0 else 0.5, 1.0, 
         corr_eth_ada.iloc[-1] if len(corr_eth_ada) > 0 else 0.7],
        [corr_btc_ada.iloc[-1] if len(corr_btc_ada) > 0 else 0.3, 
         corr_eth_ada.iloc[-1] if len(corr_eth_ada) > 0 else 0.7, 1.0]
    ])
    
    im = ax4.imshow(final_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xticklabels(['BTC', 'ETH', 'ADA'])
    ax4.set_yticklabels(['BTC', 'ETH', 'ADA'])
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            ax4.text(j, i, f'{final_corr_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')

ax4.set_title('Rolling Price Correlation Evolution', fontweight='bold', fontsize=14)
ax4.set_ylabel('Correlation Coefficient', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Format x-axis for all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Time', fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Save the plot
plt.savefig('crypto_microstructure_analysis.png', dpi=300, bbox_inches='tight')
plt.show()