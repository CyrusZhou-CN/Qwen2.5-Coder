import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load datasets with optimized sampling
datasets = {}
files = ['BTC_1sec.csv', 'BTC_1min.csv', 'BTC_5min.csv', 
         'ETH_1sec.csv', 'ETH_1min.csv', 'ETH_5min.csv',
         'ADA_1sec.csv', 'ADA_1min.csv', 'ADA_5min.csv']

print("Loading datasets...")
for file in files:
    try:
        # Load only first 5000 rows for performance
        df = pd.read_csv(file, nrows=5000)
        df['system_time'] = pd.to_datetime(df['system_time'])
        datasets[file.replace('.csv', '')] = df
        print(f"Loaded {file}: {len(df)} rows")
    except Exception as e:
        print(f"Could not load {file}: {e}")
        continue

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color schemes
colors = {'BTC': '#F7931A', 'ETH': '#627EEA', 'ADA': '#0033AD'}
time_colors = {'1sec': '#FFB6C1', '1min': '#4169E1', '5min': '#191970'}

print("Creating visualizations...")

# Top row: Individual cryptocurrency analysis
for i, crypto in enumerate(['BTC', 'ETH', 'ADA']):
    ax = plt.subplot(2, 3, i+1)
    
    # Process each timeframe
    for freq in ['1sec', '1min', '5min']:
        key = f"{crypto}_{freq}"
        if key in datasets:
            df = datasets[key]
            # Sample every 50th point for performance
            sample_df = df.iloc[::50].copy()
            
            if len(sample_df) > 10:
                # Calculate volatility bands
                sample_df['volatility'] = sample_df['spread'].rolling(window=5, min_periods=1).std()
                sample_df['upper_band'] = sample_df['midpoint'] + sample_df['volatility']
                sample_df['lower_band'] = sample_df['midpoint'] - sample_df['volatility']
                
                # Plot price evolution
                x_vals = range(len(sample_df))
                ax.plot(x_vals, sample_df['midpoint'], 
                       color=time_colors[freq], alpha=0.8, linewidth=1.5, 
                       label=f'{freq}')
                
                # Add volatility bands
                ax.fill_between(x_vals, sample_df['upper_band'], sample_df['lower_band'], 
                               color=time_colors[freq], alpha=0.2)
    
    # Secondary y-axis for volume imbalance
    ax2 = ax.twinx()
    key = f"{crypto}_1min"
    if key in datasets:
        df = datasets[key]
        sample_df = df.iloc[::50].copy()
        if len(sample_df) > 10:
            # Calculate volume imbalance
            sample_df['volume_imbalance'] = sample_df['buys'] - sample_df['sells']
            sample_df['cum_imbalance'] = sample_df['volume_imbalance'].cumsum()
            
            ax2.plot(range(len(sample_df)), sample_df['cum_imbalance'], 
                    color='red', alpha=0.7, linewidth=2, linestyle='--')
            ax2.set_ylabel('Cumulative Volume Imbalance', color='red', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title(f'{crypto} Price Evolution & Volume Dynamics', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time Index', fontweight='bold')
    ax.set_ylabel('Midpoint Price', fontweight='bold', color=colors[crypto])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

# Bottom left: Cross-cryptocurrency correlation analysis
ax = plt.subplot(2, 3, 4)

# Calculate correlations
correlation_data = {}
for crypto in ['BTC', 'ETH', 'ADA']:
    key = f"{crypto}_1min"
    if key in datasets:
        df = datasets[key]
        # Sample every 20th point
        sample_df = df.iloc[::20]
        correlation_data[crypto] = sample_df['midpoint'].values

# Align data lengths
if len(correlation_data) >= 2:
    min_len = min([len(v) for v in correlation_data.values()])
    for crypto in correlation_data:
        correlation_data[crypto] = correlation_data[crypto][:min_len]
    
    # Create correlation matrix
    corr_df = pd.DataFrame(correlation_data)
    correlation_matrix = corr_df.corr()
    
    # Plot heatmap
    im = ax.imshow(correlation_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, fontweight='bold')
    ax.set_yticklabels(correlation_matrix.index, fontweight='bold')
    
    # Add correlation values
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}', 
                   ha='center', va='center', fontweight='bold', color='white')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.6)

ax.set_title('Cross-Crypto Correlation Matrix', fontweight='bold', fontsize=12)

# Bottom center: Order book depth evolution
ax = plt.subplot(2, 3, 5)

depth_data = {}
for crypto in ['BTC', 'ETH', 'ADA']:
    key = f"{crypto}_1min"
    if key in datasets:
        df = datasets[key]
        sample_df = df.iloc[::100].copy()  # Heavy sampling for performance
        
        # Calculate total order book depth
        bid_cols = [col for col in df.columns if 'bids_notional_' in col and col.endswith(('_0', '_1', '_2'))]
        ask_cols = [col for col in df.columns if 'asks_notional_' in col and col.endswith(('_0', '_1', '_2'))]
        
        if bid_cols and ask_cols:
            total_depth = (sample_df[bid_cols].sum(axis=1) + sample_df[ask_cols].sum(axis=1))
            depth_data[crypto] = total_depth.values
            
            # Stacked area chart
            x_vals = range(len(sample_df))
            ax.fill_between(x_vals, 0, total_depth, 
                           color=colors[crypto], alpha=0.6, label=f'{crypto} Depth')

# Overlay spread dynamics
ax2 = ax.twinx()
for crypto in ['BTC', 'ETH', 'ADA']:
    key = f"{crypto}_1min"
    if key in datasets:
        df = datasets[key]
        sample_df = df.iloc[::100]
        if len(sample_df) > 5:
            ax2.plot(range(len(sample_df)), sample_df['spread'], 
                    color=colors[crypto], linewidth=2, alpha=0.8, linestyle='--')

ax.set_title('Order Book Depth Evolution', fontweight='bold', fontsize=12)
ax.set_xlabel('Time Index', fontweight='bold')
ax.set_ylabel('Total Order Book Depth', fontweight='bold')
ax2.set_ylabel('Bid-Ask Spread', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Bottom right: Spread distribution and volume relationship
ax = plt.subplot(2, 3, 6)

# Collect spread data for violin plots
spread_data = []
labels = []
for crypto in ['BTC', 'ETH', 'ADA']:
    for freq in ['1min', '5min']:  # Limit to avoid timeout
        key = f"{crypto}_{freq}"
        if key in datasets:
            df = datasets[key]
            sample_spread = df['spread'].dropna().iloc[::200]  # Heavy sampling
            if len(sample_spread) > 5:
                spread_data.append(sample_spread.values)
                labels.append(f'{crypto}_{freq}')

# Create violin plot
if spread_data and len(spread_data) > 0:
    positions = range(len(spread_data))
    try:
        parts = ax.violinplot(spread_data, positions=positions, widths=0.6, showmeans=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            if i < len(labels):
                if 'BTC' in labels[i]:
                    pc.set_facecolor(colors['BTC'])
                elif 'ETH' in labels[i]:
                    pc.set_facecolor(colors['ETH'])
                else:
                    pc.set_facecolor(colors['ADA'])
                pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    except Exception as e:
        print(f"Violin plot error: {e}")
        # Fallback to box plot
        ax.boxplot(spread_data, labels=labels)

# Overlay scatter plot
ax2 = ax.twinx()
for crypto in ['BTC', 'ETH', 'ADA']:
    key = f"{crypto}_1min"
    if key in datasets:
        df = datasets[key]
        sample_df = df.iloc[::200].copy()  # Heavy sampling
        
        if len(sample_df) > 10:
            # Calculate volume intensity
            sample_df['volume_intensity'] = sample_df['buys'] + sample_df['sells']
            
            # Scatter plot
            ax2.scatter(sample_df['spread'], sample_df['volume_intensity'], 
                       color=colors[crypto], alpha=0.6, s=30, label=f'{crypto}')

ax.set_title('Spread Distributions & Volume Relationships', fontweight='bold', fontsize=12)
ax.set_ylabel('Spread Distribution', fontweight='bold')
ax2.set_ylabel('Volume Intensity', fontweight='bold')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)

print("Visualization complete!")
plt.savefig('crypto_analysis.png', dpi=300, bbox_inches='tight')
plt.show()