import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Get list of available CSV files in the directory
available_files = []
for file in os.listdir('.'):
    if file.endswith('.csv') and 'Global' in file or 'Globla' in file:
        available_files.append(file)

print(f"Found {len(available_files)} data files")

# Load all available data files
all_data = []
for file in available_files:
    try:
        df = pd.read_csv(file)
        print(f"Loaded {file}: {len(df)} rows")
        all_data.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

# Check if we have any data
if not all_data:
    print("No data files could be loaded. Creating sample data for demonstration.")
    # Create sample data for demonstration
    dates = pd.date_range('2008-01-01', '2023-12-31', freq='D')
    tickers = ['^NYA', '^IXIC', '^DJI', '^GSPC', '^NSEI', '^BSESN', '^N225', '000001.SS', '^FTSE', '^N100', 'GC=F', 'CL=F']
    
    sample_data = []
    for ticker in tickers:
        base_price = np.random.uniform(1000, 5000)
        prices = base_price * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
        
        for i, date in enumerate(dates):
            if np.random.random() > 0.1:  # 90% data availability
                sample_data.append({
                    'Ticker': ticker,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Open': prices[i] * 0.99,
                    'High': prices[i] * 1.02,
                    'Low': prices[i] * 0.98,
                    'Close': prices[i],
                    'Adj Close': prices[i],
                    'Volume': np.random.uniform(1e6, 1e9)
                })
    
    df = pd.DataFrame(sample_data)
else:
    # Combine all loaded data
    df = pd.concat(all_data, ignore_index=True)

# Process the data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Define market groups
us_indices = ['^NYA', '^IXIC', '^DJI', '^GSPC']
asian_indices = ['^NSEI', '^BSESN', '^N225', '000001.SS']
european_indices = ['^FTSE', '^N100']
commodities = ['GC=F', 'CL=F']

# Create figure with 3x2 subplots
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Helper function to normalize prices
def normalize_prices(group):
    group = group.sort_values('Date').copy()
    if len(group) > 0 and group['Close'].iloc[0] != 0:
        group['Normalized'] = group['Close'] / group['Close'].iloc[0] * 100
    else:
        group['Normalized'] = 100
    return group

# Helper function to calculate volatility
def calculate_volatility(prices, window=30):
    if len(prices) < window:
        return pd.Series([np.nan] * len(prices))
    returns = prices.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return volatility

# Subplot 1: US indices with volatility (2008-2015)
ax1 = plt.subplot(3, 2, 1)
ax1_vol = ax1.twinx()

period1_data = df[(df['Date'] >= '2008-01-01') & (df['Date'] <= '2015-12-31')]
us_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

combined_volatility = []
volatility_dates = None

for i, ticker in enumerate(us_indices):
    ticker_data = period1_data[period1_data['Ticker'] == ticker].copy()
    if len(ticker_data) > 10:
        ticker_data = normalize_prices(ticker_data)
        ax1.plot(ticker_data['Date'], ticker_data['Normalized'], 
                color=us_colors[i], linewidth=2, label=ticker, alpha=0.8)
        
        vol = calculate_volatility(ticker_data['Close'])
        if len(vol.dropna()) > 0:
            combined_volatility.append(vol.fillna(0).values)
            if volatility_dates is None:
                volatility_dates = ticker_data['Date']

if combined_volatility and volatility_dates is not None:
    # Calculate average volatility
    min_len = min(len(v) for v in combined_volatility)
    if min_len > 0:
        avg_vol = np.mean([v[:min_len] for v in combined_volatility], axis=0)
        vol_dates = volatility_dates.iloc[:min_len]
        ax1_vol.fill_between(vol_dates, 0, avg_vol, alpha=0.3, color='gray', label='Combined Volatility')

ax1.set_title('US Markets: Crisis & Recovery (2008-2015)\nNormalized Prices with Volatility Overlay', 
              fontweight='bold', fontsize=12, pad=15)
ax1.set_ylabel('Normalized Price (Base=100)', fontweight='bold')
ax1_vol.set_ylabel('Volatility (%)', fontweight='bold', color='gray')
ax1.legend(loc='upper left', fontsize=8)
ax1_vol.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# Add crisis annotation
ax1.axvline(pd.to_datetime('2008-09-15'), color='red', linestyle='--', alpha=0.7, linewidth=1)
ax1.text(pd.to_datetime('2008-09-15'), ax1.get_ylim()[1]*0.9, 'Lehman Crisis', 
         rotation=90, fontsize=8, ha='right')

# Subplot 2: Asian markets with volume spikes (2008-2015)
ax2 = plt.subplot(3, 2, 2)
asian_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for i, ticker in enumerate(asian_indices):
    ticker_data = period1_data[period1_data['Ticker'] == ticker].copy()
    if len(ticker_data) > 10:
        ticker_data = normalize_prices(ticker_data)
        ax2.plot(ticker_data['Date'], ticker_data['Normalized'], 
                color=asian_colors[i], linewidth=2, label=ticker, alpha=0.8)
        
        # Volume spikes (95th percentile)
        if 'Volume' in ticker_data.columns and ticker_data['Volume'].notna().sum() > 0:
            vol_95 = ticker_data['Volume'].quantile(0.95)
            spikes = ticker_data[ticker_data['Volume'] > vol_95]
            if len(spikes) > 0:
                spike_sizes = np.clip(spikes['Volume']/1e8, 10, 100)
                ax2.scatter(spikes['Date'], spikes['Normalized'], 
                           s=spike_sizes, alpha=0.6, color=asian_colors[i])

ax2.set_title('Asian Markets: Crisis & Recovery (2008-2015)\nNormalized Prices with Volume Spikes', 
              fontweight='bold', fontsize=12, pad=15)
ax2.set_ylabel('Normalized Price (Base=100)', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: European & Commodities with price bands (2016-2019)
ax3 = plt.subplot(3, 2, 3)
period2_data = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
eur_colors = ['#17becf', '#bcbd22']
comm_colors = ['#ffbb78', '#c5b0d5']

# European indices
for i, ticker in enumerate(european_indices):
    ticker_data = period2_data[period2_data['Ticker'] == ticker].copy()
    if len(ticker_data) > 10:
        ticker_data = normalize_prices(ticker_data)
        ax3.plot(ticker_data['Date'], ticker_data['Normalized'], 
                color=eur_colors[i], linewidth=2, label=ticker, alpha=0.8)
        
        # Price bands (High-Low range)
        if ticker_data['Close'].iloc[0] != 0:
            high_norm = ticker_data['High'] / ticker_data['Close'].iloc[0] * 100
            low_norm = ticker_data['Low'] / ticker_data['Close'].iloc[0] * 100
            ax3.fill_between(ticker_data['Date'], low_norm, high_norm, 
                            alpha=0.2, color=eur_colors[i])

# Commodities
for i, ticker in enumerate(commodities):
    ticker_data = period2_data[period2_data['Ticker'] == ticker].copy()
    if len(ticker_data) > 10:
        ticker_data = normalize_prices(ticker_data)
        ax3.plot(ticker_data['Date'], ticker_data['Normalized'], 
                color=comm_colors[i], linewidth=2, label=ticker, alpha=0.8, linestyle='--')

ax3.set_title('European Markets & Commodities: Growth Period (2016-2019)\nNormalized Prices with Trading Bands', 
              fontweight='bold', fontsize=12, pad=15)
ax3.set_ylabel('Normalized Price (Base=100)', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Correlation evolution with volume (2016-2019)
ax4 = plt.subplot(3, 2, 4)
ax4_vol = ax4.twinx()

# Calculate rolling correlations between major indices
major_indices = ['^GSPC', '^FTSE', '^NSEI']
correlations = {}

for i in range(len(major_indices)):
    for j in range(i+1, len(major_indices)):
        idx1, idx2 = major_indices[i], major_indices[j]
        data1 = period2_data[period2_data['Ticker'] == idx1]
        data2 = period2_data[period2_data['Ticker'] == idx2]
        
        if len(data1) > 90 and len(data2) > 90:
            # Merge data on date
            merged = pd.merge(data1[['Date', 'Close']], data2[['Date', 'Close']], 
                            on='Date', suffixes=('_1', '_2'))
            if len(merged) > 90:
                merged = merged.set_index('Date').sort_index()
                rolling_corr = merged['Close_1'].rolling(90).corr(merged['Close_2'])
                correlations[f'{idx1}-{idx2}'] = rolling_corr

corr_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (pair, corr) in enumerate(correlations.items()):
    if len(corr.dropna()) > 0:
        ax4.plot(corr.index, corr.values, color=corr_colors[i % len(corr_colors)], 
                linewidth=2, label=f'Corr {pair.split("-")[0][-4:]}-{pair.split("-")[1][-4:]}', alpha=0.8)

# Average volume representation
volume_data = []
volume_dates = []
for ticker in major_indices:
    ticker_data = period2_data[period2_data['Ticker'] == ticker]
    if len(ticker_data) > 0 and 'Volume' in ticker_data.columns:
        monthly_data = ticker_data.groupby(ticker_data['Date'].dt.to_period('M')).agg({
            'Volume': 'mean',
            'Date': 'first'
        })
        volume_data.extend(monthly_data['Volume'].values)
        volume_dates.extend(monthly_data['Date'].values)

if volume_data:
    # Sample every 3rd point to avoid overcrowding
    sample_indices = range(0, len(volume_data), 3)
    sampled_volumes = [volume_data[i] for i in sample_indices]
    sampled_dates = [volume_dates[i] for i in sample_indices]
    
    ax4_vol.bar(sampled_dates, np.array(sampled_volumes)/1e9, alpha=0.3, color='gray', 
               width=20, label='Avg Volume (B)')

ax4.set_title('Global Market Correlations: Growth Period (2016-2019)\nRolling Correlations with Trading Volume', 
              fontweight='bold', fontsize=12, pad=15)
ax4.set_ylabel('Correlation Coefficient', fontweight='bold')
ax4_vol.set_ylabel('Volume (Billions)', fontweight='bold', color='gray')
ax4.legend(loc='upper left', fontsize=8)
ax4_vol.legend(loc='upper right', fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Regional market cap changes with drawdowns (2020-2023)
ax5 = plt.subplot(3, 2, 5)
period3_data = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]

# Regional representation using major indices
regions = {
    'US': '^GSPC',
    'Europe': '^FTSE', 
    'Asia': '^NSEI'
}

regional_data = {}
for region, ticker in regions.items():
    ticker_data = period3_data[period3_data['Ticker'] == ticker].copy()
    if len(ticker_data) > 10:
        ticker_data = normalize_prices(ticker_data)
        regional_data[region] = ticker_data

# Create stacked area chart
if len(regional_data) >= 2:
    # Find common date range
    all_dates = []
    for data in regional_data.values():
        all_dates.extend(data['Date'].tolist())
    
    common_dates = sorted(list(set(all_dates)))
    
    if len(common_dates) > 50:
        # Sample dates to avoid overcrowding
        sample_step = max(1, len(common_dates) // 100)
        sampled_dates = common_dates[::sample_step]
        
        values = []
        labels = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (region, data) in enumerate(regional_data.items()):
            # Interpolate values for sampled dates
            region_values = []
            for date in sampled_dates:
                closest_data = data[data['Date'] <= date]
                if len(closest_data) > 0:
                    region_values.append(closest_data['Normalized'].iloc[-1])
                else:
                    region_values.append(100)
            
            values.append(region_values)
            labels.append(region)
        
        if values:
            # Normalize to percentages
            values_array = np.array(values)
            total = np.sum(values_array, axis=0)
            total[total == 0] = 1  # Avoid division by zero
            percentages = (values_array / total) * 100
            
            ax5.stackplot(sampled_dates, *percentages, labels=labels, colors=colors[:len(labels)], alpha=0.7)
            
            # Add drawdown indication
            for region, data in regional_data.items():
                if len(data) > 20:
                    rolling_max = data['Normalized'].expanding().max()
                    drawdown = (data['Normalized'] - rolling_max) / rolling_max * 100
                    major_drawdowns = drawdown < -10
                    
                    if major_drawdowns.any():
                        drawdown_periods = data[major_drawdowns]
                        if len(drawdown_periods) > 0:
                            ax5.scatter(drawdown_periods['Date'], [50] * len(drawdown_periods), 
                                      color='red', alpha=0.6, s=10, marker='v')

ax5.set_title('Regional Market Dynamics: Pandemic Era (2020-2023)\nRelative Market Share with Drawdown Periods', 
              fontweight='bold', fontsize=12, pad=15)
ax5.set_ylabel('Market Share (%)', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Add COVID annotation
ax5.axvline(pd.to_datetime('2020-03-15'), color='red', linestyle='--', alpha=0.7, linewidth=1)
ax5.text(pd.to_datetime('2020-03-15'), 80, 'COVID-19', rotation=90, fontsize=8, ha='right')

# Subplot 6: Correlation evolution (2020-2023)
ax6 = plt.subplot(3, 2, 6)

# Calculate market synchronization index
all_indices = [idx for idx in us_indices + asian_indices + european_indices if idx in df['Ticker'].unique()]
monthly_correlations = []
correlation_dates = []

# Group by month and calculate average correlations
monthly_groups = period3_data.groupby(period3_data['Date'].dt.to_period('M'))

for month, month_data in monthly_groups:
    if len(month_data) > 50:  # Ensure sufficient data
        # Get available indices for this month
        available_indices = month_data['Ticker'].unique()
        common_indices = [idx for idx in all_indices if idx in available_indices]
        
        if len(common_indices) >= 3:
            # Calculate pairwise correlations
            correlations_month = []
            for i in range(len(common_indices)):
                for j in range(i+1, len(common_indices)):
                    idx1_data = month_data[month_data['Ticker'] == common_indices[i]]['Close']
                    idx2_data = month_data[month_data['Ticker'] == common_indices[j]]['Close']
                    
                    if len(idx1_data) > 5 and len(idx2_data) > 5:
                        # Simple correlation approximation
                        corr = np.corrcoef(idx1_data.values[:min(len(idx1_data), len(idx2_data))], 
                                         idx2_data.values[:min(len(idx1_data), len(idx2_data))])[0, 1]
                        if not np.isnan(corr):
                            correlations_month.append(abs(corr))
            
            if correlations_month:
                avg_corr = np.mean(correlations_month)
                monthly_correlations.append(avg_corr)
                correlation_dates.append(pd.to_datetime(str(month)))

# Plot correlation evolution
if monthly_correlations:
    ax6.plot(correlation_dates, monthly_correlations, color='#d62728', linewidth=3, 
            label='Market Synchronization Index', alpha=0.8)
    ax6.fill_between(correlation_dates, monthly_correlations, alpha=0.3, color='#d62728')

# Add trend line
if len(monthly_correlations) > 3:
    z = np.polyfit(range(len(monthly_correlations)), monthly_correlations, 1)
    p = np.poly1d(z)
    ax6.plot(correlation_dates, p(range(len(monthly_correlations))), 
            color='black', linestyle=':', linewidth=2, alpha=0.7, label='Trend')

# Add annotations for major events
ax6.axvline(pd.to_datetime('2020-03-15'), color='red', linestyle='--', alpha=0.7, linewidth=1)
ax6.axvline(pd.to_datetime('2021-01-01'), color='green', linestyle='--', alpha=0.7, linewidth=1)

ax6.set_title('Market Synchronization: Pandemic Era (2020-2023)\nEvolution of Cross-Market Correlations', 
              fontweight='bold', fontsize=12, pad=15)
ax6.set_ylabel('Average Correlation', fontweight='bold')
ax6.set_xlabel('Date', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Overall layout adjustments
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Add main title
fig.suptitle('Global Stock Market Evolution & Volatility Analysis (2008-2023)\nComprehensive Multi-Period Market Dynamics', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('global_markets_analysis.png', dpi=300, bbox_inches='tight')
plt.show()