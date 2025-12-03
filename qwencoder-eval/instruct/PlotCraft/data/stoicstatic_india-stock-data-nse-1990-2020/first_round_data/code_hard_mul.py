import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load stock data efficiently
def load_stock_data():
    stocks = {}
    stock_files = ['RELIANCE.csv', 'HINDUNILVR.csv', 'SBIN.csv', 'INFY.csv', 'HDFCBANK.csv', 'TCS.csv', 'ITC.csv', 'ICICIBANK.csv']
    stock_names = ['RELIANCE', 'HINDUNILVR', 'SBIN', 'INFY', 'HDFCBANK', 'TCS', 'ITC', 'ICICIBANK']
    
    for file, name in zip(stock_files, stock_names):
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            df = df.dropna(subset=['Close', 'Volume'])
            # Sample data to reduce processing time
            if len(df) > 1000:
                df = df.iloc[::max(1, len(df)//1000)].reset_index(drop=True)
            stocks[name] = df
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue
    return stocks

stocks = load_stock_data()

# Create simplified NIFTY data
def create_nifty_data():
    if not stocks:
        return pd.DataFrame()
    
    # Use the stock with most data as base
    base_stock = max(stocks.keys(), key=lambda x: len(stocks[x]))
    base_dates = stocks[base_stock]['Date'].copy()
    
    # Create synthetic NIFTY using weighted average
    weights = {'RELIANCE': 0.2, 'HDFCBANK': 0.15, 'INFY': 0.12, 'TCS': 0.1, 
               'ICICIBANK': 0.1, 'SBIN': 0.08, 'ITC': 0.08, 'HINDUNILVR': 0.07}
    
    nifty_close = []
    nifty_volume = []
    
    for date in base_dates:
        weighted_price = 0
        total_weight = 0
        total_volume = 0
        
        for name, weight in weights.items():
            if name in stocks:
                stock_data = stocks[name]
                closest_data = stock_data[stock_data['Date'] <= date]
                if not closest_data.empty:
                    latest_data = closest_data.iloc[-1]
                    weighted_price += latest_data['Close'] * weight
                    total_weight += weight
                    total_volume += latest_data['Volume'] * weight
        
        if total_weight > 0:
            nifty_close.append(weighted_price / total_weight * 50)  # Scale to index level
            nifty_volume.append(total_volume)
        else:
            nifty_close.append(np.nan)
            nifty_volume.append(np.nan)
    
    nifty_df = pd.DataFrame({
        'Date': base_dates,
        'Close': nifty_close,
        'Volume': nifty_volume
    }).dropna()
    
    return nifty_df

nifty_df = create_nifty_data()

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('white')

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Subplot 1,1: NIFTY 50 price evolution with volume bars
ax1 = plt.subplot(3, 2, 1)
ax1_twin = ax1.twinx()

if not nifty_df.empty:
    # Normalize NIFTY prices
    nifty_normalized = (nifty_df['Close'] / nifty_df['Close'].iloc[0]) * 100
    
    # Monthly volume aggregation
    nifty_df['YearMonth'] = nifty_df['Date'].dt.to_period('M')
    monthly_data = nifty_df.groupby('YearMonth').agg({
        'Volume': 'mean',
        'Date': 'first'
    }).reset_index()
    
    # Plot normalized price line
    ax1.plot(nifty_df['Date'], nifty_normalized, color=colors[0], linewidth=2, label='NIFTY 50 (Normalized)')
    ax1.set_ylabel('Normalized Price Index', fontweight='bold', fontsize=10)
    
    # Plot volume bars (sample every 10th point to reduce density)
    sample_monthly = monthly_data.iloc[::10]
    ax1_twin.bar(sample_monthly['Date'], sample_monthly['Volume'], 
                alpha=0.3, color=colors[1], width=30, label='Monthly Volume')
    ax1_twin.set_ylabel('Average Monthly Volume', fontweight='bold', fontsize=10)

ax1.set_title('NIFTY 50 Price Evolution with Trading Volume', fontweight='bold', fontsize=12, pad=15)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=8)
ax1_twin.legend(loc='upper right', fontsize=8)

# Subplot 1,2: Top 5 stocks cumulative returns
ax2 = plt.subplot(3, 2, 2)

top_stocks = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK']
for i, stock in enumerate(top_stocks):
    if stock in stocks and len(stocks[stock]) > 10:
        stock_data = stocks[stock].copy()
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Cumulative_Returns'] = (1 + stock_data['Returns'].fillna(0)).cumprod() * 100
        ax2.plot(stock_data['Date'], stock_data['Cumulative_Returns'], 
                color=colors[i], linewidth=2, label=stock, alpha=0.8)

ax2.set_ylabel('Cumulative Returns (%)', fontweight='bold', fontsize=10)
ax2.set_xlabel('Year', fontweight='bold', fontsize=10)
ax2.set_title('Top 5 Stocks Cumulative Returns', fontweight='bold', fontsize=12, pad=15)
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 2,1: NIFTY trend and seasonal analysis
ax3 = plt.subplot(3, 2, 3)

if not nifty_df.empty and len(nifty_df) > 50:
    # Original prices
    ax3.plot(nifty_df['Date'], nifty_df['Close'], color=colors[0], linewidth=1.5, alpha=0.7, label='Original')
    
    # Trend component (rolling mean)
    window_size = min(60, len(nifty_df)//4)
    trend = nifty_df['Close'].rolling(window=window_size, center=True).mean()
    ax3.plot(nifty_df['Date'], trend, color=colors[2], linewidth=3, label='Trend')
    
    # Seasonal pattern
    nifty_df['Month'] = nifty_df['Date'].dt.month
    monthly_avg = nifty_df.groupby('Month')['Close'].mean()
    seasonal_effect = nifty_df['Month'].map(monthly_avg)
    seasonal_normalized = (seasonal_effect - seasonal_effect.mean()) * 0.05 + nifty_df['Close'].mean()
    
    ax3.fill_between(nifty_df['Date'], nifty_df['Close'].mean() - nifty_df['Close'].std()*0.1, 
                    seasonal_normalized, alpha=0.3, color=colors[3], label='Seasonal Pattern')

ax3.set_ylabel('NIFTY 50 Index', fontweight='bold', fontsize=10)
ax3.set_xlabel('Year', fontweight='bold', fontsize=10)
ax3.set_title('NIFTY 50 Trend and Seasonal Analysis', fontweight='bold', fontsize=12, pad=15)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 2,2: Sector correlation heatmap
ax4 = plt.subplot(3, 2, 4)

# Create simplified sector correlation matrix
sectors = ['Banking', 'IT', 'FMCG', 'Energy', 'Diversified']
sector_mapping = {
    'Banking': ['HDFCBANK', 'SBIN', 'ICICIBANK'],
    'IT': ['INFY', 'TCS'],
    'FMCG': ['HINDUNILVR', 'ITC'],
    'Energy': ['RELIANCE'],
    'Diversified': ['RELIANCE', 'ITC']
}

# Calculate sector returns
sector_returns = {}
for sector, stock_list in sector_mapping.items():
    returns_list = []
    for stock in stock_list:
        if stock in stocks and len(stocks[stock]) > 10:
            returns = stocks[stock]['Close'].pct_change().dropna()
            if len(returns) > 0:
                returns_list.append(returns.iloc[:min(500, len(returns))])  # Limit data size
    
    if returns_list:
        # Align series by taking common length
        min_len = min(len(r) for r in returns_list)
        aligned_returns = [r.iloc[:min_len].reset_index(drop=True) for r in returns_list]
        sector_returns[sector] = pd.concat(aligned_returns, axis=1).mean(axis=1)

# Create correlation matrix
if len(sector_returns) > 1:
    corr_df = pd.DataFrame(sector_returns)
    corr_matrix = corr_df.corr()
    
    # Plot heatmap
    im = ax4.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.index)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax4.set_yticklabels(corr_matrix.index, fontsize=9)
    
    # Add correlation values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold', fontsize=9)
    
    plt.colorbar(im, ax=ax4, shrink=0.6)

ax4.set_title('Sector Correlation Matrix', fontweight='bold', fontsize=12, pad=15)

# Subplot 3,1: Volatility analysis
ax5 = plt.subplot(3, 2, 5)

if not nifty_df.empty and len(nifty_df) > 30:
    # Calculate returns and volatility
    daily_returns = nifty_df['Close'].pct_change().dropna() * 100
    rolling_vol = daily_returns.rolling(window=min(30, len(daily_returns)//3)).std()
    
    # Plot volatility time series
    ax5.plot(nifty_df['Date'][1:len(rolling_vol)+1], rolling_vol, 
            color=colors[4], linewidth=2, label='Rolling Volatility')
    
    # Add volatility bands
    if len(rolling_vol.dropna()) > 0:
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        upper_band = vol_mean + 1.5 * vol_std
        lower_band = max(0, vol_mean - 1.5 * vol_std)
        
        ax5.axhline(y=upper_band, color=colors[5], linestyle='--', alpha=0.7, label='Upper Band')
        ax5.axhline(y=lower_band, color=colors[5], linestyle='--', alpha=0.7, label='Lower Band')
        ax5.fill_between(nifty_df['Date'][1:len(rolling_vol)+1], lower_band, upper_band, 
                        alpha=0.2, color=colors[5])

ax5.set_ylabel('Volatility (%)', fontweight='bold', fontsize=10)
ax5.set_xlabel('Year', fontweight='bold', fontsize=10)
ax5.set_title('NIFTY 50 Volatility Analysis', fontweight='bold', fontsize=12, pad=15)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 3,2: Market regime identification
ax6 = plt.subplot(3, 2, 6)

if not nifty_df.empty and len(nifty_df) > 100:
    # Calculate moving averages
    short_window = min(20, len(nifty_df)//10)
    long_window = min(50, len(nifty_df)//5)
    
    short_ma = nifty_df['Close'].rolling(window=short_window).mean()
    long_ma = nifty_df['Close'].rolling(window=long_window).mean()
    
    # Identify bull/bear markets
    bull_market = short_ma > long_ma
    
    # Plot price and moving averages
    ax6.plot(nifty_df['Date'], nifty_df['Close'], color='black', linewidth=2, label='NIFTY 50')
    ax6.plot(nifty_df['Date'], short_ma, color=colors[0], linewidth=1, alpha=0.7, label=f'{short_window}-day MA')
    ax6.plot(nifty_df['Date'], long_ma, color=colors[1], linewidth=1, alpha=0.7, label=f'{long_window}-day MA')
    
    # Add regime background (sample to avoid too many patches)
    sample_indices = range(0, len(nifty_df)-1, max(1, len(nifty_df)//100))
    for i in sample_indices:
        if i < len(bull_market) - 1:
            color = 'green' if bull_market.iloc[i] else 'red'
            ax6.axvspan(nifty_df['Date'].iloc[i], nifty_df['Date'].iloc[min(i+10, len(nifty_df)-1)], 
                       alpha=0.1, color=color)

ax6.set_ylabel('NIFTY 50 Index', fontweight='bold', fontsize=10)
ax6.set_xlabel('Year', fontweight='bold', fontsize=10)
ax6.set_title('Market Regime Identification', fontweight='bold', fontsize=12, pad=15)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Add main title
fig.suptitle('NSE Market Analysis: Temporal Evolution & Performance Patterns (1996-2021)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('nse_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()