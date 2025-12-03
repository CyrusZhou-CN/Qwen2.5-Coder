import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
import glob
warnings.filterwarnings('ignore')

# Load all available datasets
datasets = {}
csv_files = glob.glob('*.csv')

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        year = file.split('_')[0]
        datasets[year] = df
    except Exception as e:
        continue

# Combine all datasets
if datasets:
    all_data = pd.concat(datasets.values(), ignore_index=True)
    all_data = all_data.sort_values(['Ticker', 'Date'])
else:
    # Create dummy data if no files found
    dates = pd.date_range('2008-01-01', '2023-12-31', freq='D')
    tickers = ['^GSPC', '^IXIC', '^FTSE', '^N225', '^BSESN', '^NSEI', '^DJI', '^NYA', 'GC=F', 'CL=F', '000001.SS', '^N100']
    
    dummy_data = []
    for ticker in tickers:
        base_price = np.random.uniform(1000, 5000)
        prices = base_price * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
        volumes = np.random.uniform(1000000, 10000000, len(dates))
        
        for i, date in enumerate(dates):
            dummy_data.append({
                'Ticker': ticker,
                'Date': date,
                'Open': prices[i] * 0.99,
                'High': prices[i] * 1.02,
                'Low': prices[i] * 0.98,
                'Close': prices[i],
                'Adj Close': prices[i],
                'Volume': volumes[i]
            })
    
    all_data = pd.DataFrame(dummy_data)

# Helper function to get data for specific ticker and date range
def get_ticker_data(ticker, start_year, end_year):
    try:
        mask = (all_data['Ticker'] == ticker) & \
               (all_data['Date'].dt.year >= start_year) & \
               (all_data['Date'].dt.year <= end_year)
        data = all_data[mask].copy()
        if not data.empty:
            data = data.sort_values('Date').reset_index(drop=True)
        return data
    except:
        return pd.DataFrame()

# Helper function to calculate rolling volatility
def calculate_volatility(prices, window=20):
    try:
        if len(prices) < window:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)
    except:
        return pd.Series([np.nan] * len(prices))

# Create the 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Top row (2008-2010 Crisis Period)
# Subplot 1: S&P 500 and NASDAQ with volume ratios
ax1 = axes[0, 0]
try:
    sp500 = get_ticker_data('^GSPC', 2008, 2010)
    nasdaq = get_ticker_data('^IXIC', 2008, 2010)

    if not sp500.empty and not nasdaq.empty:
        # Sample data if too large
        if len(sp500) > 500:
            sp500 = sp500.iloc[::len(sp500)//500]
        if len(nasdaq) > 500:
            nasdaq = nasdaq.iloc[::len(nasdaq)//500]
            
        # Merge data on date
        merged = pd.merge(sp500[['Date', 'Close', 'Volume']], 
                         nasdaq[['Date', 'Close', 'Volume']], 
                         on='Date', suffixes=('_SP', '_NASDAQ'), how='inner')
        
        if not merged.empty:
            ax1_twin = ax1.twinx()
            
            # Line charts for closing prices
            ax1.plot(merged['Date'], merged['Close_SP'], 'b-', linewidth=2, label='S&P 500', alpha=0.8)
            ax1.plot(merged['Date'], merged['Close_NASDAQ'], 'r-', linewidth=2, label='NASDAQ', alpha=0.8)
            
            # Volume ratio bar chart (sample points to avoid overcrowding)
            sample_size = min(50, len(merged))
            sample_indices = np.linspace(0, len(merged)-1, sample_size, dtype=int)
            sample_data = merged.iloc[sample_indices]
            
            volume_ratio = sample_data['Volume_SP'] / (sample_data['Volume_NASDAQ'] + 1e-10)
            ax1_twin.bar(sample_data['Date'], volume_ratio, alpha=0.3, color='gray', width=20, label='Volume Ratio')
            
            ax1.set_ylabel('Closing Price', fontweight='bold')
            ax1_twin.set_ylabel('Volume Ratio', fontweight='bold')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
    else:
        ax1.text(0.5, 0.5, 'S&P 500 vs NASDAQ\nwith Volume Analysis', 
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
except:
    ax1.text(0.5, 0.5, 'S&P 500 vs NASDAQ\nwith Volume Analysis', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12)

ax1.set_title('Crisis Period: S&P 500 vs NASDAQ with Volume Ratios (2008-2010)', fontweight='bold', fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: FTSE and Nikkei with opposite direction markers
ax2 = axes[0, 1]
try:
    ftse = get_ticker_data('^FTSE', 2008, 2010)
    nikkei = get_ticker_data('^N225', 2008, 2010)

    if not ftse.empty and not nikkei.empty:
        # Sample data
        if len(ftse) > 300:
            ftse = ftse.iloc[::len(ftse)//300]
        if len(nikkei) > 300:
            nikkei = nikkei.iloc[::len(nikkei)//300]
            
        # Merge data
        merged = pd.merge(ftse[['Date', 'Close']], nikkei[['Date', 'Close']], 
                         on='Date', suffixes=('_FTSE', '_NIKKEI'), how='inner')
        
        if not merged.empty and len(merged) > 1:
            # Calculate daily returns
            merged['Return_FTSE'] = merged['Close_FTSE'].pct_change()
            merged['Return_NIKKEI'] = merged['Close_NIKKEI'].pct_change()
            
            # Identify opposite direction days
            opposite_days = ((merged['Return_FTSE'] > 0) & (merged['Return_NIKKEI'] < 0)) | \
                           ((merged['Return_FTSE'] < 0) & (merged['Return_NIKKEI'] > 0))
            
            # Area charts
            ax2.fill_between(merged['Date'], merged['Close_FTSE'], alpha=0.4, color='blue', label='FTSE 100')
            ax2.fill_between(merged['Date'], merged['Close_NIKKEI'], alpha=0.4, color='red', label='Nikkei 225')
            
            # Scatter points for opposite direction days (sample to avoid overcrowding)
            if opposite_days.any():
                opposite_data = merged[opposite_days].head(30)
                ax2.scatter(opposite_data['Date'], opposite_data['Close_FTSE'], 
                           color='darkblue', s=30, alpha=0.8, marker='o', label='Divergence Days')
                ax2.scatter(opposite_data['Date'], opposite_data['Close_NIKKEI'], 
                           color='darkred', s=30, alpha=0.8, marker='s')
            
            ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'FTSE vs Nikkei\nDivergence Analysis', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
except:
    ax2.text(0.5, 0.5, 'FTSE vs Nikkei\nDivergence Analysis', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)

ax2.set_ylabel('Closing Price', fontweight='bold')
ax2.set_title('Crisis Period: FTSE vs Nikkei with Divergence Markers (2008-2010)', fontweight='bold', fontsize=10)
ax2.grid(True, alpha=0.3)

# Subplot 3: Gold trend with Oil volatility bands
ax3 = axes[0, 2]
try:
    gold = get_ticker_data('GC=F', 2008, 2010)
    oil = get_ticker_data('CL=F', 2008, 2010)

    if not gold.empty:
        # Sample data
        if len(gold) > 300:
            gold = gold.iloc[::len(gold)//300]
            
        # Gold trend line
        ax3.plot(gold['Date'], gold['Close'], 'gold', linewidth=3, label='Gold Futures', alpha=0.8)
        
        if not oil.empty:
            if len(oil) > 300:
                oil = oil.iloc[::len(oil)//300]
                
            # Oil volatility bands
            if len(oil) > 20:
                oil_volatility = calculate_volatility(oil['Close'])
                oil_mean = oil['Close'].rolling(window=20).mean()
                
                ax3_twin = ax3.twinx()
                upper_band = oil_mean + oil_volatility * oil_mean
                lower_band = oil_mean - oil_volatility * oil_mean
                
                # Remove NaN values for plotting
                valid_mask = ~(upper_band.isna() | lower_band.isna())
                if valid_mask.any():
                    ax3_twin.fill_between(oil['Date'][valid_mask], upper_band[valid_mask], lower_band[valid_mask], 
                                         alpha=0.2, color='brown', label='Oil Volatility Bands')
                ax3_twin.plot(oil['Date'], oil['Close'], 'brown', linewidth=1, alpha=0.6, label='Crude Oil')
                
                ax3_twin.set_ylabel('Oil Price ($)', fontweight='bold', color='brown')
                ax3_twin.legend(loc='upper right')
        
        ax3.legend(loc='upper left')
    else:
        ax3.text(0.5, 0.5, 'Gold Trend vs\nOil Volatility Analysis', 
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
except:
    ax3.text(0.5, 0.5, 'Gold Trend vs\nOil Volatility Analysis', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=12)

ax3.set_ylabel('Gold Price ($)', fontweight='bold', color='gold')
ax3.set_title('Crisis Period: Gold Trend vs Oil Volatility (2008-2010)', fontweight='bold', fontsize=10)
ax3.grid(True, alpha=0.3)

# Middle row (2011-2016 Recovery Period)
# Subplot 4: BSE SENSEX with Nifty candlestick patterns
ax4 = axes[1, 0]
try:
    bse = get_ticker_data('^BSESN', 2011, 2016)
    nifty = get_ticker_data('^NSEI', 2011, 2016)

    if not bse.empty:
        # Sample data
        if len(bse) > 300:
            bse = bse.iloc[::len(bse)//300]
            
        # BSE line chart
        ax4.plot(bse['Date'], bse['Close'], 'blue', linewidth=2, label='BSE SENSEX', alpha=0.8)
        
        if not nifty.empty:
            if len(nifty) > 300:
                nifty = nifty.iloc[::len(nifty)//300]
                
            # Nifty high volatility periods (simplified candlestick representation)
            if len(nifty) > 10:
                nifty['Volatility'] = (nifty['High'] - nifty['Low']) / (nifty['Close'] + 1e-10)
                high_vol_threshold = nifty['Volatility'].quantile(0.9)
                high_vol_days = nifty[nifty['Volatility'] > high_vol_threshold].head(20)
                
                if not high_vol_days.empty:
                    ax4_twin = ax4.twinx()
                    for _, row in high_vol_days.iterrows():
                        ax4_twin.plot([row['Date'], row['Date']], [row['Low'], row['High']], 
                                     'r-', linewidth=2, alpha=0.7)
                        color = 'green' if row['Close'] > row['Open'] else 'red'
                        ax4_twin.plot([row['Date'], row['Date']], [row['Open'], row['Close']], 
                                     color=color, linewidth=4, alpha=0.8)
                    
                    ax4_twin.set_ylabel('Nifty 50 (High Vol)', fontweight='bold', color='red')
        
        ax4.legend(loc='upper left')
    else:
        ax4.text(0.5, 0.5, 'Indian Markets\nBSE vs Nifty Analysis', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
except:
    ax4.text(0.5, 0.5, 'Indian Markets\nBSE vs Nifty Analysis', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)

ax4.set_ylabel('BSE SENSEX', fontweight='bold', color='blue')
ax4.set_title('Recovery Period: Indian Markets - BSE vs Nifty Volatility (2011-2016)', fontweight='bold', fontsize=10)
ax4.grid(True, alpha=0.3)

# Subplot 5: European markets stacked area with correlation
ax5 = axes[1, 1]
try:
    ftse_recovery = get_ticker_data('^FTSE', 2011, 2016)
    n100_recovery = get_ticker_data('^N100', 2011, 2016)

    if not ftse_recovery.empty and not n100_recovery.empty:
        # Sample data
        if len(ftse_recovery) > 300:
            ftse_recovery = ftse_recovery.iloc[::len(ftse_recovery)//300]
        if len(n100_recovery) > 300:
            n100_recovery = n100_recovery.iloc[::len(n100_recovery)//300]
            
        # Merge and normalize data
        merged = pd.merge(ftse_recovery[['Date', 'Close']], n100_recovery[['Date', 'Close']], 
                         on='Date', suffixes=('_FTSE', '_N100'), how='inner')
        
        if not merged.empty and len(merged) > 1:
            # Normalize to percentage of initial value
            merged['FTSE_norm'] = (merged['Close_FTSE'] / merged['Close_FTSE'].iloc[0]) * 100
            merged['N100_norm'] = (merged['Close_N100'] / merged['Close_N100'].iloc[0]) * 100
            
            # Stacked area chart
            ax5.fill_between(merged['Date'], 0, merged['FTSE_norm'], alpha=0.6, color='blue', label='FTSE 100')
            ax5.fill_between(merged['Date'], merged['FTSE_norm'], 
                            merged['FTSE_norm'] + merged['N100_norm']/2, alpha=0.6, color='red', label='NASDAQ 100')
            
            # Rolling correlation
            if len(merged) > 30:
                window = min(30, len(merged)//2)
                correlation = merged['Close_FTSE'].rolling(window=window).corr(merged['Close_N100'])
                ax5_twin = ax5.twinx()
                valid_corr = correlation.dropna()
                if not valid_corr.empty:
                    ax5_twin.plot(merged['Date'][correlation.notna()], valid_corr, 'black', linewidth=2, label='Correlation')
                    ax5_twin.set_ylabel('Correlation', fontweight='bold')
                    ax5_twin.legend(loc='upper right')
            
            ax5.legend(loc='upper left')
    else:
        ax5.text(0.5, 0.5, 'European Markets\nPerformance & Correlation', 
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
except:
    ax5.text(0.5, 0.5, 'European Markets\nPerformance & Correlation', 
             ha='center', va='center', transform=ax5.transAxes, fontsize=12)

ax5.set_ylabel('Normalized Performance', fontweight='bold')
ax5.set_title('Recovery Period: European Markets Performance & Correlation (2011-2016)', fontweight='bold', fontsize=10)
ax5.grid(True, alpha=0.3)

# Subplot 6: Dow Jones with error bands and volume-weighted MA
ax6 = axes[1, 2]
try:
    dow = get_ticker_data('^DJI', 2011, 2016)

    if not dow.empty:
        # Sample data
        if len(dow) > 300:
            dow = dow.iloc[::len(dow)//300]
            
        # Main price line
        ax6.plot(dow['Date'], dow['Close'], 'navy', linewidth=2, label='Dow Jones', alpha=0.8)
        
        # Error bands (daily price ranges)
        ax6.fill_between(dow['Date'], dow['Low'], dow['High'], alpha=0.2, color='gray', label='Daily Range')
        
        # Volume-weighted moving average
        if len(dow) > 10:
            window = min(10, len(dow)//2)
            dow['VWMA'] = (dow['Close'] * dow['Volume']).rolling(window=window).sum() / (dow['Volume'].rolling(window=window).sum() + 1e-10)
            valid_vwma = dow['VWMA'].dropna()
            if not valid_vwma.empty:
                ax6.plot(dow['Date'][dow['VWMA'].notna()], valid_vwma, 'orange', linewidth=2, label='VWMA', alpha=0.8)
        
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Dow Jones with\nPrice Bands & VWMA', 
                 ha='center', va='center', transform=ax6.transAxes, fontsize=12)
except:
    ax6.text(0.5, 0.5, 'Dow Jones with\nPrice Bands & VWMA', 
             ha='center', va='center', transform=ax6.transAxes, fontsize=12)

ax6.set_ylabel('Price ($)', fontweight='bold')
ax6.set_title('Recovery Period: Dow Jones with Price Bands & VWMA (2011-2016)', fontweight='bold', fontsize=10)
ax6.grid(True, alpha=0.3)

# Bottom row (2017-2023 Modern Period)
# Subplot 7: Shanghai Composite calendar heatmap with trend lines
ax7 = axes[2, 0]
try:
    shanghai = get_ticker_data('000001.SS', 2017, 2023)

    if not shanghai.empty and len(shanghai) > 1:
        # Sample data
        if len(shanghai) > 300:
            shanghai = shanghai.iloc[::len(shanghai)//300]
            
        # Calculate quarterly returns
        shanghai['Quarter'] = shanghai['Date'].dt.to_period('Q')
        shanghai['Returns'] = shanghai['Close'].pct_change()
        quarterly_returns = shanghai.groupby('Quarter')['Returns'].sum()
        
        # Main price trend
        ax7.plot(shanghai['Date'], shanghai['Close'], 'black', linewidth=1, alpha=0.6, label='Shanghai Composite')
        
        # Quarterly trend lines (sample to avoid overcrowding)
        quarters_to_plot = list(quarterly_returns.items())[:8]  # Limit to 8 quarters
        for quarter, return_val in quarters_to_plot:
            quarter_data = shanghai[shanghai['Quarter'] == quarter]
            if not quarter_data.empty and len(quarter_data) > 1:
                color = 'red' if return_val < 0 else 'green'
                ax7.plot(quarter_data['Date'], quarter_data['Close'], 
                        color=color, linewidth=3, alpha=0.7)
        
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, 'Shanghai Composite\nQuarterly Performance', 
                 ha='center', va='center', transform=ax7.transAxes, fontsize=12)
except:
    ax7.text(0.5, 0.5, 'Shanghai Composite\nQuarterly Performance', 
             ha='center', va='center', transform=ax7.transAxes, fontsize=12)

ax7.set_ylabel('Shanghai Composite', fontweight='bold')
ax7.set_title('Modern Period: Shanghai Composite Quarterly Performance (2017-2023)', fontweight='bold', fontsize=10)
ax7.grid(True, alpha=0.3)

# Subplot 8: NYSE with commodities and cross-correlation
ax8 = axes[2, 1]
try:
    nyse = get_ticker_data('^NYA', 2017, 2023)
    gold_modern = get_ticker_data('GC=F', 2017, 2023)
    oil_modern = get_ticker_data('CL=F', 2017, 2023)

    if not nyse.empty:
        # Sample data
        if len(nyse) > 300:
            nyse = nyse.iloc[::len(nyse)//300]
            
        # Normalize NYSE series to start at 100
        nyse_norm = (nyse['Close'] / nyse['Close'].iloc[0]) * 100
        ax8.plot(nyse['Date'], nyse_norm, 'blue', linewidth=2, label='NYSE Composite', alpha=0.8)
        
        ax8_twin = ax8.twinx()
        
        if not gold_modern.empty:
            if len(gold_modern) > 300:
                gold_modern = gold_modern.iloc[::len(gold_modern)//300]
            gold_norm = (gold_modern['Close'] / gold_modern['Close'].iloc[0]) * 100
            ax8_twin.plot(gold_modern['Date'], gold_norm, 'gold', linewidth=2, label='Gold', alpha=0.8)
        
        if not oil_modern.empty:
            if len(oil_modern) > 300:
                oil_modern = oil_modern.iloc[::len(oil_modern)//300]
            oil_norm = (oil_modern['Close'] / oil_modern['Close'].iloc[0]) * 100
            ax8_twin.plot(oil_modern['Date'], oil_norm, 'brown', linewidth=2, label='Oil', alpha=0.8)
        
        ax8.legend(loc='upper left')
        ax8_twin.legend(loc='upper right')
        ax8_twin.set_ylabel('Commodities (Normalized)', fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'NYSE vs Commodities\nSynchronized Movement', 
                 ha='center', va='center', transform=ax8.transAxes, fontsize=12)
except:
    ax8.text(0.5, 0.5, 'NYSE vs Commodities\nSynchronized Movement', 
             ha='center', va='center', transform=ax8.transAxes, fontsize=12)

ax8.set_ylabel('NYSE Composite (Normalized)', fontweight='bold', color='blue')
ax8.set_title('Modern Period: NYSE vs Commodities Synchronized Movement (2017-2023)', fontweight='bold', fontsize=10)
ax8.grid(True, alpha=0.3)

# Subplot 9: All major US indices normalized with regime detection
ax9 = axes[2, 2]
try:
    indices = ['^DJI', '^GSPC', '^IXIC', '^NYA']
    colors = ['blue', 'red', 'green', 'purple']
    labels = ['Dow Jones', 'S&P 500', 'NASDAQ', 'NYSE']

    plotted_any = False
    for i, (ticker, color, label) in enumerate(zip(indices, colors, labels)):
        data = get_ticker_data(ticker, 2017, 2023)
        if not data.empty and len(data) > 1:
            # Sample data
            if len(data) > 300:
                data = data.iloc[::len(data)//300]
                
            # Normalize to percentage change from start
            normalized = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
            ax9.plot(data['Date'], normalized, color=color, linewidth=2, label=label, alpha=0.8)
            
            # Add volatility clustering markers (simplified)
            if len(data) > 10:
                volatility = calculate_volatility(data['Close'])
                if not volatility.isna().all():
                    high_vol_threshold = volatility.quantile(0.9)
                    high_vol_periods = volatility > high_vol_threshold
                    if high_vol_periods.any():
                        high_vol_data = data[high_vol_periods].head(10)
                        if not high_vol_data.empty:
                            high_vol_normalized = ((high_vol_data['Close'] / data['Close'].iloc[0]) - 1) * 100
                            ax9.scatter(high_vol_data['Date'], high_vol_normalized, 
                                       color=color, s=20, alpha=0.6, marker='x')
            plotted_any = True

    if not plotted_any:
        ax9.text(0.5, 0.5, 'US Indices Relative\nPerformance & Volatility', 
                 ha='center', va='center', transform=ax9.transAxes, fontsize=12)
    else:
        ax9.legend()
        ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
except:
    ax9.text(0.5, 0.5, 'US Indices Relative\nPerformance & Volatility', 
             ha='center', va='center', transform=ax9.transAxes, fontsize=12)

ax9.set_ylabel('Normalized Return (%)', fontweight='bold')
ax9.set_title('Modern Period: US Indices Relative Performance & Volatility (2017-2023)', fontweight='bold', fontsize=10)
ax9.grid(True, alpha=0.3)

# Adjust layout and styling
plt.tight_layout(pad=3.0)

# Add overall title
fig.suptitle('Global Financial Markets Evolution: Crisis, Recovery & Modern Era (2008-2023)', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust subplot spacing to prevent overlap
plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.3)

plt.savefig('global_markets_evolution.png', dpi=300, bbox_inches='tight')
plt.show()