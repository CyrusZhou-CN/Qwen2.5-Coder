import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load all datasets with error handling
companies = {}
file_mapping = {
    'Apple': 'apple.csv',
    'Samsung': 'samsung.csv',
    'Google Pixel': 'pixel.csv',
    'Xiaomi': 'xiaomi.csv',
    'Nokia': 'nokia.csv',
    'LG': 'lg.csv',
    'Lenovo': 'lenovo.csv',
    'ZTE': 'zte.csv',
    'Alcatel': 'Alcatel Lucent.csv',
    'VIVO': 'VIVO.csv'
}

for name, filename in file_mapping.items():
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Returns'] = df['Close'].pct_change()
        companies[name] = df
    except:
        print(f"Could not load {filename}")

# Helper functions (optimized for speed)
def calculate_moving_average(data, window=30):
    return data.rolling(window=window, min_periods=1).mean()

def calculate_volatility_bands(data, window=30):
    ma = calculate_moving_average(data, window)
    std = data.rolling(window=window, min_periods=1).std()
    return ma + std, ma - std

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def normalize_prices(data, start_value=100):
    return (data / data.iloc[0]) * start_value

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color schemes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Row 1, Subplot 1: Apple vs Samsung with moving averages and volatility bands
ax1 = plt.subplot(3, 3, 1)
if 'Apple' in companies and 'Samsung' in companies:
    apple_data = companies['Apple']
    samsung_data = companies['Samsung']
    
    # Sample data for performance (every 5th point)
    apple_sample = apple_data.iloc[::5]
    samsung_sample = samsung_data.iloc[::5]
    
    ax1.plot(apple_sample['Date'], apple_sample['Close'], color=colors[0], linewidth=2, label='Apple', alpha=0.8)
    ax1.plot(samsung_sample['Date'], samsung_sample['Close'], color=colors[1], linewidth=2, label='Samsung', alpha=0.8)
    
    # Add moving averages
    apple_ma = calculate_moving_average(apple_sample['Close'])
    samsung_ma = calculate_moving_average(samsung_sample['Close'])
    ax1.plot(apple_sample['Date'], apple_ma, color=colors[0], linestyle='--', alpha=0.7, label='Apple 30-day MA')
    ax1.plot(samsung_sample['Date'], samsung_ma, color=colors[1], linestyle='--', alpha=0.7, label='Samsung 30-day MA')
    
    # Add volatility bands for Apple
    apple_upper, apple_lower = calculate_volatility_bands(apple_sample['Close'])
    ax1.fill_between(apple_sample['Date'], apple_upper, apple_lower, color=colors[0], alpha=0.1)

ax1.set_title('Apple vs Samsung: Price Evolution with Moving Averages', fontweight='bold', fontsize=10)
ax1.set_ylabel('Stock Price ($)')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Google Pixel vs Xiaomi with volume bars
ax2 = plt.subplot(3, 3, 2)
if 'Google Pixel' in companies and 'Xiaomi' in companies:
    pixel_data = companies['Google Pixel'].iloc[::5]  # Sample for performance
    xiaomi_data = companies['Xiaomi'].iloc[::3]  # Sample for performance
    
    ax2.plot(pixel_data['Date'], pixel_data['Close'], color=colors[2], linewidth=2, label='Google Pixel')
    ax2.plot(xiaomi_data['Date'], xiaomi_data['Close'], color=colors[3], linewidth=2, label='Xiaomi')
    
    # Add simple trend lines
    if len(pixel_data) > 1:
        pixel_trend = np.polyfit(range(len(pixel_data)), pixel_data['Close'], 1)
        ax2.plot(pixel_data['Date'], np.poly1d(pixel_trend)(range(len(pixel_data))), 
                color=colors[2], linestyle=':', alpha=0.7, label='Pixel Trend')
    
    if len(xiaomi_data) > 1:
        xiaomi_trend = np.polyfit(range(len(xiaomi_data)), xiaomi_data['Close'], 1)
        ax2.plot(xiaomi_data['Date'], np.poly1d(xiaomi_trend)(range(len(xiaomi_data))), 
                color=colors[3], linestyle=':', alpha=0.7, label='Xiaomi Trend')

ax2.set_title('Google Pixel vs Xiaomi: Prices with Trends', fontweight='bold', fontsize=10)
ax2.set_ylabel('Stock Price ($)')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# Row 1, Subplot 3: Nokia vs LG with normalized prices and RSI
ax3 = plt.subplot(3, 3, 3)
if 'Nokia' in companies and 'LG' in companies:
    nokia_data = companies['Nokia'].iloc[::5]  # Sample for performance
    lg_data = companies['LG'].iloc[::5]
    
    # Normalize prices
    nokia_norm = normalize_prices(nokia_data['Close'])
    lg_norm = normalize_prices(lg_data['Close'])
    
    ax3.plot(nokia_data['Date'], nokia_norm, color=colors[4], linewidth=2, label='Nokia (Normalized)')
    ax3.plot(lg_data['Date'], lg_norm, color=colors[5], linewidth=2, label='LG (Normalized)')
    
    # Add RSI as area chart
    nokia_rsi = calculate_rsi(nokia_data['Close'])
    ax3_rsi = ax3.twinx()
    ax3_rsi.fill_between(nokia_data['Date'], 0, nokia_rsi, alpha=0.3, color=colors[4], label='Nokia RSI')
    ax3_rsi.set_ylabel('RSI', color='gray', fontsize=8)
    ax3_rsi.set_ylim(0, 100)

ax3.set_title('Nokia vs LG: Normalized Prices with RSI', fontweight='bold', fontsize=10)
ax3.set_ylabel('Normalized Price')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

# Row 2, Subplot 4: Simplified candlestick representation
ax4 = plt.subplot(3, 3, 4)
if 'Apple' in companies:
    apple_data = companies['Apple']
    # Create monthly data
    apple_monthly = apple_data.set_index('Date').resample('M').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # Limit to first 24 months for performance
    apple_monthly = apple_monthly.head(24)
    
    # Simple candlestick representation
    for i, (date, row) in enumerate(apple_monthly.iterrows()):
        color = colors[0] if row['Close'] >= row['Open'] else colors[1]
        ax4.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
        ax4.plot([i-0.3, i+0.3], [row['Open'], row['Open']], color=color, linewidth=2)
        ax4.plot([i-0.3, i+0.3], [row['Close'], row['Close']], color=color, linewidth=3)
    
    # Add Bollinger Bands
    bb_upper, bb_lower = calculate_volatility_bands(apple_monthly['Close'], window=3)
    ax4.plot(range(len(apple_monthly)), bb_upper, color='red', linestyle='--', alpha=0.7, label='Upper BB')
    ax4.plot(range(len(apple_monthly)), bb_lower, color='red', linestyle='--', alpha=0.7, label='Lower BB')

ax4.set_title('Apple Monthly OHLC with Bollinger Bands', fontweight='bold', fontsize=10)
ax4.set_ylabel('Price ($)')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Box plots of daily returns by year
ax5 = plt.subplot(3, 3, 5)
box_data = []
box_labels = []

# Collect returns data by year (sample for performance)
for year in [2017, 2018, 2019, 2020, 2021]:
    year_returns = []
    for name, df in companies.items():
        df['Year'] = df['Date'].dt.year
        year_data = df[df['Year'] == year]['Returns'].dropna()
        if len(year_data) > 10:
            # Sample the data for performance
            sample_size = min(100, len(year_data))
            year_returns.extend(year_data.sample(sample_size).values)
    
    if year_returns:
        box_data.append(year_returns)
        box_labels.append(str(year))

if box_data:
    bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

ax5.set_title('Daily Returns Distribution by Year', fontweight='bold', fontsize=10)
ax5.set_ylabel('Daily Returns')
ax5.set_xlabel('Year')
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Cumulative returns with drawdowns
ax6 = plt.subplot(3, 3, 6)
target_companies = ['Lenovo', 'ZTE', 'Alcatel']

for i, name in enumerate(target_companies):
    if name in companies:
        df = companies[name].iloc[::5]  # Sample for performance
        returns = df['Returns'].fillna(0)
        cum_returns = (1 + returns).cumprod() - 1
        
        # Calculate drawdown
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / (peak + 1e-10)
        
        ax6.plot(df['Date'], cum_returns, color=colors[i], linewidth=2, label=f'{name} Returns')
        ax6.fill_between(df['Date'], 0, drawdown, where=(drawdown < 0), 
                        color=colors[i], alpha=0.3, label=f'{name} Drawdown')

ax6.set_title('Cumulative Returns with Drawdown', fontweight='bold', fontsize=10)
ax6.set_ylabel('Cumulative Returns')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# Row 3, Subplot 7: Seasonal analysis heatmap
ax7 = plt.subplot(3, 3, 7)
monthly_returns = np.zeros((len(companies), 12))
company_names = list(companies.keys())

for i, (name, df) in enumerate(companies.items()):
    df['Month'] = df['Date'].dt.month
    for month in range(1, 13):
        month_data = df[df['Month'] == month]['Returns'].dropna()
        if len(month_data) > 0:
            monthly_returns[i, month-1] = month_data.mean()

im = ax7.imshow(monthly_returns, cmap='RdYlBu_r', aspect='auto')
ax7.set_xticks(range(12))
ax7.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=8)
ax7.set_yticks(range(len(company_names)))
ax7.set_yticklabels(company_names, fontsize=7)

ax7.set_title('Seasonal Returns Heatmap', fontweight='bold', fontsize=10)

# Row 3, Subplot 8: Rolling correlation analysis (simplified)
ax8 = plt.subplot(3, 3, 8)
if 'Apple' in companies and 'Samsung' in companies:
    # Calculate simple correlation over time windows
    apple_returns = companies['Apple']['Returns'].dropna()
    samsung_returns = companies['Samsung']['Returns'].dropna()
    
    # Align the data
    min_len = min(len(apple_returns), len(samsung_returns))
    apple_aligned = apple_returns.iloc[:min_len]
    samsung_aligned = samsung_returns.iloc[:min_len]
    
    # Calculate rolling correlation with larger window for performance
    window = 60
    correlations = []
    dates = []
    
    for i in range(window, min_len, 10):  # Step by 10 for performance
        corr = apple_aligned.iloc[i-window:i].corr(samsung_aligned.iloc[i-window:i])
        if not np.isnan(corr):
            correlations.append(corr)
            dates.append(companies['Apple']['Date'].iloc[i])
    
    if correlations:
        ax8.plot(dates, correlations, color=colors[0], linewidth=2, label='Apple-Samsung Correlation')

ax8.set_title('Rolling Correlation Analysis', fontweight='bold', fontsize=10)
ax8.set_ylabel('Correlation Coefficient')
ax8.legend(fontsize=7)
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Row 3, Subplot 9: Risk-return scatter
ax9 = plt.subplot(3, 3, 9)
annual_returns = []
annual_volatilities = []
avg_volumes = []
company_labels = []

for name, df in companies.items():
    returns = df['Returns'].dropna()
    if len(returns) > 100:
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        avg_volume = df['Volume'].mean()
        
        annual_returns.append(annual_return)
        annual_volatilities.append(annual_vol)
        avg_volumes.append(avg_volume)
        company_labels.append(name)

if annual_returns:
    # Normalize bubble sizes
    max_volume = max(avg_volumes)
    bubble_sizes = [(vol/max_volume) * 300 + 50 for vol in avg_volumes]
    
    scatter = ax9.scatter(annual_volatilities, annual_returns, 
                         s=bubble_sizes, c=colors[:len(company_labels)], 
                         alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add company labels
    for i, label in enumerate(company_labels):
        ax9.annotate(label, (annual_volatilities[i], annual_returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=7)

ax9.set_title('Risk-Return Analysis', fontweight='bold', fontsize=10)
ax9.set_xlabel('Annualized Volatility')
ax9.set_ylabel('Annualized Return')
ax9.grid(True, alpha=0.3)

# Final layout adjustment
plt.tight_layout(pad=1.5)
plt.savefig('smartphone_stock_analysis.png', dpi=300, bbox_inches='tight')
plt.show()