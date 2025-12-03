import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Get all CSV files in the directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"Available CSV files: {csv_files}")

# Load and combine all data files
all_data = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"Successfully loaded {file} with shape {df.shape}")
        all_data.append(df)
    except Exception as e:
        print(f"Failed to load {file}: {e}")
        continue

# Check if we have any data
if not all_data:
    print("No data files could be loaded. Creating sample visualization...")
    # Create sample data for demonstration
    dates = pd.date_range('2008-01-01', '2023-12-31', freq='D')
    sample_data = []
    tickers = ['^NYA', '^IXIC', '^DJI', '^GSPC', '^NSEI', '^BSESN', '^N225', '000001.SS', '^FTSE', '^N100']
    
    for ticker in tickers:
        for date in dates[::30]:  # Sample every 30 days
            sample_data.append({
                'Ticker': ticker,
                'Date': date,
                'Close': np.random.uniform(1000, 5000),
                'Volume': np.random.uniform(1000000, 10000000)
            })
    
    df = pd.DataFrame(sample_data)
else:
    df = pd.concat(all_data, ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

# Create figure with 2x2 subplot grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define market groups
us_indices = ['^NYA', '^IXIC', '^DJI', '^GSPC']
asian_indices = ['^NSEI', '^BSESN', '^N225', '000001.SS']
european_indices = ['^FTSE', '^N100']

# Filter available tickers
available_tickers = df['Ticker'].unique()
us_indices = [t for t in us_indices if t in available_tickers]
asian_indices = [t for t in asian_indices if t in available_tickers]
european_indices = [t for t in european_indices if t in available_tickers]

print(f"Available US indices: {us_indices}")
print(f"Available Asian indices: {asian_indices}")
print(f"Available European indices: {european_indices}")

# Top-left: Multi-line time series with volatility bands
if us_indices:
    us_data = df[df['Ticker'].isin(us_indices)].copy()
    us_data = us_data.groupby(['Ticker', 'Date'])['Close'].first().reset_index()
    
    # Normalize to 100 at start
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for i, ticker in enumerate(us_indices[:4]):  # Limit to 4 colors
        ticker_data = us_data[us_data['Ticker'] == ticker].copy()
        if len(ticker_data) > 0:
            ticker_data = ticker_data.sort_values('Date')
            first_value = ticker_data['Close'].iloc[0]
            if first_value > 0:
                ticker_data['Normalized'] = (ticker_data['Close'] / first_value) * 100
                ax1.plot(ticker_data['Date'], ticker_data['Normalized'], 
                        label=ticker, color=colors[i % len(colors)], linewidth=2.5)
    
    # Add volatility bands for first available US index
    if len(us_indices) > 0:
        main_ticker = us_indices[0]
        main_data = us_data[us_data['Ticker'] == main_ticker].copy()
        if len(main_data) > 10:
            main_data = main_data.sort_values('Date')
            main_data['Returns'] = main_data['Close'].pct_change()
            main_data['Volatility'] = main_data['Returns'].rolling(min(30, len(main_data)//2)).std() * 100
            
            first_value = main_data['Close'].iloc[0]
            if first_value > 0:
                main_data['Normalized'] = (main_data['Close'] / first_value) * 100
                
                upper_band = main_data['Normalized'] + main_data['Volatility'].fillna(0)
                lower_band = main_data['Normalized'] - main_data['Volatility'].fillna(0)
                
                ax1.fill_between(main_data['Date'], lower_band, upper_band, 
                               alpha=0.2, color='gray', label=f'{main_ticker} Volatility Bands')

ax1.set_title('US Market Indices Performance with Volatility Bands (2008-2023)', 
              fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Normalized Price (Base 100)', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('white')

# Top-right: Dual-axis combination chart for Asian markets
if asian_indices:
    asian_data = df[df['Ticker'].isin(asian_indices)].copy()
    asian_annual = asian_data.groupby(['Ticker', asian_data['Date'].dt.year]).agg({
        'Close': 'mean',
        'Volume': 'mean'
    }).reset_index()
    
    years = sorted(asian_annual['Date'].unique())
    if years:
        x_pos = np.arange(len(years))
        width = 0.2
        
        colors_asian = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, ticker in enumerate(asian_indices[:4]):  # Limit to 4 tickers
            ticker_annual = asian_annual[asian_annual['Ticker'] == ticker]
            if len(ticker_annual) > 0:
                prices = []
                for year in years:
                    year_data = ticker_annual[ticker_annual['Date'] == year]
                    if len(year_data) > 0:
                        prices.append(year_data['Close'].iloc[0])
                    else:
                        prices.append(0)
                
                ax2.bar(x_pos + i*width, prices, width, label=ticker, 
                       color=colors_asian[i % len(colors_asian)], alpha=0.8)
        
        ax2.set_xticks(x_pos + width * 1.5)
        ax2.set_xticklabels(years, rotation=45)
        
        # Secondary y-axis for volume
        ax2_twin = ax2.twinx()
        for ticker in asian_indices[:2]:  # Limit volume lines for clarity
            ticker_annual = asian_annual[asian_annual['Ticker'] == ticker]
            if len(ticker_annual) > 0:
                volumes = []
                for year in years:
                    year_data = ticker_annual[ticker_annual['Date'] == year]
                    if len(year_data) > 0:
                        volumes.append(year_data['Volume'].iloc[0])
                    else:
                        volumes.append(0)
                
                ax2_twin.plot(years, volumes, marker='o', linewidth=2, alpha=0.7, label=f'{ticker} Volume')
        
        ax2_twin.set_ylabel('Average Trading Volume', fontweight='bold', color='gray')
        ax2_twin.legend(loc='upper right')

ax2.set_title('Asian Markets: Annual Average Prices with Volume Trends', 
              fontweight='bold', fontsize=14, pad=20)
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Average Closing Price', fontweight='bold', color='black')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('white')

# Bottom-left: Stacked area chart with market cap contribution
all_indices = us_indices + european_indices + asian_indices
if all_indices:
    market_data = df[df['Ticker'].isin(all_indices)].copy()
    market_annual = market_data.groupby(['Ticker', market_data['Date'].dt.year])['Close'].mean().reset_index()
    
    years = sorted(market_annual['Date'].unique())
    if years:
        us_contrib = []
        eu_contrib = []
        asia_contrib = []
        
        for year in years:
            year_data = market_annual[market_annual['Date'] == year]
            
            us_total = year_data[year_data['Ticker'].isin(us_indices)]['Close'].sum()
            eu_total = year_data[year_data['Ticker'].isin(european_indices)]['Close'].sum()
            asia_total = year_data[year_data['Ticker'].isin(asian_indices)]['Close'].sum()
            
            total = us_total + eu_total + asia_total
            if total > 0:
                us_contrib.append(us_total / total * 100)
                eu_contrib.append(eu_total / total * 100)
                asia_contrib.append(asia_total / total * 100)
            else:
                us_contrib.append(33.33)
                eu_contrib.append(33.33)
                asia_contrib.append(33.33)
        
        ax3.stackplot(years, us_contrib, eu_contrib, asia_contrib,
                      labels=['US Markets', 'European Markets', 'Asian Markets'],
                      colors=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
        
        # Add scatter points for crisis periods
        crisis_years = [2008, 2009, 2020]
        crisis_years_filtered = [year for year in crisis_years if year in years]
        if crisis_years_filtered:
            crisis_values = []
            for year in crisis_years_filtered:
                idx = years.index(year)
                crisis_values.append(us_contrib[idx] + eu_contrib[idx] + asia_contrib[idx])
            
            ax3.scatter(crisis_years_filtered, crisis_values, color='red', s=100, 
                       marker='*', label='Crisis Periods', zorder=5)

ax3.set_title('Regional Market Capitalization Contribution with Crisis Periods', 
              fontweight='bold', fontsize=14, pad=20)
ax3.set_xlabel('Year', fontweight='bold')
ax3.set_ylabel('Relative Contribution (%)', fontweight='bold')
ax3.legend(loc='center right')
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('white')

# Bottom-right: Correlation analysis
if all_indices and len(all_indices) > 1:
    # Calculate correlations for available data
    correlation_data = []
    avg_correlations = []
    
    years = sorted(df['Date'].dt.year.unique())
    valid_years = []
    
    for year in years:
        year_data = df[(df['Date'].dt.year == year) & (df['Ticker'].isin(all_indices))]
        if len(year_data) > 0:
            pivot_data = year_data.pivot_table(values='Close', index='Date', columns='Ticker')
            pivot_data = pivot_data.dropna(axis=1, how='all')  # Remove columns with all NaN
            
            if pivot_data.shape[1] > 1:  # Need at least 2 columns for correlation
                corr_matrix = pivot_data.corr()
                
                # Get upper triangle correlations (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                upper_triangle = corr_matrix.where(mask)
                avg_corr = upper_triangle.stack().mean()
                
                if not np.isnan(avg_corr):
                    avg_correlations.append(avg_corr)
                    correlation_data.append(corr_matrix)
                    valid_years.append(year)
    
    # Create heatmap for most recent correlation matrix
    if correlation_data:
        recent_corr = correlation_data[-1]
        
        # Create heatmap
        im = ax4.imshow(recent_corr.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax4.set_xticks(range(len(recent_corr.columns)))
        ax4.set_yticks(range(len(recent_corr.index)))
        ax4.set_xticklabels(recent_corr.columns, rotation=45, ha='right')
        ax4.set_yticklabels(recent_corr.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontweight='bold')
        
        # Overlay trend line
        if len(avg_correlations) > 1:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(range(len(valid_years)), avg_correlations, 
                          color='black', linewidth=3, marker='o', markersize=6,
                          label='Average Cross-Market Correlation')
            ax4_twin.set_ylabel('Average Correlation', fontweight='bold')
            ax4_twin.legend(loc='upper right')

ax4.set_title('Market Correlation Matrix with Trend Analysis', 
              fontweight='bold', fontsize=14, pad=20)
ax4.set_xlabel('Market Index', fontweight='bold')
ax4.set_ylabel('Market Index', fontweight='bold')

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.savefig('global_markets_analysis.png', dpi=300, bbox_inches='tight')
plt.show()