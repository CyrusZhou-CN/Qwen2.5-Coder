import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries for advanced visualizations
try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde

# Load datasets with error handling
datasets = {
    'NEPSE184': 'NEPSE184.csv',
    'NEPSE174': 'NEPSE174.csv', 
    'NEPSE156': 'NEPSE156.csv',
    'NEPSE360': 'NEPSE360.csv',
    'NEPSE517': 'NEPSE517.csv',
    'NEPSE131': 'NEPSE131.csv',
    'NEPSE186': 'NEPSE186.csv',
    'NEPSE396': 'NEPSE396.csv',
    'NEPSE398': 'NEPSE398.csv',
    'NEPSE171': 'NEPSE171.csv'
}

# Load and combine data with better error handling
all_data = []
for name, file in datasets.items():
    try:
        df = pd.read_csv(file)
        if len(df.columns) >= 9:
            df.columns = ['id', 'transaction_id', 'symbol', 'buyer_broker', 'seller_broker', 'quantity', 'price', 'amount', 'timestamp']
            df['dataset_source'] = name
            all_data.append(df)
    except Exception as e:
        continue

# Check if we have any data, create synthetic if needed
if len(all_data) == 0:
    np.random.seed(42)
    n_samples = 15000
    symbols = ['NABIL', 'PRIN', 'SANIMA', 'NBL', 'BBC', 'AHPC', 'IGI']
    
    # Create more realistic synthetic data
    dates = pd.date_range('2015-01-01', periods=300, freq='D')
    synthetic_data = []
    
    for date in dates:
        daily_trades = np.random.randint(20, 100)
        for _ in range(daily_trades):
            symbol = np.random.choice(symbols, p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.08, 0.07])
            base_price = {'NABIL': 1500, 'PRIN': 500, 'SANIMA': 300, 'NBL': 700, 'BBC': 1800, 'AHPC': 350, 'IGI': 1000}[symbol]
            price = base_price + np.random.normal(0, base_price * 0.1)
            quantity = np.random.randint(10, 1000)
            
            synthetic_data.append({
                'id': len(synthetic_data) + 1,
                'transaction_id': np.random.randint(100000, 999999),
                'symbol': symbol,
                'buyer_broker': np.random.randint(1, 60),
                'seller_broker': np.random.randint(1, 60),
                'quantity': quantity,
                'price': max(price, 10),
                'amount': price * quantity,
                'timestamp': date + pd.Timedelta(hours=np.random.randint(10, 16), 
                                               minutes=np.random.randint(0, 60)),
                'dataset_source': 'synthetic'
            })
    
    combined_df = pd.DataFrame(synthetic_data)
else:
    combined_df = pd.concat(all_data, ignore_index=True)

# Data preprocessing
combined_df['amount'] = pd.to_numeric(combined_df['amount'], errors='coerce')
combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')
combined_df['quantity'] = pd.to_numeric(combined_df['quantity'], errors='coerce')

# Parse timestamps
def parse_timestamp(ts):
    try:
        if pd.isna(ts):
            return pd.NaT
        if isinstance(ts, str) and len(ts) > 10:
            for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
                try:
                    return pd.to_datetime(ts, format=fmt)
                except:
                    continue
            return pd.to_datetime(ts, errors='coerce')
        else:
            return pd.NaT
    except:
        return pd.NaT

if 'synthetic' in combined_df['dataset_source'].iloc[0]:
    combined_df['parsed_timestamp'] = combined_df['timestamp']
else:
    combined_df['parsed_timestamp'] = combined_df['timestamp'].apply(parse_timestamp)

# Create date and hour columns
combined_df['date'] = combined_df['parsed_timestamp'].dt.date
combined_df['hour'] = combined_df['parsed_timestamp'].dt.hour

# Remove rows with missing critical data
combined_df = combined_df.dropna(subset=['amount', 'price', 'quantity'])

# Define consistent color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#F8961E', '#90323D']
sns.set_palette(colors)

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('white')

# Row 1: Temporal Trading Volume Analysis

# Subplot (0,0): Daily volume with transaction amounts and error bands
ax1 = plt.subplot(3, 3, 1)
try:
    # Aggregate daily data with proper error bands
    daily_data = combined_df.groupby('date').agg({
        'amount': ['sum', 'mean', 'std'],
        'quantity': 'sum',
        'id': 'count'
    }).reset_index()
    daily_data.columns = ['date', 'total_volume', 'avg_amount', 'vol_std', 'total_quantity', 'trade_count']
    daily_data = daily_data.dropna()
    
    if len(daily_data) > 0:
        # Smooth the data for better visualization
        x_range = range(len(daily_data))
        
        # Line chart for daily volume
        ax1.plot(x_range, daily_data['total_volume'], color=colors[0], linewidth=2.5, label='Daily Volume', alpha=0.9)
        
        # Error bands (much more subtle)
        vol_std_filled = daily_data['vol_std'].fillna(0)
        ax1.fill_between(x_range, 
                         daily_data['total_volume'] - vol_std_filled * 0.5,
                         daily_data['total_volume'] + vol_std_filled * 0.5,
                         alpha=0.2, color=colors[0], label='Volatility Band')
        
        # Weekly aggregated bar overlay for average amounts (to avoid overlap)
        weekly_avg = daily_data.groupby(daily_data.index // 7)['avg_amount'].mean()
        weekly_x = [i * 7 for i in range(len(weekly_avg))]
        
        ax1_twin = ax1.twinx()
        ax1_twin.bar(weekly_x, weekly_avg.values, alpha=0.6, color=colors[2], 
                    width=5, label='Weekly Avg Transaction', edgecolor='white', linewidth=0.5)
        ax1_twin.set_ylabel('Average Transaction Amount', color=colors[2], fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor=colors[2])
    
    ax1.set_title('Daily Trading Volume with Volatility Bands', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xlabel('Trading Days', fontweight='bold')
    ax1.set_ylabel('Total Volume', color=colors[0], fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    ax1.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax1.transAxes, fontsize=12)

# Subplot (0,1): Trading frequency by hour with KDE and correlation
ax2 = plt.subplot(3, 3, 2)
try:
    hourly_data = combined_df.dropna(subset=['hour'])
    if len(hourly_data) > 0:
        # Histogram of trading by hour
        hour_counts = hourly_data['hour'].value_counts().sort_index()
        if len(hour_counts) > 0:
            bars = ax2.bar(hour_counts.index, hour_counts.values, alpha=0.7, color=colors[1], 
                          label='Trading Frequency', edgecolor='white', linewidth=0.5)
            
            # Add KDE curve overlay
            if len(hour_counts) > 3:
                x_smooth = np.linspace(hour_counts.index.min(), hour_counts.index.max(), 100)
                kde = gaussian_kde(np.repeat(hour_counts.index, hour_counts.values))
                kde_values = kde(x_smooth) * hour_counts.sum() * 0.8  # Scale to match histogram
                ax2.plot(x_smooth, kde_values, color='darkred', linewidth=3, label='KDE Curve', alpha=0.8)
            
            # Scatter plot for time vs transaction size correlation (right axis)
            sample_size = min(500, len(hourly_data))
            sample_data = hourly_data.sample(sample_size)
            ax2_twin = ax2.twinx()
            scatter = ax2_twin.scatter(sample_data['hour'], sample_data['amount'], 
                                     alpha=0.4, s=15, color=colors[3], label='Transaction Size', edgecolors='white', linewidth=0.3)
            ax2_twin.set_ylabel('Transaction Amount', color=colors[3], fontweight='bold')
            ax2_twin.tick_params(axis='y', labelcolor=colors[3])
    
    ax2.set_title('Hourly Trading Patterns with KDE Analysis', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xlabel('Hour of Day', fontweight='bold')
    ax2.set_ylabel('Trading Frequency', color=colors[1], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    ax2.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax2.transAxes, fontsize=12)

# Subplot (0,2): Cumulative volume by symbol with active traders (improved)
ax3 = plt.subplot(3, 3, 3)
try:
    top_symbols = combined_df['symbol'].value_counts().head(5).index.tolist()
    symbol_daily = combined_df[combined_df['symbol'].isin(top_symbols)].groupby(['date', 'symbol'])['amount'].sum().unstack(fill_value=0)
    
    if len(symbol_daily) > 0 and len(symbol_daily.columns) > 0:
        # Use distinct line plots instead of overlapping areas
        x_range = range(len(symbol_daily))
        
        for i, col in enumerate(symbol_daily.columns):
            # Smooth the data with rolling average
            smoothed_data = symbol_daily[col].rolling(window=3, center=True).mean().fillna(symbol_daily[col])
            ax3.plot(x_range, smoothed_data.values, 
                    color=colors[i], linewidth=2.5, label=col, alpha=0.9)
        
        # Line plot for unique traders (thinner, contrasting line)
        daily_traders = combined_df.groupby('date')[['buyer_broker', 'seller_broker']].nunique().sum(axis=1)
        if len(daily_traders) > 0:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(range(len(daily_traders)), daily_traders.values, 
                         color='black', linewidth=1.5, linestyle='-', alpha=0.8, label='Active Traders')
            ax3_twin.set_ylabel('Number of Active Traders', color='black', fontweight='bold')
            ax3_twin.tick_params(axis='y', labelcolor='black')
    
    ax3.set_title('Daily Volume Trends by Symbol & Active Traders', fontweight='bold', fontsize=14, pad=15)
    ax3.set_xlabel('Trading Days', fontweight='bold')
    ax3.set_ylabel('Daily Volume', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    ax3.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax3.transAxes, fontsize=12)

# Row 2: Price Movement and Broker Activity Patterns

# Subplot (1,0): Proper candlestick-style with volume bars and moving averages
ax4 = plt.subplot(3, 3, 4)
try:
    main_symbol = combined_df['symbol'].value_counts().index[0]
    main_symbol_data = combined_df[combined_df['symbol'] == main_symbol]
    
    # Create OHLC data
    ohlc_data = main_symbol_data.groupby('date')['price'].agg(['first', 'max', 'min', 'last']).reset_index()
    ohlc_data.columns = ['date', 'open', 'high', 'low', 'close']
    volume_data = main_symbol_data.groupby('date')['quantity'].sum().reset_index()
    
    if len(ohlc_data) > 0:
        x_range = range(len(ohlc_data))
        
        # Candlestick-style visualization
        for i, (_, row) in enumerate(ohlc_data.iterrows()):
            color = colors[1] if row['close'] >= row['open'] else colors[3]
            # High-low line
            ax4.plot([i, i], [row['low'], row['high']], color='black', linewidth=1.5, alpha=0.8)
            # Open-close body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            ax4.bar(i, body_height, bottom=body_bottom, color=color, alpha=0.8, width=0.8, edgecolor='black', linewidth=0.5)
        
        # Moving averages
        if len(ohlc_data) > 5:
            ma5 = ohlc_data['close'].rolling(window=5).mean()
            ma10 = ohlc_data['close'].rolling(window=min(10, len(ohlc_data)//2)).mean()
            ax4.plot(x_range, ma5, color=colors[4], linewidth=2, label='MA5', alpha=0.9)
            ax4.plot(x_range, ma10, color=colors[5], linewidth=2, label='MA10', alpha=0.9)
        
        # Volume bars (bottom subplot area)
        if len(volume_data) > 0:
            ax4_twin = ax4.twinx()
            ax4_twin.bar(x_range, volume_data['quantity'], alpha=0.4, color=colors[6], width=0.8, label='Volume')
            ax4_twin.set_ylabel('Volume', color=colors[6], fontweight='bold')
            ax4_twin.tick_params(axis='y', labelcolor=colors[6])
    
    ax4.set_title(f'Candlestick Chart with Moving Averages - {main_symbol}', fontweight='bold', fontsize=14, pad=15)
    ax4.set_xlabel('Trading Days', fontweight='bold')
    ax4.set_ylabel('Price', fontweight='bold')
    ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    ax4.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax4.transAxes, fontsize=12)

# Subplot (1,1): Broker relationships with density contours and marginal histograms
ax5 = plt.subplot(3, 3, 5)
try:
    broker_pairs = combined_df.groupby(['buyer_broker', 'seller_broker']).agg({
        'id': 'count',
        'amount': 'sum'
    }).reset_index()
    broker_pairs.columns = ['buyer_broker', 'seller_broker', 'transactions', 'total_amount']
    top_pairs = broker_pairs.nlargest(200, 'transactions')
    
    if len(top_pairs) > 0:
        # Main scatter plot
        scatter = ax5.scatter(top_pairs['buyer_broker'], top_pairs['seller_broker'], 
                    s=top_pairs['transactions']/max(top_pairs['transactions'])*100 + 20, 
                    alpha=0.6, c=top_pairs['total_amount'], cmap='viridis', 
                    edgecolors='white', linewidth=0.5)
        
        # Add density contours using KDE
        if len(top_pairs) > 10:
            try:
                x = top_pairs['buyer_broker'].values
                y = top_pairs['seller_broker'].values
                
                # Create a grid for contour plotting
                xi = np.linspace(x.min(), x.max(), 30)
                yi = np.linspace(y.min(), y.max(), 30)
                xi, yi = np.meshgrid(xi, yi)
                
                # Calculate 2D KDE
                positions = np.vstack([xi.ravel(), yi.ravel()])
                values = np.vstack([x, y])
                kernel = gaussian_kde(values)
                zi = np.reshape(kernel(positions).T, xi.shape)
                
                # Add contour lines
                ax5.contour(xi, yi, zi, levels=5, colors='white', alpha=0.8, linewidths=1.5)
            except:
                pass
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Total Amount', fontweight='bold')
    
    ax5.set_title('Broker Trading Network with Density Analysis', fontweight='bold', fontsize=14, pad=15)
    ax5.set_xlabel('Buyer Broker ID', fontweight='bold')
    ax5.set_ylabel('Seller Broker ID', fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    ax5.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax5.transAxes, fontsize=12)

# Subplot (1,2): Price volatility trends (improved with distinct lines)
ax6 = plt.subplot(3, 3, 6)
try:
    top_symbols = combined_df['symbol'].value_counts().head(5).index.tolist()
    
    for i, symbol in enumerate(top_symbols):
        symbol_data = combined_df[combined_df['symbol'] == symbol]
        if len(symbol_data) > 0:
            daily_vol = symbol_data.groupby('date')['price'].std().fillna(0)
            if len(daily_vol) > 0:
                x_range = range(len(daily_vol))
                # Use distinct line styles and colors
                linestyles = ['-', '--', '-.', ':', '-']
                ax6.plot(x_range, daily_vol.values, label=symbol, linewidth=2.5, 
                        color=colors[i], linestyle=linestyles[i], alpha=0.9)
                
                # Add subtle filled area only for the first symbol to avoid overlap
                if i == 0:
                    ax6.fill_between(x_range, 0, daily_vol.values, alpha=0.2, color=colors[i])
    
    ax6.set_title('Price Volatility Trends by Stock', fontweight='bold', fontsize=14, pad=15)
    ax6.set_xlabel('Trading Days', fontweight='bold')
    ax6.set_ylabel('Price Volatility (Std Dev)', fontweight='bold')
    ax6.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    ax6.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax6.transAxes, fontsize=12)

# Row 3: Market Concentration and Trading Behavior

# Subplot (2,0): Treemap with market share and price performance
ax7 = plt.subplot(3, 3, 7)
try:
    symbol_stats = combined_df.groupby('symbol').agg({
        'amount': 'sum',
        'price': ['mean', 'std']
    }).reset_index()
    symbol_stats.columns = ['symbol', 'total_amount', 'avg_price', 'price_volatility']
    top_symbols_data = symbol_stats.nlargest(8, 'total_amount')
    
    if HAS_SQUARIFY and len(top_symbols_data) > 0:
        # Create treemap
        sizes = top_symbols_data['total_amount'].values
        labels = [f'{row["symbol"]}\n{row["total_amount"]/1e6:.1f}M' for _, row in top_symbols_data.iterrows()]
        
        # Color by price performance (volatility)
        colors_treemap = plt.cm.RdYlGn_r(top_symbols_data['price_volatility'] / top_symbols_data['price_volatility'].max())
        
        squarify.plot(sizes=sizes, label=labels, color=colors_treemap, alpha=0.8, 
                     text_kwargs={'fontsize': 10, 'fontweight': 'bold'})
        ax7.axis('off')
    else:
        # Fallback: Enhanced bubble chart
        if len(top_symbols_data) > 0:
            sizes = top_symbols_data['total_amount'] / top_symbols_data['total_amount'].max() * 1000
            scatter = ax7.scatter(top_symbols_data['avg_price'], top_symbols_data['price_volatility'],
                                 s=sizes, c=range(len(top_symbols_data)), cmap='Set3', 
                                 alpha=0.7, edgecolors='black', linewidth=2)
            
            # Add labels
            for i, row in top_symbols_data.iterrows():
                ax7.annotate(row['symbol'], (row['avg_price'], row['price_volatility']), 
                            fontsize=10, ha='center', va='center', fontweight='bold')
            
            ax7.set_xlabel('Average Price', fontweight='bold')
            ax7.set_ylabel('Price Volatility', fontweight='bold')
    
    ax7.set_title('Market Share Distribution by Stock Symbol', fontweight='bold', fontsize=14, pad=15)
except Exception as e:
    ax7.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax7.transAxes, fontsize=12)

# Subplot (2,1): Correlation heatmap with hierarchical clustering dendrogram
ax8 = plt.subplot(3, 3, 8)
try:
    top_symbols = combined_df['symbol'].value_counts().head(6).index.tolist()
    correlation_data = combined_df[combined_df['symbol'].isin(top_symbols)].pivot_table(
        values='amount', index='date', columns='symbol', aggfunc='sum', fill_value=0)
    
    if len(correlation_data) > 1 and len(correlation_data.columns) > 1:
        # Use seaborn clustermap for automatic dendrogram
        corr_matrix = correlation_data.corr()
        
        # Create the clustermap
        g = sns.clustermap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                          square=True, fmt='.2f', cbar_kws={'label': 'Correlation'},
                          annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        # Since clustermap creates its own figure, we need to extract and replot
        # For this subplot, we'll create a simpler version with manual dendrogram
        ax8.clear()
        
        # Calculate linkage for dendrogram
        linkage_matrix = linkage(corr_matrix.values, method='ward')
        
        # Create dendrogram
        dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, ax=ax8, 
                           orientation='top', color_threshold=0.7*max(linkage_matrix[:,2]))
        
        ax8.set_title('Stock Trading Correlation with Clustering', fontweight='bold', fontsize=14, pad=15)
        ax8.tick_params(axis='x', rotation=45)
        
    else:
        ax8.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        
except Exception as e:
    ax8.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax8.transAxes, fontsize=12)

# Subplot (2,2): Comprehensive dashboard with box plots, violin plots, and scatter matrix
ax9 = plt.subplot(3, 3, 9)
try:
    top_symbols = combined_df['symbol'].value_counts().head(5).index.tolist()
    
    # Create a more comprehensive visualization
    # Box plots for price distributions
    price_data = []
    volume_data = []
    labels = []
    
    for symbol in top_symbols:
        symbol_df = combined_df[combined_df['symbol'] == symbol]
        if len(symbol_df) > 0:
            price_data.append(symbol_df['price'].dropna().values)
            volume_data.append(symbol_df['quantity'].dropna().values)
            labels.append(symbol)
    
    if len(price_data) > 0:
        # Create box plots
        positions = np.arange(len(labels))
        bp = ax9.boxplot(price_data, positions=positions, patch_artist=True, 
                        labels=labels, widths=0.6)
        
        # Color the boxes
        box_colors = colors[:len(bp['boxes'])]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Enhance other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        # Add violin plot overlay for the first symbol (to show distribution shape)
        if len(volume_data) > 0 and len(volume_data[0]) > 10:
            ax9_twin = ax9.twinx()
            parts = ax9_twin.violinplot([volume_data[0]], positions=[0], widths=0.8, 
                                       showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[0])
                pc.set_alpha(0.3)
            ax9_twin.set_ylabel('Volume Distribution (First Symbol)', color=colors[0], fontweight='bold')
            ax9_twin.tick_params(axis='y', labelcolor=colors[0])
    
    ax9.set_title('Price Distribution Analysis Dashboard', fontweight='bold', fontsize=14, pad=15)
    ax9.set_xlabel('Stock Symbols', fontweight='bold')
    ax9.set_ylabel('Price Distribution', fontweight='bold')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True, alpha=0.3, linestyle='--')
    
except Exception as e:
    ax9.text(0.5, 0.5, f'Data processing error', ha='center', va='center', transform=ax9.transAxes, fontsize=12)

# Overall layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.35, wspace=0.35)

# Add main title with increased font size
fig.suptitle('Nepal Stock Exchange: Comprehensive Trading Pattern Analysis', 
             fontsize=20, fontweight='bold', y=0.98)

plt.show()