import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Define file mappings with exact names from the data description
file_mappings = {
    'Apple': 'apple.csv',
    'Samsung': 'samsung.csv', 
    'Google Pixel': 'pixel.csv',
    'Nokia': 'nokia.csv',
    'LG': 'lg.csv',
    'Xiaomi': 'xiaomi.csv',
    'Lenovo': 'lenovo.csv',
    'Vivo': 'VIVO.csv',
    'ZTE': 'zte.csv',
    'Alcatel': 'Alcatel Lucent.csv'
}

# Load and preprocess data with error handling
data = {}
successfully_loaded = []

for company, filename in file_mappings.items():
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Company'] = company
            # Handle missing volume data
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            # Convert volume to numeric, handling any string values
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            data[company] = df
            successfully_loaded.append(company)
            print(f"Successfully loaded {company}")
        else:
            print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading {company}: {e}")

print(f"Successfully loaded {len(successfully_loaded)} companies: {successfully_loaded}")

# Ensure we have enough data to proceed
if len(successfully_loaded) < 3:
    print("Not enough data files found. Creating sample visualization...")
    # Create sample data for demonstration
    dates = pd.date_range('2016-08-23', '2021-12-31', freq='D')
    for i, company in enumerate(['Apple', 'Samsung', 'Google Pixel']):
        np.random.seed(i)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        volumes = np.random.randint(1000000, 10000000, len(dates))
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Adj Close': prices,
            'Volume': volumes,
            'Company': company
        })
        data[company] = df
        successfully_loaded.append(company)

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 12), facecolor='white')
fig.patch.set_facecolor('white')

# Select companies for each subplot based on available data
available_companies = list(data.keys())
companies_top = available_companies[:3] if len(available_companies) >= 3 else available_companies
companies_mid = available_companies[3:6] if len(available_companies) >= 6 else available_companies[:3]
companies_right = available_companies[6:9] if len(available_companies) >= 9 else available_companies[:3]

# Subplot 1: Normalized closing prices with volume overlay
ax1 = plt.subplot(2, 3, 1)
ax1.set_facecolor('white')

colors_top = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, company in enumerate(companies_top):
    if company in data:
        df = data[company]
        # Normalize to starting value
        if len(df) > 0:
            normalized_close = (df['Close'] / df['Close'].iloc[0]) * 100
            ax1.plot(df['Date'], normalized_close, label=company, color=colors_top[i % len(colors_top)], linewidth=2)

ax1.set_title('Normalized Stock Prices with Combined Trading Volume', fontweight='bold', fontsize=12)
ax1.set_ylabel('Normalized Price (Base=100)', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Secondary axis for combined volume
ax1_vol = ax1.twinx()
if companies_top and companies_top[0] in data:
    reference_df = data[companies_top[0]]
    combined_volume = np.zeros(len(reference_df))
    
    for company in companies_top:
        if company in data:
            df = data[company]
            # Simple volume addition for overlapping dates
            if len(df) == len(reference_df):
                combined_volume += df['Volume'].values
            else:
                # Handle different length series
                min_len = min(len(combined_volume), len(df))
                combined_volume[:min_len] += df['Volume'].iloc[:min_len].values
    
    ax1_vol.fill_between(reference_df['Date'], combined_volume, alpha=0.3, color='gray', label='Combined Volume')
    ax1_vol.set_ylabel('Combined Trading Volume', fontweight='bold')
    ax1_vol.legend(loc='upper right')

# Subplot 2: Monthly averages with volatility bands
ax2 = plt.subplot(2, 3, 2)
ax2.set_facecolor('white')

colors_mid = ['#d62728', '#9467bd', '#8c564b']

for i, company in enumerate(companies_mid):
    if company in data:
        df = data[company]
        if len(df) > 0:
            # Calculate monthly statistics
            df_copy = df.copy()
            df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
            monthly_stats = df_copy.groupby('YearMonth').agg({
                'Close': ['mean', 'std'],
                'High': 'max',
                'Low': 'min'
            }).reset_index()
            
            monthly_stats.columns = ['YearMonth', 'Close_mean', 'Close_std', 'High_max', 'Low_min']
            monthly_stats['Date'] = monthly_stats['YearMonth'].dt.to_timestamp()
            
            # Plot monthly average
            ax2.plot(monthly_stats['Date'], monthly_stats['Close_mean'], 
                     label=f'{company} Avg', color=colors_mid[i % len(colors_mid)], linewidth=2)
            
            # Add volatility bands (high-low range)
            ax2.fill_between(monthly_stats['Date'], 
                             monthly_stats['Low_min'], 
                             monthly_stats['High_max'],
                             alpha=0.2, color=colors_mid[i % len(colors_mid)])

ax2.set_title('Monthly Average Prices with Volatility Bands', fontweight='bold', fontsize=12)
ax2.set_ylabel('Price', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Quarterly performance with trend lines
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor('white')

colors_right = ['#e377c2', '#7f7f7f', '#bcbd22']

# Calculate quarterly returns
quarterly_data = []
for company in companies_right:
    if company in data:
        df = data[company]
        if len(df) > 0:
            df_copy = df.copy()
            df_copy['Quarter'] = df_copy['Date'].dt.to_period('Q')
            quarterly = df_copy.groupby('Quarter').agg({
                'Close': ['first', 'last']
            }).reset_index()
            quarterly.columns = ['Quarter', 'Close_first', 'Close_last']
            quarterly['Return_pct'] = ((quarterly['Close_last'] - quarterly['Close_first']) / quarterly['Close_first']) * 100
            quarterly['Company'] = company
            quarterly_data.append(quarterly)

if quarterly_data:
    # Combine quarterly data
    quarterly_combined = pd.concat(quarterly_data, ignore_index=True)
    
    # Create grouped bar chart
    quarters = sorted(quarterly_combined['Quarter'].unique())
    x = np.arange(len(quarters))
    width = 0.25
    
    for i, company in enumerate(companies_right):
        if company in data:
            company_data = quarterly_combined[quarterly_combined['Company'] == company]
            returns = []
            for q in quarters:
                matching_data = company_data[company_data['Quarter'] == q]
                if len(matching_data) > 0:
                    returns.append(matching_data['Return_pct'].iloc[0])
                else:
                    returns.append(0)
            
            if len(returns) > 0:
                bars = ax3.bar(x + i*width, returns, width, label=company, 
                              color=colors_right[i % len(colors_right)], alpha=0.8)
                
                # Add trend line if we have enough data points
                if len(returns) > 1:
                    z = np.polyfit(x, returns, 1)
                    p = np.poly1d(z)
                    ax3.plot(x + i*width, p(x), color=colors_right[i % len(colors_right)], 
                            linestyle='--', alpha=0.7)

ax3.set_title('Quarterly Returns with Trend Lines', fontweight='bold', fontsize=12)
ax3.set_ylabel('Quarterly Return (%)', fontweight='bold')
ax3.set_xlabel('Quarter', fontweight='bold')
if 'quarterly_combined' in locals():
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([str(q) for q in quarters], rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Daily volume with moving averages
ax4 = plt.subplot(2, 3, 4)
ax4.set_facecolor('white')

if available_companies:
    # Use first available company as reference
    reference_company = available_companies[0]
    reference_df = data[reference_company]
    
    # Combine volumes from all companies
    total_volume = np.zeros(len(reference_df))
    for company in available_companies:
        if company in data:
            df = data[company]
            min_len = min(len(total_volume), len(df))
            total_volume[:min_len] += df['Volume'].iloc[:min_len].values
    
    # Sample every 20th day for readability
    sample_indices = range(0, len(reference_df), max(1, len(reference_df)//50))
    sampled_dates = reference_df['Date'].iloc[sample_indices]
    sampled_volume = total_volume[sample_indices]
    
    ax4.bar(sampled_dates, sampled_volume, alpha=0.6, color='lightblue', width=10)
    
    # Add moving averages on secondary axis
    ax4_ma = ax4.twinx()
    
    # Calculate moving averages for reference company
    ref_df = reference_df.copy()
    ref_df['MA30'] = ref_df['Close'].rolling(window=30, min_periods=1).mean()
    ref_df['MA90'] = ref_df['Close'].rolling(window=90, min_periods=1).mean()
    
    ax4_ma.plot(ref_df['Date'], ref_df['MA30'], label='30-day MA', color='red', linewidth=2)
    ax4_ma.plot(ref_df['Date'], ref_df['MA90'], label='90-day MA', color='darkred', linewidth=2)
    
    ax4_ma.set_ylabel('Price (Moving Average)', fontweight='bold')
    ax4_ma.legend()

ax4.set_title('Daily Trading Volume with Moving Averages', fontweight='bold', fontsize=12)
ax4.set_ylabel('Combined Daily Volume', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Subplot 5: Correlation heatmap
ax5 = plt.subplot(2, 3, 5)
ax5.set_facecolor('white')

# Calculate monthly returns for correlation
monthly_returns = pd.DataFrame()

for company in available_companies:
    if company in data:
        df = data[company].copy()
        if len(df) > 0:
            df['YearMonth'] = df['Date'].dt.to_period('M')
            monthly = df.groupby('YearMonth')['Close'].last().reset_index()
            if len(monthly) > 1:
                monthly['Return'] = monthly['Close'].pct_change() * 100
                monthly_returns[company] = monthly['Return']

# Remove NaN values and ensure we have data
monthly_returns = monthly_returns.dropna()

if len(monthly_returns.columns) > 1 and len(monthly_returns) > 0:
    # Create correlation matrix
    corr_matrix = monthly_returns.corr()
    
    # Create heatmap
    im = ax5.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    ax5.set_xticks(range(len(corr_matrix.columns)))
    ax5.set_yticks(range(len(corr_matrix)))
    ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax5.set_yticklabels(corr_matrix.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
             ha='center', va='center', transform=ax5.transAxes)

ax5.set_title('Monthly Returns Correlation Matrix', fontweight='bold', fontsize=12)

# Subplot 6: Box plots for return distributions
ax6 = plt.subplot(2, 3, 6)
ax6.set_facecolor('white')

if len(monthly_returns.columns) > 0 and len(monthly_returns) > 0:
    # Prepare data for box plots
    box_data = []
    box_labels = []
    
    for company in monthly_returns.columns:
        returns = monthly_returns[company].dropna()
        if len(returns) > 0:
            box_data.append(returns)
            box_labels.append(company)
    
    if box_data:
        # Create box plots
        bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        colors_box = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'No return data available', 
                ha='center', va='center', transform=ax6.transAxes)
else:
    ax6.text(0.5, 0.5, 'Insufficient data for distribution analysis', 
             ha='center', va='center', transform=ax6.transAxes)

ax6.set_title('Monthly Return Distributions', fontweight='bold', fontsize=12)
ax6.set_ylabel('Monthly Return (%)', fontweight='bold')
ax6.set_xlabel('Company', fontweight='bold')
ax6.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=3.0)
plt.savefig('smartphone_stock_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()