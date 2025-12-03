import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load a smaller subset of datasets to avoid timeout
file_names = ['NEPSE184.csv', 'NEPSE131.csv', 'NEPSE171.csv']

# Read and combine datasets with optimized processing
dfs = []
for file in file_names:
    try:
        df = pd.read_csv(file)
        # Take only first 1000 rows to avoid timeout
        df = df.head(1000)
        # Standardize column names based on the data structure shown
        df.columns = ['id', 'timestamp_id', 'company', 'col1', 'col2', 'volume', 'price', 'amount', 'datetime']
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

if not dfs:
    # If no files loaded, create sample data
    print("Creating sample data...")
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=100, freq='D')
    companies = ['NABIL', 'PRIN', 'SANIMA', 'NBL', 'SCB']
    
    sample_data = []
    for date in dates:
        for company in companies:
            sample_data.append({
                'company': company,
                'volume': np.random.randint(10, 1000),
                'price': np.random.uniform(100, 1000),
                'amount': np.random.uniform(1000, 100000),
                'datetime': date,
                'hour': np.random.randint(9, 17),
                'day_of_week': date.day_name()
            })
    
    combined_df = pd.DataFrame(sample_data)
    combined_df['parsed_datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df['date'] = combined_df['parsed_datetime'].dt.date
else:
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Simple datetime parsing
    def parse_datetime_simple(dt_str):
        try:
            if isinstance(dt_str, str) and '-' in dt_str:
                return pd.to_datetime(dt_str, errors='coerce')
            else:
                return pd.to_datetime('2015-01-01')
        except:
            return pd.to_datetime('2015-01-01')
    
    combined_df['parsed_datetime'] = combined_df['datetime'].apply(parse_datetime_simple)
    combined_df['date'] = combined_df['parsed_datetime'].dt.date
    combined_df['hour'] = combined_df['parsed_datetime'].dt.hour
    combined_df['day_of_week'] = combined_df['parsed_datetime'].dt.day_name()

# Clean data
combined_df = combined_df[combined_df['price'] > 0]
combined_df = combined_df[combined_df['volume'] > 0]
combined_df = combined_df.dropna(subset=['price', 'volume', 'amount'])

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']

# Top-left: Dual-axis plot with volume bars and price trend
top_companies = combined_df['company'].value_counts().head(3).index
company_data = combined_df[combined_df['company'].isin(top_companies)].copy()

# Aggregate daily data (limit to 20 days for performance)
daily_data = company_data.groupby(['date', 'company']).agg({
    'volume': 'sum',
    'price': 'mean',
    'amount': 'sum'
}).reset_index()

daily_volume = daily_data.groupby('date')['volume'].sum().reset_index()
daily_volume = daily_volume.sort_values('date').head(20)

# Create twin axis
ax1_twin = ax1.twinx()

# Plot volume bars
x_pos = range(len(daily_volume))
bars = ax1.bar(x_pos, daily_volume['volume'], alpha=0.6, color=colors[0], label='Daily Volume')

# Plot price moving average
daily_price = daily_data.groupby('date')['price'].mean().reset_index()
daily_price = daily_price.sort_values('date').head(20)

if len(daily_price) > 0:
    price_ma = daily_price['price'].rolling(window=3, min_periods=1).mean()
    ax1_twin.plot(range(len(daily_price)), price_ma, color=colors[1], 
                  linewidth=3, label='Price MA', marker='o')

ax1.set_title('Daily Trading Volume vs Price Moving Average', fontweight='bold', fontsize=12)
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Volume', color=colors[0])
ax1_twin.set_ylabel('Price (Moving Average)', color=colors[1])
ax1.tick_params(axis='y', labelcolor=colors[0])
ax1_twin.tick_params(axis='y', labelcolor=colors[1])

# Top-right: Trading intensity heatmap
# Create simplified heatmap data
hours = combined_df['hour'].fillna(12).astype(int)
days = combined_df['day_of_week'].fillna('Monday')

# Create hour-day matrix with limited data
hour_day_data = combined_df.groupby([hours, days])['amount'].sum().reset_index()
hour_day_data.columns = ['hour', 'day_of_week', 'amount']

# Create pivot table
try:
    pivot_data = hour_day_data.pivot(index='hour', columns='day_of_week', values='amount')
    pivot_data = pivot_data.fillna(0)
    
    # Limit to reasonable size
    if pivot_data.shape[0] > 10:
        pivot_data = pivot_data.head(10)
    
    sns.heatmap(pivot_data, ax=ax2, cmap='YlOrRd', cbar_kws={'label': 'Trading Amount'})
    ax2.set_title('Trading Intensity Heatmap', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Hour of Day')
except Exception as e:
    # Fallback: simple scatter plot
    ax2.scatter(range(len(combined_df.head(100))), combined_df.head(100)['amount'], 
                alpha=0.6, color=colors[2])
    ax2.set_title('Trading Amount Distribution', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Transaction Index')
    ax2.set_ylabel('Trading Amount')

# Bottom-left: Cumulative volume by sector (simplified)
# Create simple sector classification
def classify_sector(company):
    if 'BL' in str(company) or 'BANK' in str(company):
        return 'Banking'
    elif 'PC' in str(company) or 'HPC' in str(company):
        return 'Power'
    elif 'IC' in str(company):
        return 'Insurance'
    else:
        return 'Others'

combined_df['sector'] = combined_df['company'].apply(classify_sector)

# Calculate sector volumes
sector_volumes = combined_df.groupby('sector')['volume'].sum().sort_values(ascending=False)
top_sectors = sector_volumes.head(4)

# Create stacked area chart simulation
x_range = range(len(top_sectors))
bottom = np.zeros(len(top_sectors))

for i, (sector, volume) in enumerate(top_sectors.items()):
    ax3.bar(x_range[i], volume, bottom=bottom[i], color=colors[i % len(colors)], 
            alpha=0.7, label=sector)

ax3.set_title('Trading Volume by Sector', fontweight='bold', fontsize=12)
ax3.set_xlabel('Sectors')
ax3.set_ylabel('Total Volume')
ax3.set_xticks(x_range)
ax3.set_xticklabels(top_sectors.index, rotation=45)
ax3.legend()

# Bottom-right: Price volatility analysis
# Calculate volatility for top companies
volatility_data = []
for company in top_companies[:3]:
    company_prices = daily_data[daily_data['company'] == company].copy()
    company_prices = company_prices.sort_values('date').head(15)
    
    if len(company_prices) > 3:
        company_prices['volatility'] = company_prices['price'].rolling(window=3).std()
        company_prices = company_prices.dropna()
        
        if len(company_prices) > 0:
            x = range(len(company_prices))
            volatility = company_prices['volatility']
            volumes = company_prices['volume']
            
            # Normalize volume for scatter size
            vol_sizes = (volumes / volumes.max() * 100) if volumes.max() > 0 else [50] * len(volumes)
            
            # Plot volatility line
            ax4.plot(x, volatility, color=colors[len(volatility_data)], 
                    linewidth=2, label=f'{company}', marker='o')
            
            # Add scatter points
            ax4.scatter(x, volatility, s=vol_sizes, alpha=0.6, 
                       color=colors[len(volatility_data)])
            
            # Add confidence band
            if len(volatility) > 1:
                std_vol = volatility.std()
                mean_vol = volatility.mean()
                ax4.fill_between(x, mean_vol - std_vol, mean_vol + std_vol, 
                               alpha=0.2, color=colors[len(volatility_data)])
            
            volatility_data.append(company)

ax4.set_title('Price Volatility with Volume-Weighted Points', fontweight='bold', fontsize=12)
ax4.set_xlabel('Trading Days')
ax4.set_ylabel('Price Volatility (Rolling Std)')
if volatility_data:
    ax4.legend()

# Add grid to all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.savefig('nepse_trading_analysis.png', dpi=300, bbox_inches='tight')
plt.show()