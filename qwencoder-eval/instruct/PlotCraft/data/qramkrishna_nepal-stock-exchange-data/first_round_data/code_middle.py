import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load and combine all datasets with proper error handling
file_names = ['NEPSE131.csv', 'NEPSE132.csv', 'NEPSE133.csv', 'NEPSE134.csv', 
              'NEPSE135.csv', 'NEPSE136.csv', 'NEPSE137.csv', 'NEPSE138.csv', 
              'NEPSE139.csv', 'NEPSE140.csv']

# Load and combine all data
all_data = []
for file in file_names:
    try:
        # Read CSV without specifying column names first to see the structure
        df_temp = pd.read_csv(file)
        if not df_temp.empty:
            # Rename columns to standard names based on the data structure shown
            df_temp.columns = ['row_id', 'transaction_id', 'symbol', 'bid_price', 'ask_price', 
                              'volume', 'rate', 'amount', 'timestamp_str']
            all_data.append(df_temp)
            print(f"Successfully loaded {file} with {len(df_temp)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

# Check if we have any data
if not all_data:
    print("No data files could be loaded. Creating synthetic data for demonstration.")
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_records = 10000
    
    symbols = ['NABIL', 'SCB', 'EBL', 'NBB', 'NIB', 'HBL', 'SBI', 'MBL']
    
    df = pd.DataFrame({
        'row_id': range(1, n_records + 1),
        'transaction_id': np.random.randint(126000, 127000, n_records),
        'symbol': np.random.choice(symbols, n_records),
        'bid_price': np.random.randint(10, 50, n_records),
        'ask_price': np.random.randint(10, 50, n_records),
        'volume': np.random.randint(10, 1000, n_records),
        'rate': np.random.randint(200, 3000, n_records),
        'amount': np.random.uniform(10000, 500000, n_records),
        'timestamp_str': ['synthetic'] * n_records
    })
else:
    # Combine all dataframes
    df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset has {len(df)} total records")

# Data preprocessing
df = df.dropna()
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
df = df.dropna()

# Create time-based data
df = df.sort_values('transaction_id')
base_date = datetime(2020, 1, 1)
df['date'] = pd.date_range(start=base_date, periods=len(df), freq='1H')
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['day_name'] = df['date'].dt.day_name()

# Filter for major companies (ensure they exist in data)
available_symbols = df['symbol'].unique()
major_companies = [sym for sym in ['NABIL', 'SCB', 'EBL', 'NBB'] if sym in available_symbols]
if len(major_companies) < 2:
    major_companies = list(available_symbols[:4])  # Take first 4 available

df_major = df[df['symbol'].isin(major_companies)].copy()

# Create broker categories
np.random.seed(42)
df['broker_category'] = np.random.choice(['Institutional', 'Retail', 'Foreign', 'Corporate'], 
                                       size=len(df), p=[0.3, 0.4, 0.15, 0.15])

# Set up the figure
plt.style.use('default')
fig = plt.figure(figsize=(16, 12), facecolor='white')

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#F2CC8F']
company_colors = dict(zip(major_companies, colors[:len(major_companies)]))

# Top-left: Dual-axis plot with line chart (prices) and bar chart (volumes)
ax1 = plt.subplot(2, 2, 1)

# Aggregate daily data for major companies
daily_data = df_major.groupby([df_major['date'].dt.date, 'symbol']).agg({
    'rate': 'mean',
    'volume': 'sum',
    'amount': 'sum'
}).reset_index()

# Sample data for better visualization
sample_size = min(50, len(daily_data))
daily_sample = daily_data.sample(n=sample_size).sort_values('date')

# Plot average prices (line chart)
for company in major_companies:
    company_data = daily_sample[daily_sample['symbol'] == company]
    if not company_data.empty:
        ax1.plot(range(len(company_data)), company_data['rate'], 
                color=company_colors[company], linewidth=2, label=f'{company} Price', 
                marker='o', markersize=4, alpha=0.8)

ax1.set_ylabel('Average Stock Price', fontweight='bold', color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.grid(True, alpha=0.3)

# Create second y-axis for volume bars
ax1_twin = ax1.twinx()

# Aggregate volume data for bar chart
volume_data = daily_sample.groupby('symbol')['volume'].mean()
x_pos = np.arange(len(volume_data))
bars = ax1_twin.bar(x_pos, volume_data.values, alpha=0.6, 
                   color=[company_colors[sym] for sym in volume_data.index])

ax1_twin.set_ylabel('Average Trading Volume', fontweight='bold', color='#A23B72')
ax1_twin.tick_params(axis='y', labelcolor='#A23B72')

ax1.set_title('Stock Prices vs Trading Volumes', fontweight='bold', fontsize=12, pad=15)
ax1.legend(loc='upper left', fontsize=8)
ax1.set_xlabel('Time Period', fontweight='bold')

# Top-right: Stacked area chart with line overlay
ax2 = plt.subplot(2, 2, 2)

# Aggregate by broker category over time
time_periods = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')[:30]
broker_amounts = []

for period in time_periods:
    period_data = df[df['date'].dt.date == period.date()]
    if not period_data.empty:
        broker_summary = period_data.groupby('broker_category')['amount'].sum()
        broker_amounts.append(broker_summary)

if broker_amounts:
    broker_df = pd.DataFrame(broker_amounts, index=time_periods[:len(broker_amounts)])
    broker_df = broker_df.fillna(0)
    
    # Create stacked area chart
    ax2.stackplot(range(len(broker_df)), 
                  broker_df.get('Institutional', 0), 
                  broker_df.get('Retail', 0),
                  broker_df.get('Foreign', 0), 
                  broker_df.get('Corporate', 0),
                  labels=['Institutional', 'Retail', 'Foreign', 'Corporate'],
                  colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], alpha=0.7)

# Overlay line plot for transaction count
daily_transactions = df.groupby(df['date'].dt.date).size()
sample_transactions = daily_transactions.head(30)

ax2_twin = ax2.twinx()
ax2_twin.plot(range(len(sample_transactions)), sample_transactions.values, 
              color='black', linewidth=2, label='Transaction Count', marker='o', markersize=3)

ax2.set_ylabel('Trading Amount', fontweight='bold')
ax2_twin.set_ylabel('Transaction Count', fontweight='bold')
ax2.set_title('Trading Amounts by Broker Category', fontweight='bold', fontsize=12, pad=15)
ax2.legend(loc='upper left', fontsize=8)
ax2_twin.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time Period', fontweight='bold')

# Bottom-left: Violin and box plots with scatter overlay
ax3 = plt.subplot(2, 2, 3)

# Prepare data for distribution analysis by year
years = sorted(df['year'].unique())
rate_by_year = []
year_labels = []

for year in years:
    year_data = df[df['year'] == year]['rate'].values
    if len(year_data) > 10:  # Only include years with sufficient data
        rate_by_year.append(year_data)
        year_labels.append(str(year))

if rate_by_year:
    # Create violin plots
    parts = ax3.violinplot(rate_by_year, positions=range(len(year_labels)), 
                          widths=0.6, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.6)

    # Overlay box plots
    bp = ax3.boxplot(rate_by_year, positions=range(len(year_labels)), 
                    widths=0.3, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#A23B72')
        patch.set_alpha(0.8)

    # Add scatter points for high-volume transactions
    high_volume_threshold = df['volume'].quantile(0.9)
    for i, year in enumerate([int(y) for y in year_labels]):
        year_high_vol = df[(df['year'] == year) & (df['volume'] > high_volume_threshold)]
        if not year_high_vol.empty:
            sample_points = year_high_vol['rate'].sample(min(20, len(year_high_vol)))
            x_scatter = np.full(len(sample_points), i) + np.random.normal(0, 0.05, len(sample_points))
            ax3.scatter(x_scatter, sample_points, alpha=0.6, s=20, color='#F18F01', 
                       label='High Volume' if i == 0 else "")

    ax3.set_xticks(range(len(year_labels)))
    ax3.set_xticklabels(year_labels)
    ax3.set_ylabel('Stock Rate Distribution', fontweight='bold')
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.set_title('Stock Rate Distribution by Year', fontweight='bold', fontsize=12, pad=15)
    ax3.grid(True, alpha=0.3)
    if len(df[df['volume'] > high_volume_threshold]) > 0:
        ax3.legend(fontsize=8)

# Bottom-right: Heatmap with trading intensity
ax4 = plt.subplot(2, 2, 4)

# Create trading intensity matrix (hour vs day of week)
intensity_data = df.groupby(['hour', 'day_of_week']).size().reset_index(name='count')
intensity_pivot = intensity_data.pivot(index='hour', columns='day_of_week', values='count')
intensity_pivot = intensity_pivot.fillna(0)

# Create heatmap
im = ax4.imshow(intensity_pivot.values, cmap='YlOrRd', aspect='auto', origin='lower')

# Set labels
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax4.set_xticks(range(len(intensity_pivot.columns)))
ax4.set_xticklabels([day_names[int(col)] for col in intensity_pivot.columns])
ax4.set_yticks(range(0, len(intensity_pivot.index), 4))
ax4.set_yticklabels(range(0, 24, 4))
ax4.set_ylabel('Hour of Day', fontweight='bold')
ax4.set_xlabel('Day of Week', fontweight='bold')
ax4.set_title('Trading Intensity Heatmap', fontweight='bold', fontsize=12, pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
cbar.set_label('Transaction Frequency', fontweight='bold')

# Overall layout adjustments
plt.tight_layout(pad=2.0)

# Add overall title
fig.suptitle('Nepal Stock Exchange (NEPSE) - Comprehensive Trading Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('nepse_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()