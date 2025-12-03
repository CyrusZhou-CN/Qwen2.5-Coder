import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Load and combine all datasets with proper error handling
datasets = ['NEPSE184.csv', 'NEPSE174.csv', 'NEPSE156.csv', 'NEPSE360.csv', 
           'NEPSE517.csv', 'NEPSE131.csv', 'NEPSE186.csv', 'NEPSE396.csv', 
           'NEPSE398.csv', 'NEPSE171.csv']

all_data = []

# Try to load each dataset individually
for dataset in datasets:
    if os.path.exists(dataset):
        try:
            df = pd.read_csv(dataset)
            # Use the actual column names from the data
            if len(df.columns) >= 9:
                df.columns = ['id', 'transaction_id', 'company', 'buyer_id', 'seller_id', 
                             'quantity', 'rate', 'amount', 'timestamp']
                all_data.append(df)
                print(f"Successfully loaded {dataset} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue

# If no files loaded successfully, create sample data for demonstration
if len(all_data) == 0:
    print("No CSV files found. Creating sample data for demonstration...")
    
    # Create sample data that mimics the Nepal stock exchange data
    np.random.seed(42)
    companies = ['NABIL', 'NBL', 'SANIMA', 'NLBBL', 'IGI', 'PRIN', 'BBC', 'AHPC', 'SBBLJ', 'ODBL']
    
    sample_data = []
    start_date = datetime(2007, 1, 1)
    
    for i in range(50000):  # Generate 50k sample transactions
        days_offset = np.random.randint(0, 4748)  # ~13 years of days
        current_date = start_date + pd.Timedelta(days=days_offset)
        
        company = np.random.choice(companies)
        base_rate = np.random.uniform(100, 2000)
        quantity = np.random.randint(10, 1000)
        amount = base_rate * quantity
        
        sample_data.append({
            'id': i + 1,
            'transaction_id': f"2013{i:08d}",
            'company': company,
            'buyer_id': np.random.randint(1, 100),
            'seller_id': np.random.randint(1, 100),
            'quantity': quantity,
            'rate': base_rate,
            'amount': amount,
            'timestamp': current_date.strftime('%Y-%m-%d %H:%M:%S.%f')
        })
    
    combined_df = pd.DataFrame(sample_data)
else:
    # Combine all loaded datasets
    combined_df = pd.concat(all_data, ignore_index=True)

# Clean and process timestamp data
def parse_timestamp(ts):
    try:
        if isinstance(ts, str) and '-' in ts:
            return pd.to_datetime(ts)
        else:
            # Handle numeric timestamps
            ts_str = str(ts)
            if len(ts_str) >= 8:
                year = int(ts_str[:4]) if ts_str[:4].isdigit() and int(ts_str[:4]) >= 2007 else 2013
                month = int(ts_str[4:6]) if len(ts_str) > 4 and ts_str[4:6].isdigit() and 1 <= int(ts_str[4:6]) <= 12 else 1
                day = int(ts_str[6:8]) if len(ts_str) > 6 and ts_str[6:8].isdigit() and 1 <= int(ts_str[6:8]) <= 31 else 1
                return datetime(year, month, day)
            else:
                return datetime(2013, 1, 1)
    except:
        return datetime(2013, 1, 1)

combined_df['date'] = combined_df['timestamp'].apply(parse_timestamp)
combined_df = combined_df[(combined_df['date'].dt.year >= 2007) & (combined_df['date'].dt.year <= 2019)]

# Ensure we have valid data
if len(combined_df) == 0:
    print("No valid data found. Exiting...")
    exit()

# Create company categories for market concentration analysis
bank_companies = ['NABIL', 'NBL', 'SANIMA', 'NLBBL']
insurance_companies = ['IGI', 'PRIN']
manufacturing_companies = ['BBC', 'AHPC']
other_companies = ['SBBLJ', 'ODBL']

def categorize_company(company):
    if company in bank_companies:
        return 'Banking'
    elif company in insurance_companies:
        return 'Insurance'
    elif company in manufacturing_companies:
        return 'Manufacturing'
    else:
        return 'Others'

combined_df['category'] = combined_df['company'].apply(categorize_company)

# Create the 2x2 subplot dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color scheme
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F7B0F', '#FF6B35', '#004E89']

# Top-left: Dual-axis plot - Daily trading rate vs Monthly volume
monthly_data = combined_df.groupby([combined_df['date'].dt.to_period('M')]).agg({
    'rate': 'mean',
    'amount': 'sum',
    'quantity': 'sum'
}).reset_index()

if len(monthly_data) > 0:
    monthly_data['date'] = monthly_data['date'].dt.to_timestamp()
    
    # Line chart for average daily trading rate
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(monthly_data['date'], monthly_data['rate'], color=colors[0], linewidth=2.5, label='Avg Trading Rate')
    ax1.set_ylabel('Average Trading Rate (NPR)', fontweight='bold', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    
    # Bar chart for monthly trading volume
    bars = ax1_twin.bar(monthly_data['date'], monthly_data['amount']/1000000, alpha=0.6, color=colors[1], width=20, label='Monthly Volume (Millions NPR)')
    ax1_twin.set_ylabel('Monthly Trading Volume (Million NPR)', fontweight='bold', color=colors[1])
    ax1_twin.tick_params(axis='y', labelcolor=colors[1])

ax1.set_title('Nepal Stock Market: Trading Rate vs Volume Evolution', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Year', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Top-right: Multi-company comparison with volatility background
top_companies = combined_df['company'].value_counts().head(5).index.tolist()
monthly_volatility = combined_df.groupby([combined_df['date'].dt.to_period('M')])['rate'].std().reset_index()

if len(monthly_volatility) > 0:
    monthly_volatility['date'] = monthly_volatility['date'].dt.to_timestamp()
    monthly_volatility['rate'] = monthly_volatility['rate'].fillna(0)
    
    # Background area chart for market volatility
    ax2.fill_between(monthly_volatility['date'], monthly_volatility['rate'], alpha=0.2, color='gray', label='Market Volatility')
    
    # Line plots for top 5 companies
    for i, company in enumerate(top_companies):
        company_data = combined_df[combined_df['company'] == company].groupby(
            combined_df[combined_df['company'] == company]['date'].dt.to_period('M')
        )['rate'].mean().reset_index()
        
        if len(company_data) > 0:
            company_data['date'] = company_data['date'].dt.to_timestamp()
            ax2.plot(company_data['date'], company_data['rate'], color=colors[i % len(colors)], linewidth=2, 
                     marker='o', markersize=4, label=f'{company}')

ax2.set_title('Top 5 Companies Performance with Market Volatility', fontweight='bold', fontsize=14, pad=20)
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Average Trading Rate (NPR)', fontweight='bold')
ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)

# Bottom-left: Seasonal heatmap with trend lines
pivot_data = combined_df.groupby([combined_df['date'].dt.year, combined_df['date'].dt.month])['rate'].mean().unstack()
pivot_data = pivot_data.fillna(0)

if not pivot_data.empty:
    # Create heatmap
    im = ax3.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax3.set_yticks(range(len(pivot_data.index)))
    ax3.set_yticklabels(pivot_data.index)
    
    # Add trend lines for each year
    for i, year in enumerate(pivot_data.index):
        year_data = pivot_data.loc[year].values
        valid_months = np.where(year_data > 0)[0]
        if len(valid_months) > 1:
            ax3.plot(valid_months, [i] * len(valid_months), 'w-', linewidth=2, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Average Trading Rate (NPR)', fontweight='bold')

ax3.set_title('Seasonal Trading Patterns by Year', fontweight='bold', fontsize=14, pad=20)
ax3.set_xlabel('Month', fontweight='bold')
ax3.set_ylabel('Year', fontweight='bold')

# Bottom-right: Market concentration with large transactions
category_monthly = combined_df.groupby([combined_df['date'].dt.to_period('M'), 'category'])['amount'].sum().unstack(fill_value=0)

if not category_monthly.empty:
    category_monthly.index = category_monthly.index.to_timestamp()
    
    # Calculate percentages
    category_pct = category_monthly.div(category_monthly.sum(axis=1), axis=0) * 100
    
    # Ensure all categories exist
    for cat in ['Banking', 'Insurance', 'Manufacturing', 'Others']:
        if cat not in category_pct.columns:
            category_pct[cat] = 0
    
    ax4.stackplot(category_pct.index, 
                  category_pct.get('Banking', 0), 
                  category_pct.get('Insurance', 0), 
                  category_pct.get('Manufacturing', 0), 
                  category_pct.get('Others', 0),
                  labels=['Banking', 'Insurance', 'Manufacturing', 'Others'],
                  colors=colors[:4], alpha=0.8)
    
    # Scatter plot for large transactions
    large_transactions = combined_df[combined_df['amount'] > 100000]
    if not large_transactions.empty:
        scatter_dates = large_transactions['date']
        scatter_amounts = (large_transactions['amount'] / large_transactions['amount'].max()) * 50  # Normalize for visibility
        ax4.scatter(scatter_dates, scatter_amounts, color='red', alpha=0.6, s=30, 
                   label='Large Transactions (>100K)', zorder=5)

ax4.set_title('Market Concentration by Sector with Large Transactions', fontweight='bold', fontsize=14, pad=20)
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Market Share (%)', fontweight='bold')
ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3)

# Add annotations for significant periods (if we have enough data)
if len(monthly_data) > 12:
    mid_date = monthly_data['date'].iloc[len(monthly_data)//2]
    max_rate = monthly_data['rate'].max()
    ax1.annotate('Market Activity Peak', xy=(mid_date, max_rate), 
                xytext=(mid_date, max_rate * 1.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5), 
                fontweight='bold', color='red')

# Synchronize time axes where possible
if len(monthly_data) > 0:
    start_date = monthly_data['date'].min()
    end_date = monthly_data['date'].max()
    for ax in [ax1, ax2, ax4]:
        ax.set_xlim(start_date, end_date)

# Final layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('nepal_stock_market_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()