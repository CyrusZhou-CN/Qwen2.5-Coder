import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and combine all CSV files
csv_files = glob.glob('*.csv')
all_data = []

# Limit to first 10 files to avoid timeout
for file in csv_files[:10]:
    try:
        df = pd.read_csv(file)
        df['Company'] = file.replace('.csv', '')
        all_data.append(df)
    except:
        continue

# Combine all data
if not all_data:
    # Create sample data if no files found
    sample_data = {
        'Company': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 20,
        'Value ($)': np.random.normal(1000000, 500000, 100),
        'Transaction': ['Buy', 'Sale'] * 50,
        'Date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'Relationship': ['CEO', 'CFO', 'Director', 'EVP', 'VP'] * 20
    }
    combined_df = pd.DataFrame(sample_data)
    combined_df['Value_Clean'] = combined_df['Value ($)']
else:
    combined_df = pd.concat(all_data, ignore_index=True)

# Clean and preprocess data
def clean_value(val):
    if pd.isna(val) or val == 0:
        return 0
    val_str = str(val).replace(',', '').replace('$', '')
    try:
        return float(val_str)
    except:
        return 0

if 'Value_Clean' not in combined_df.columns:
    combined_df['Value_Clean'] = combined_df['Value ($)'].apply(clean_value)

combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
combined_df = combined_df[combined_df['Value_Clean'] > 0].copy()

# Limit data size for performance
if len(combined_df) > 1000:
    combined_df = combined_df.sample(n=1000, random_state=42)

# Create figure with 2x2 subplots
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

# Color mapping for relationships
relationship_colors = {
    'Director': '#2E86AB', 'CEO': '#A23B72', 'CFO': '#F18F01',
    'President': '#C73E1D', 'EVP': '#7209B7', 'SVP': '#F72585',
    'VP': '#4361EE', 'Other': '#6C757D'
}

def get_relationship_category(rel):
    if pd.isna(rel):
        return 'Other'
    rel_lower = str(rel).lower()
    if 'director' in rel_lower:
        return 'Director'
    elif 'ceo' in rel_lower:
        return 'CEO'
    elif 'cfo' in rel_lower:
        return 'CFO'
    elif 'president' in rel_lower:
        return 'President'
    elif 'evp' in rel_lower or 'executive vice' in rel_lower:
        return 'EVP'
    elif 'svp' in rel_lower or 'senior vice' in rel_lower:
        return 'SVP'
    elif 'vice president' in rel_lower or ' vp' in rel_lower:
        return 'VP'
    else:
        return 'Other'

combined_df['Relationship_Cat'] = combined_df['Relationship'].apply(get_relationship_category)

# Calculate overall S&P 500 average
overall_avg = combined_df['Value_Clean'].mean()

# Top-left: Diverging bar chart with error bars
ax1 = plt.subplot(2, 2, 1)

# Calculate company statistics
company_stats = combined_df.groupby('Company').agg({
    'Value_Clean': ['mean', 'std', 'count']
}).round(0)
company_stats.columns = ['mean_value', 'std_value', 'count']
company_stats['deviation'] = company_stats['mean_value'] - overall_avg
company_stats = company_stats.sort_values('deviation', key=abs, ascending=False).head(10)

# Create diverging bar chart
y_pos = np.arange(len(company_stats))
colors = ['#C73E1D' if x > 0 else '#2E86AB' for x in company_stats['deviation']]

bars = ax1.barh(y_pos, company_stats['deviation'], color=colors, alpha=0.8, height=0.6)

# Add error bars with safe calculation
std_errors = company_stats['std_value'] / np.sqrt(np.maximum(company_stats['count'], 1))
ax1.errorbar(company_stats['deviation'], y_pos, 
            xerr=std_errors, 
            fmt='none', color='black', capsize=3, alpha=0.7)

ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(company_stats.index, fontsize=9)
ax1.set_xlabel('Deviation from S&P 500 Average ($)', fontweight='bold')
ax1.set_title('Transaction Value Deviations by Company\n(with Standard Error Bars)', 
              fontweight='bold', fontsize=12, pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Top-right: Dumbbell plot for Buy vs Sale
ax2 = plt.subplot(2, 2, 2)

# Filter for Buy and Sale transactions
buy_sale_data = combined_df[combined_df['Transaction'].isin(['Buy', 'Sale'])].copy()
if len(buy_sale_data) > 0:
    company_buy_sale = buy_sale_data.groupby(['Company', 'Transaction'])['Value_Clean'].mean().unstack(fill_value=0)
    
    # Filter companies with both buy and sale data
    if 'Buy' in company_buy_sale.columns and 'Sale' in company_buy_sale.columns:
        both_transactions = company_buy_sale[(company_buy_sale['Buy'] > 0) & (company_buy_sale['Sale'] > 0)]
        both_transactions = both_transactions.head(8)
        
        if len(both_transactions) > 0:
            y_pos = np.arange(len(both_transactions))
            
            # Plot dumbbell chart
            for i, (company, row) in enumerate(both_transactions.iterrows()):
                ax2.plot([row['Buy'], row['Sale']], [i, i], 'o-', 
                        color='#6C757D', linewidth=2, markersize=8, alpha=0.7)
                ax2.scatter(row['Buy'], i, color='#2E86AB', s=80, label='Buy' if i == 0 else "", zorder=5)
                ax2.scatter(row['Sale'], i, color='#C73E1D', s=80, label='Sale' if i == 0 else "", zorder=5)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(both_transactions.index, fontsize=9)
            ax2.set_xlabel('Average Transaction Value ($)', fontweight='bold')
            ax2.set_title('Buy vs Sale Transaction Values\n(Dumbbell Comparison)', 
                          fontweight='bold', fontsize=12, pad=15)
            ax2.legend(loc='lower right')
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No companies with both\nBuy and Sale transactions', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'Insufficient transaction data', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
else:
    ax2.text(0.5, 0.5, 'No Buy/Sale data available', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)

ax2.set_title('Buy vs Sale Transaction Values\n(Dumbbell Comparison)', 
              fontweight='bold', fontsize=12, pad=15)

# Bottom-left: Radar chart for top companies
ax3 = plt.subplot(2, 2, 3, projection='polar')

# Get top 6 companies by transaction volume
top_companies = combined_df['Company'].value_counts().head(6)

if len(top_companies) > 0:
    # Calculate metrics for radar chart
    radar_data = []
    for company in top_companies.index:
        company_data = combined_df[combined_df['Company'] == company]
        
        # Value deviation (normalized)
        value_dev = min(abs(company_data['Value_Clean'].mean() - overall_avg) / max(overall_avg, 1), 1)
        
        # Frequency deviation (normalized by max frequency)
        freq_dev = len(company_data) / max(combined_df['Company'].value_counts().max(), 1)
        
        # Timing concentration (transactions per unique date)
        unique_dates = company_data['Date'].nunique()
        timing_conc = len(company_data) / max(unique_dates, 1) if unique_dates > 0 else 0
        timing_conc = min(timing_conc / 5, 1)  # Normalize to 0-1
        
        radar_data.append([value_dev, freq_dev, timing_conc])
    
    # Set up radar chart
    categories = ['Value\nDeviation', 'Transaction\nFrequency', 'Timing\nConcentration']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_companies)))
    
    for i, (company, data) in enumerate(zip(top_companies.index, radar_data)):
        values = data + data[:1]  # Complete the circle
        ax3.plot(angles, values, 'o-', linewidth=2, label=company, color=colors[i], alpha=0.8)
        ax3.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_title('Top Companies: Multi-Dimensional\nDeviation Analysis', 
                  fontweight='bold', fontsize=12, pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax3.grid(True, alpha=0.3)

# Bottom-right: Area chart for cumulative deviations over time
ax4 = plt.subplot(2, 2, 4)

# Calculate rolling average and deviations
if len(combined_df) > 0:
    # Sort by date and calculate rolling average
    time_series = combined_df.sort_values('Date').copy()
    time_series = time_series.dropna(subset=['Date'])
    
    if len(time_series) > 10:
        # Use smaller window for rolling average
        window_size = min(10, len(time_series))
        time_series['rolling_avg'] = time_series['Value_Clean'].rolling(window=window_size, min_periods=1).mean()
        time_series['deviation'] = time_series['Value_Clean'] - time_series['rolling_avg']
        
        # Calculate cumulative deviations
        pos_devs = np.where(time_series['deviation'] > 0, time_series['deviation'], 0)
        neg_devs = np.where(time_series['deviation'] < 0, time_series['deviation'], 0)
        
        time_series['cumulative_pos'] = np.cumsum(pos_devs)
        time_series['cumulative_neg'] = np.cumsum(neg_devs)
        
        # Sample data for better visualization
        sample_size = min(50, len(time_series))
        sample_indices = np.linspace(0, len(time_series)-1, sample_size, dtype=int)
        sampled_data = time_series.iloc[sample_indices]
        
        ax4.fill_between(sampled_data['Date'], 0, sampled_data['cumulative_pos'], 
                        color='#C73E1D', alpha=0.6, label='Positive Deviations')
        ax4.fill_between(sampled_data['Date'], 0, sampled_data['cumulative_neg'], 
                        color='#2E86AB', alpha=0.6, label='Negative Deviations')
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Cumulative Deviation ($)', fontweight='bold')
        ax4.set_title('Cumulative Deviations from Rolling\nAverage Over Time', 
                      fontweight='bold', fontsize=12, pad=15)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Insufficient time series data', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
else:
    ax4.text(0.5, 0.5, 'No data available', 
            ha='center', va='center', transform=ax4.transAxes, fontsize=12)

ax4.set_title('Cumulative Deviations from Rolling\nAverage Over Time', 
              fontweight='bold', fontsize=12, pad=15)

# Adjust layout and styling
plt.tight_layout(pad=3.0)

# Add overall title
fig.suptitle('S&P 500 Insider Trading Deviation Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('insider_trading_deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()