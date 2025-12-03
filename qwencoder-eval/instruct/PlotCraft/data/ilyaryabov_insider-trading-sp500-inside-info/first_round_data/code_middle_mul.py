import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Get list of available CSV files in the current directory
available_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"Available files: {available_files}")

# Load and combine all available datasets
all_data = []
for file in available_files:
    try:
        df = pd.read_csv(file)
        if len(df) > 0:  # Only add non-empty dataframes
            df['Company'] = file.replace('.csv', '')
            all_data.append(df)
            print(f"Successfully loaded {file} with {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

# Check if we have any data
if len(all_data) == 0:
    print("No data files could be loaded. Creating sample visualization...")
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = []
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 
                 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'PFE', 'KO', 'ABBV', 'PEP', 'WMT']
    
    for i, company in enumerate(companies):
        n_transactions = np.random.randint(5, 50)
        for j in range(n_transactions):
            sample_data.append({
                'Company': company,
                'Insider Trading': f'Insider_{j%5}',
                'Relationship': np.random.choice(['CEO', 'CFO', 'VP Sales', 'Director', 'EVP']),
                'Date': pd.Timestamp('2022-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
                'Transaction': np.random.choice(['Buy', 'Sale', 'Option Exercise'], p=[0.2, 0.6, 0.2]),
                'Cost': np.random.uniform(50, 500),
                'Shares': np.random.randint(100, 50000),
                'Value ($)': np.random.uniform(10000, 5000000),
                'Shares Total': np.random.randint(1000, 100000),
                'SEC Form 4': 'Sample'
            })
    
    combined_df = pd.DataFrame(sample_data)
else:
    # Combine all loaded data
    combined_df = pd.concat(all_data, ignore_index=True)

print(f"Combined dataset shape: {combined_df.shape}")

# Data preprocessing
def clean_numeric_column(col):
    if col.dtype == 'object':
        # Handle string values with commas and dollar signs
        cleaned = col.astype(str).str.replace(',', '').str.replace('$', '').str.replace(' ', '')
        return pd.to_numeric(cleaned, errors='coerce')
    return col

# Clean numeric columns
combined_df['Shares'] = clean_numeric_column(combined_df['Shares'])
combined_df['Value ($)'] = clean_numeric_column(combined_df['Value ($)'])
combined_df['Shares Total'] = clean_numeric_column(combined_df['Shares Total'])

# Ensure Date is datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

# Remove rows with missing critical data
initial_rows = len(combined_df)
combined_df = combined_df.dropna(subset=['Cost', 'Shares', 'Value ($)', 'Date'])
print(f"Removed {initial_rows - len(combined_df)} rows with missing data")

# Ensure we have enough data
if len(combined_df) == 0:
    raise ValueError("No valid data remaining after cleaning")

# Create seniority categories
def categorize_seniority(relationship):
    if pd.isna(relationship):
        return 'Other'
    relationship = str(relationship).upper()
    if any(title in relationship for title in ['CEO', 'CFO', 'COO', 'CHAIRMAN', 'PRESIDENT']):
        return 'C-level'
    elif any(title in relationship for title in ['VP', 'VICE']):
        return 'VP-level'
    elif 'DIRECTOR' in relationship:
        return 'Director'
    else:
        return 'Other'

combined_df['Seniority'] = combined_df['Relationship'].apply(categorize_seniority)

# Calculate company-level metrics
company_metrics = combined_df.groupby('Company').agg({
    'Value ($)': ['sum', 'mean', 'count'],
    'Insider Trading': 'nunique',
    'Relationship': 'nunique',
    'Transaction': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Sale',
    'Shares': 'sum'
}).round(2)

company_metrics.columns = ['Total_Value', 'Avg_Transaction_Value', 'Num_Transactions', 
                          'Unique_Insiders', 'Relationship_Diversity', 'Dominant_Transaction', 
                          'Total_Shares']

# Calculate concentration ratio (top 3 insiders)
concentration_ratios = []
for company in combined_df['Company'].unique():
    company_data = combined_df[combined_df['Company'] == company]
    insider_values = company_data.groupby('Insider Trading')['Value ($)'].sum().sort_values(ascending=False)
    if len(insider_values) >= 3:
        top3_ratio = insider_values.head(3).sum() / insider_values.sum() * 100
    elif len(insider_values) > 0:
        top3_ratio = 100  # If less than 3 insiders, top insiders represent 100%
    else:
        top3_ratio = 0
    concentration_ratios.append(top3_ratio)

company_metrics['Concentration_Ratio'] = concentration_ratios

# Calculate average days between transactions per company
avg_days_between = []
for company in combined_df['Company'].unique():
    company_data = combined_df[combined_df['Company'] == company].sort_values('Date')
    if len(company_data) > 1:
        date_diffs = company_data['Date'].diff().dt.days.dropna()
        avg_days = date_diffs.mean() if len(date_diffs) > 0 else 0
    else:
        avg_days = 0
    avg_days_between.append(max(0, avg_days))  # Ensure non-negative

company_metrics['Avg_Days_Between'] = avg_days_between

# Fill any remaining NaN values
company_metrics = company_metrics.fillna(0)

print(f"Company metrics calculated for {len(company_metrics)} companies")

# Create the 2x2 subplot visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Color palette
colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']

# Top-left: Scatter plot with marginal histograms
transaction_type_colors = {
    'Buy': colors[0],
    'Sale': colors[1], 
    'Option Exercise': colors[2]
}

scatter_colors = [transaction_type_colors.get(dt, colors[3]) for dt in company_metrics['Dominant_Transaction']]

scatter = ax1.scatter(company_metrics['Total_Value'], company_metrics['Unique_Insiders'], 
                     c=scatter_colors, alpha=0.7, s=100, edgecolors='white', linewidth=1.5)

ax1.set_xlabel('Total Transaction Value ($)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Number of Unique Insiders', fontweight='bold', fontsize=11)
ax1.set_title('Transaction Value vs Unique Insiders\nby Dominant Transaction Type', 
              fontweight='bold', pad=15, fontsize=12)
ax1.grid(True, alpha=0.3)

# Format x-axis for better readability
ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Add legend for transaction types
legend_elements = [plt.scatter([], [], c=colors[0], s=80, label='Buy', edgecolors='white'),
                  plt.scatter([], [], c=colors[1], s=80, label='Sale', edgecolors='white'),
                  plt.scatter([], [], c=colors[2], s=80, label='Option Exercise', edgecolors='white')]
ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

# Top-right: Correlation heatmap
correlation_vars = ['Avg_Transaction_Value', 'Total_Shares', 'Num_Transactions', 'Relationship_Diversity']
corr_matrix = company_metrics[correlation_vars].corr()

# Create heatmap
im = ax2.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add text annotations
for i in range(len(correlation_vars)):
    for j in range(len(correlation_vars)):
        text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

ax2.set_xticks(range(len(correlation_vars)))
ax2.set_yticks(range(len(correlation_vars)))
ax2.set_xticklabels([var.replace('_', '\n') for var in correlation_vars], rotation=45, ha='right')
ax2.set_yticklabels([var.replace('_', '\n') for var in correlation_vars])
ax2.set_title('Correlation Matrix:\nKey Trading Variables', fontweight='bold', pad=15, fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Bottom-left: Bubble plot
max_shares = company_metrics['Total_Shares'].max()
if max_shares > 0:
    bubble_sizes = (company_metrics['Total_Shares'] / max_shares * 400) + 50
else:
    bubble_sizes = [100] * len(company_metrics)

bubble = ax3.scatter(company_metrics['Avg_Days_Between'], company_metrics['Concentration_Ratio'],
                    s=bubble_sizes, c=colors[3], alpha=0.6, edgecolors='white', linewidth=1.5)

# Add trend line if we have valid data
valid_mask = ~(np.isnan(company_metrics['Avg_Days_Between']) | np.isnan(company_metrics['Concentration_Ratio']))
valid_data = company_metrics[valid_mask]

if len(valid_data) > 1 and valid_data['Avg_Days_Between'].std() > 0:
    z = np.polyfit(valid_data['Avg_Days_Between'], valid_data['Concentration_Ratio'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(valid_data['Avg_Days_Between'].min(), 
                         valid_data['Avg_Days_Between'].max(), 100)
    ax3.plot(x_trend, p(x_trend), color=colors[4], linewidth=3, linestyle='--', alpha=0.8)

ax3.set_xlabel('Average Days Between Transactions', fontweight='bold', fontsize=11)
ax3.set_ylabel('Concentration Ratio (%)', fontweight='bold', fontsize=11)
ax3.set_title('Transaction Timing vs Concentration\n(Bubble Size = Total Volume)', 
              fontweight='bold', pad=15, fontsize=12)
ax3.grid(True, alpha=0.3)

# Bottom-right: Seniority analysis
seniority_data = combined_df.groupby(['Company', 'Seniority']).agg({
    'Date': lambda x: (x.max() - x.min()).days if len(x) > 1 else 0,
    'Value ($)': 'mean'
}).reset_index()

seniority_data = seniority_data.dropna()

seniority_colors = {
    'C-level': colors[0],
    'VP-level': colors[1], 
    'Director': colors[2],
    'Other': colors[3]
}

for seniority in ['C-level', 'VP-level', 'Director', 'Other']:
    data = seniority_data[seniority_data['Seniority'] == seniority]
    if len(data) > 0:
        ax4.scatter(data['Date'], data['Value ($)'], 
                   c=seniority_colors[seniority], label=seniority, 
                   alpha=0.7, s=80, edgecolors='white', linewidth=1.5)
        
        # Add trend line for each seniority level if enough data points
        if len(data) > 2:
            valid_data_seniority = data.dropna()
            if len(valid_data_seniority) > 1 and valid_data_seniority['Date'].std() > 0:
                z = np.polyfit(valid_data_seniority['Date'], valid_data_seniority['Value ($)'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data_seniority['Date'].min(), 
                                    valid_data_seniority['Date'].max(), 100)
                ax4.plot(x_trend, p(x_trend), color=seniority_colors[seniority], 
                        linewidth=2, alpha=0.7, linestyle='-')

ax4.set_xlabel('Days Between First and Last Transaction', fontweight='bold', fontsize=11)
ax4.set_ylabel('Average Transaction Value ($)', fontweight='bold', fontsize=11)
ax4.set_title('Seniority Level vs Transaction Patterns', fontweight='bold', pad=15, fontsize=12)
ax4.legend(loc='upper right', framealpha=0.9)
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Overall layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Add overall title
fig.suptitle('Comprehensive Insider Trading Correlation Analysis Across S&P 500 Companies', 
             fontsize=16, fontweight='bold', y=0.98)

# Save the plot
plt.savefig('insider_trading_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()