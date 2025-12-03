import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('defi_dataset.csv')

# Data preprocessing
# Get date columns (exclude metadata columns)
date_columns = [col for col in df.columns if col not in ['Name', 'Category', 'Chain', 'Type', 'Token']]

# Convert date columns to datetime for proper sorting
date_columns_sorted = sorted(date_columns, key=lambda x: datetime.strptime(x, '%d/%m/%Y'))

# Fill NaN values with 0 for TVL calculations
df_clean = df.copy()
df_clean[date_columns] = df_clean[date_columns].fillna(0)

# Create figure with 2x2 subplots and white background
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Sample dates for better performance and readability
sample_step = 15  # Every 15 days instead of 30 for smoother curves
sampled_dates = date_columns_sorted[::sample_step]
dates_for_plot = [datetime.strptime(date, '%d/%m/%Y') for date in sampled_dates]

# Top-left: Stacked area chart of top 5 DeFi categories
category_tvl = df_clean.groupby('Category')[date_columns_sorted].sum()
# Filter out categories with very low TVL to avoid scaling issues
category_totals = category_tvl.sum(axis=1)
valid_categories = category_totals[category_totals > 1e6]  # Only categories with > 1M total TVL
top_5_categories = valid_categories.nlargest(5).index

category_data = []
for category in top_5_categories:
    values = category_tvl.loc[category, sampled_dates].values / 1e9  # Convert to billions
    # Ensure no negative values
    values = np.maximum(values, 0)
    category_data.append(values)

# Create stacked area chart with professional colors
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
ax1.stackplot(dates_for_plot, *category_data, labels=top_5_categories, 
              colors=colors, alpha=0.8)
ax1.set_title('TVL Evolution by Top 5 DeFi Categories', fontsize=13, pad=15)
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('TVL (Billions USD)', fontweight='bold')
ax1.legend(loc='upper left', frameon=False)
ax1.grid(True, color='lightgray', linestyle='--', alpha=0.7, linewidth=0.5)
ax1.set_facecolor('white')

# Top-right: Monthly growth rate with rolling average
# Use monthly sampling for growth rate calculation
monthly_step = 30
monthly_dates = date_columns_sorted[::monthly_step]
total_tvl = df_clean[date_columns_sorted].sum()
monthly_tvl = total_tvl[monthly_dates]

# Calculate monthly growth rate with proper bounds
growth_rates = []
for i in range(1, len(monthly_tvl)):
    prev_val = monthly_tvl.iloc[i-1]
    curr_val = monthly_tvl.iloc[i]
    if prev_val > 1e6:  # Avoid division by very small numbers
        growth_rate = ((curr_val - prev_val) / prev_val) * 100
        # Cap extreme values
        growth_rate = np.clip(growth_rate, -100, 500)
        growth_rates.append(growth_rate)
    else:
        growth_rates.append(0)

growth_dates = [datetime.strptime(date, '%d/%m/%Y') for date in monthly_dates[1:]]

# Calculate 6-month rolling average
rolling_avg = pd.Series(growth_rates).rolling(window=6, min_periods=1).mean()

ax2.plot(growth_dates, growth_rates, 'o-', color='#E74C3C', linewidth=2, 
         markersize=4, label='Monthly Growth Rate', alpha=0.8)
ax2.plot(growth_dates, rolling_avg, '-', color='#2C3E50', linewidth=3, 
         label='6-Month Rolling Average')
ax2.set_title('Monthly TVL Growth Rate with Trend', fontsize=13, pad=15)
ax2.set_xlabel('Date', fontweight='bold')
ax2.set_ylabel('Growth Rate (%)', fontweight='bold')
ax2.legend(frameon=False)
ax2.grid(True, color='lightgray', linestyle='--', alpha=0.7, linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax2.set_facecolor('white')

# Bottom-left: Top 3 individual protocols comparison
# Find top 3 protocols by peak TVL, ensuring they are different
protocol_peak_tvl = df_clean.set_index('Name')[date_columns].max(axis=1)
# Filter out protocols with very low peak TVL
valid_protocols = protocol_peak_tvl[protocol_peak_tvl > 1e8]  # > 100M peak TVL
top_3_protocols = valid_protocols.nlargest(3).index

colors_protocols = ['#E74C3C', '#3498DB', '#2ECC71']
line_styles = ['-', '--', '-.']

for i, protocol in enumerate(top_3_protocols):
    protocol_data = df_clean[df_clean['Name'] == protocol][sampled_dates].iloc[0].values / 1e9
    # Ensure no negative values
    protocol_data = np.maximum(protocol_data, 0)
    ax3.plot(dates_for_plot, protocol_data, 
             color=colors_protocols[i], linestyle=line_styles[i], 
             linewidth=3, label=protocol, marker='o', markersize=3, alpha=0.8)

ax3.set_title('Top 3 DeFi Protocols TVL Comparison', fontsize=13, pad=15)
ax3.set_xlabel('Date', fontweight='bold')
ax3.set_ylabel('TVL (Billions USD)', fontweight='bold')
ax3.legend(frameon=False)
ax3.grid(True, color='lightgray', linestyle='--', alpha=0.7, linewidth=0.5)
ax3.set_facecolor('white')

# Bottom-right: Quarterly protocol count vs average TVL
# Create quarterly data
quarterly_step = 90
quarterly_dates = date_columns_sorted[::quarterly_step]
quarterly_datetime = [datetime.strptime(date, '%d/%m/%Y') for date in quarterly_dates]

# Count active protocols per quarter (TVL > threshold)
active_protocols = []
avg_tvl_per_protocol = []

for date in quarterly_dates:
    # Count protocols with meaningful TVL (> $10,000)
    meaningful_tvl = df_clean[date] > 10000
    active_count = meaningful_tvl.sum()
    total_tvl = df_clean[meaningful_tvl][date].sum()
    avg_tvl = total_tvl / active_count if active_count > 0 else 0
    
    active_protocols.append(active_count)
    avg_tvl_per_protocol.append(avg_tvl / 1e6)  # Convert to millions

# Create combination chart
ax4_twin = ax4.twinx()

# Bar chart for protocol count
bars = ax4.bar(quarterly_datetime, active_protocols, alpha=0.7, color='#95A5A6', 
               label='Active Protocols', width=25)

# Line chart for average TVL
line = ax4_twin.plot(quarterly_datetime, avg_tvl_per_protocol, 'o-', 
                     color='#E67E22', linewidth=3, markersize=6, 
                     label='Avg TVL per Protocol')

ax4.set_title('Protocol Count vs Average TVL per Protocol', fontsize=13, pad=15)
ax4.set_xlabel('Date', fontweight='bold')
ax4.set_ylabel('Number of Active Protocols', fontweight='bold', color='#95A5A6')
ax4_twin.set_ylabel('Average TVL per Protocol\n(Millions USD)', fontweight='bold', 
                    color='#E67E22', rotation=270, labelpad=20)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)

ax4.grid(True, color='lightgray', linestyle='--', alpha=0.7, linewidth=0.5)
ax4.set_facecolor('white')
ax4_twin.set_facecolor('white')

# Main title for entire figure - make it prominent
fig.suptitle('DeFi Protocol Evolution Analysis (2018-2022)', 
             fontsize=20, fontweight='bold', y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.subplots_adjust(hspace=0.35, wspace=0.3)

plt.show()