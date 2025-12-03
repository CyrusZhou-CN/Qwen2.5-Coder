import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data with optimizations
print("Loading data...")
df = pd.read_csv('Real_Estate_Sales_2001-2020_GL.csv')

# Sample data for performance (use 10% of data)
df = df.sample(frac=0.1, random_state=42).copy()

# Convert date and create time-based features
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
df = df.dropna(subset=['Date Recorded'])
df['Year'] = df['Date Recorded'].dt.year
df['Month'] = df['Date Recorded'].dt.month
df['YearMonth'] = df['Date Recorded'].dt.to_period('M')

# Clean data
df = df.dropna(subset=['Sale Amount', 'Assessed Value', 'Sales Ratio'])
df = df[(df['Sale Amount'] > 0) & (df['Assessed Value'] > 0)]
df = df[(df['Sales Ratio'] > 0) & (df['Sales Ratio'] < 5)]  # Remove outliers

print("Creating visualization...")

# Create figure
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: Average assessed value trends with high-value transactions
ax1 = plt.subplot(3, 3, 1)
monthly_stats = df.groupby('YearMonth').agg({
    'Assessed Value': ['mean', 'std'],
    'Sale Amount': 'count'
}).reset_index()
monthly_stats.columns = ['YearMonth', 'avg_assessed', 'std_assessed', 'count']
monthly_stats = monthly_stats.dropna()

if len(monthly_stats) > 0:
    monthly_stats['YearMonth_dt'] = monthly_stats['YearMonth'].dt.to_timestamp()
    
    # Line chart with error bars (simplified)
    ax1.plot(monthly_stats['YearMonth_dt'], monthly_stats['avg_assessed'], 
             color='#2E86AB', linewidth=2, label='Avg Assessed Value')
    
    # Overlay scatter plot for high-value transactions
    high_value = df[df['Sale Amount'] > 500000].sample(n=min(500, len(df[df['Sale Amount'] > 500000])))
    if len(high_value) > 0:
        ax1.scatter(high_value['Date Recorded'], high_value['Assessed Value'], 
                   color='#F24236', alpha=0.6, s=15, label='High-Value (>$500K)')

ax1.set_title('Market Value Evolution', fontweight='bold', fontsize=10)
ax1.set_xlabel('Year')
ax1.set_ylabel('Assessed Value ($)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Transaction volumes and sales ratios
ax2 = plt.subplot(3, 3, 2)
monthly_vol = df.groupby('YearMonth').agg({
    'Sale Amount': 'count',
    'Sales Ratio': 'mean'
}).reset_index()

if len(monthly_vol) > 0:
    monthly_vol['YearMonth_dt'] = monthly_vol['YearMonth'].dt.to_timestamp()
    monthly_vol['rolling_ratio'] = monthly_vol['Sales Ratio'].rolling(window=3, min_periods=1).mean()
    
    # Bar chart for volumes
    ax2.bar(monthly_vol['YearMonth_dt'], monthly_vol['Sale Amount'], 
            color='#A23B72', alpha=0.6, width=20, label='Monthly Transactions')
    
    # Secondary axis for rolling average
    ax2_twin = ax2.twinx()
    ax2_twin.plot(monthly_vol['YearMonth_dt'], monthly_vol['rolling_ratio'], 
                  color='#F18F01', linewidth=2, label='3-Month Avg Sales Ratio')

ax2.set_title('Transaction Volume vs Sales Ratio', fontweight='bold', fontsize=10)
ax2.set_xlabel('Year')
ax2.set_ylabel('Transaction Count', color='#A23B72')
if 'ax2_twin' in locals():
    ax2_twin.set_ylabel('Sales Ratio', color='#F18F01')

# Row 1, Subplot 3: Cumulative distribution and median trend
ax3 = plt.subplot(3, 3, 3)
monthly_sales = df.groupby('YearMonth').agg({
    'Sale Amount': ['sum', 'median']
}).reset_index()
monthly_sales.columns = ['YearMonth', 'total_sales', 'median_sales']

if len(monthly_sales) > 0:
    monthly_sales['YearMonth_dt'] = monthly_sales['YearMonth'].dt.to_timestamp()
    monthly_sales['cumulative_sales'] = monthly_sales['total_sales'].cumsum()
    
    # Area chart for cumulative sales
    ax3.fill_between(monthly_sales['YearMonth_dt'], monthly_sales['cumulative_sales'], 
                     color='#C73E1D', alpha=0.4, label='Cumulative Sales')
    
    # Secondary axis for median trend
    ax3_twin = ax3.twinx()
    ax3_twin.plot(monthly_sales['YearMonth_dt'], monthly_sales['median_sales'], 
                  color='#2E86AB', linewidth=2, label='Median Sale Price')

ax3.set_title('Cumulative Sales & Median Price', fontweight='bold', fontsize=10)
ax3.set_xlabel('Year')
ax3.set_ylabel('Cumulative Sales ($)', color='#C73E1D')
if 'ax3_twin' in locals():
    ax3_twin.set_ylabel('Median Price ($)', color='#2E86AB')

# Row 2, Subplot 4: Property type composition
ax4 = plt.subplot(3, 3, 4)
prop_monthly = df.groupby(['YearMonth', 'Property Type']).size().reset_index(name='count')

if len(prop_monthly) > 0:
    prop_monthly['YearMonth_dt'] = prop_monthly['YearMonth'].dt.to_timestamp()
    prop_pivot = prop_monthly.pivot(index='YearMonth_dt', columns='Property Type', values='count').fillna(0)
    
    if len(prop_pivot.columns) > 0:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax4.stackplot(prop_pivot.index, *[prop_pivot[col] for col in prop_pivot.columns], 
                      labels=prop_pivot.columns, colors=colors[:len(prop_pivot.columns)], alpha=0.7)

ax4.set_title('Property Type Composition', fontweight='bold', fontsize=10)
ax4.set_xlabel('Year')
ax4.set_ylabel('Transaction Count')
ax4.legend(fontsize=8)

# Row 2, Subplot 5: Sales ratio distribution by property type
ax5 = plt.subplot(3, 3, 5)
prop_types = df['Property Type'].value_counts().head(3).index
box_data = []
labels = []

for prop_type in prop_types:
    data = df[df['Property Type'] == prop_type]['Sales Ratio']
    if len(data) > 10:  # Minimum data points
        box_data.append(data.values)
        labels.append(prop_type[:10])  # Truncate long names

if len(box_data) > 0:
    bp = ax5.boxplot(box_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)

ax5.set_title('Sales Ratio by Property Type', fontweight='bold', fontsize=10)
ax5.set_xlabel('Property Type')
ax5.set_ylabel('Sales Ratio')
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Correlation heatmap
ax6 = plt.subplot(3, 3, 6)
monthly_corr = df.groupby('Month').apply(
    lambda x: x['Assessed Value'].corr(x['Sale Amount']) if len(x) > 1 else 0
).reset_index()
monthly_corr.columns = ['Month', 'Correlation']

# Create heatmap data
heatmap_data = np.zeros((12, 1))
for _, row in monthly_corr.iterrows():
    if not np.isnan(row['Correlation']):
        heatmap_data[int(row['Month'])-1, 0] = row['Correlation']

im = ax6.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
ax6.set_yticks(range(12))
ax6.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax6.set_xticks([0])
ax6.set_xticklabels(['Correlation'])
ax6.set_title('Monthly Correlation: Assessed vs Sale', fontweight='bold', fontsize=10)

# Row 3, Subplot 7: Time series decomposition
ax7 = plt.subplot(3, 3, 7)
monthly_volume = df.groupby('YearMonth').size().reset_index(name='volume')

if len(monthly_volume) > 0:
    monthly_volume['YearMonth_dt'] = monthly_volume['YearMonth'].dt.to_timestamp()
    monthly_volume['trend'] = monthly_volume['volume'].rolling(window=6, center=True, min_periods=1).mean()
    
    ax7.plot(monthly_volume['YearMonth_dt'], monthly_volume['volume'], 
             color='#2E86AB', linewidth=1, alpha=0.7, label='Original')
    ax7.plot(monthly_volume['YearMonth_dt'], monthly_volume['trend'], 
             color='#F18F01', linewidth=2, label='Trend')

ax7.set_title('Sales Volume Decomposition', fontweight='bold', fontsize=10)
ax7.set_xlabel('Year')
ax7.set_ylabel('Transaction Volume')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Row 3, Subplot 8: Bubble chart
ax8 = plt.subplot(3, 3, 8)
bubble_data = df.groupby('YearMonth').agg({
    'Years until sold': 'mean',
    'Sale Amount': 'count',
    'Sales Ratio': 'mean'
}).reset_index()

if len(bubble_data) > 0:
    bubble_data['YearMonth_dt'] = bubble_data['YearMonth'].dt.to_timestamp()
    bubble_data = bubble_data.dropna()
    
    if len(bubble_data) > 0:
        scatter = ax8.scatter(bubble_data['YearMonth_dt'], bubble_data['Years until sold'], 
                             s=bubble_data['Sale Amount']*2, c=bubble_data['Sales Ratio'], 
                             cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)

ax8.set_title('Market Timing Analysis', fontweight='bold', fontsize=10)
ax8.set_xlabel('Year')
ax8.set_ylabel('Avg Years Until Sold')

# Row 3, Subplot 9: Top towns analysis
ax9 = plt.subplot(3, 3, 9)
top_towns = df['Town'].value_counts().head(5).index
town_monthly = df[df['Town'].isin(top_towns)].groupby(['YearMonth', 'Town']).size().reset_index(name='count')

if len(town_monthly) > 0:
    town_monthly['YearMonth_dt'] = town_monthly['YearMonth'].dt.to_timestamp()
    
    # Background area for total activity
    total_monthly = df.groupby('YearMonth').size().reset_index(name='total')
    total_monthly['YearMonth_dt'] = total_monthly['YearMonth'].dt.to_timestamp()
    ax9.fill_between(total_monthly['YearMonth_dt'], total_monthly['total'], 
                     color='gray', alpha=0.2, label='Total Market')
    
    # Lines for top towns
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50']
    for i, town in enumerate(top_towns):
        town_data = town_monthly[town_monthly['Town'] == town]
        if len(town_data) > 0:
            ax9.plot(town_data['YearMonth_dt'], town_data['count'], 
                     color=colors[i], linewidth=2, label=town[:10], marker='o', markersize=2)

ax9.set_title('Top 5 Towns: Sales Trends', fontweight='bold', fontsize=10)
ax9.set_xlabel('Year')
ax9.set_ylabel('Monthly Transactions')
ax9.legend(fontsize=8, loc='upper left')
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

print("Visualization complete!")
plt.savefig('real_estate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()