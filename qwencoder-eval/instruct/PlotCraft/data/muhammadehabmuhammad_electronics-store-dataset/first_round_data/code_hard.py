import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import os
import glob
from scipy import stats
warnings.filterwarnings('ignore')

# Load and combine all monthly data using glob to find actual files
csv_files = glob.glob('Sales_*.csv')

if not csv_files:
    # If no files found with glob, try direct file names
    file_names = [
        'Sales_January_2019.csv', 'Sales_February_2019.csv', 'Sales_March_2019.csv',
        'Sales_April_2019.csv', 'Sales_May_2019.csv', 'Sales_June_2019.csv',
        'Sales_July_2019.csv', 'Sales_August_2019.csv', 'Sales_September_2019.csv',
        'Sales_October_2019.csv', 'Sales_November_2019.csv', 'Sales_December_2019.csv'
    ]
    csv_files = [f for f in file_names if os.path.exists(f)]

# Combine all datasets
all_data = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        if not df_temp.empty:
            all_data.append(df_temp)
        print(f"Loaded {file}: {len(df_temp)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

if not all_data:
    raise ValueError("No data files could be loaded successfully")

df = pd.concat(all_data, ignore_index=True)
print(f"Total combined data: {len(df)} rows")

# Data preprocessing
df = df.dropna()
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna()

# Extract additional features
df['Month'] = df['Order Date'].dt.month
df['Hour'] = df['Order Date'].dt.hour
df['Revenue'] = df['Quantity Ordered'] * df['Price Each']
df['City'] = df['Purchase Address'].str.extract(r', ([^,]+), [A-Z]{2}')

# Define consistent color palette
colors = plt.cm.viridis(np.linspace(0, 1, 12))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 15), facecolor='white')
fig.patch.set_facecolor('white')

# Row 1: Monthly Sales Performance
# Subplot 1: Monthly total revenue with average order value
ax1 = plt.subplot(3, 4, 1, facecolor='white')
monthly_revenue = df.groupby('Month')['Revenue'].sum()
monthly_aov = df.groupby('Month')['Revenue'].mean()

bars = ax1.bar(monthly_revenue.index, monthly_revenue.values, color=[colors[i-1] for i in monthly_revenue.index], alpha=0.7, label='Total Revenue')
ax1_twin = ax1.twinx()
line = ax1_twin.plot(monthly_aov.index, monthly_aov.values, 'r-o', linewidth=2, label='Avg Order Value')

ax1.set_title('Monthly Revenue & Average Order Value', fontweight='bold', fontsize=11)
ax1.set_xlabel('Month')
ax1.set_ylabel('Total Revenue ($)', color='black')
ax1_twin.set_ylabel('Average Order Value ($)', color='red')
ax1.set_xticks(monthly_revenue.index)
ax1.set_xticklabels([month_names[i-1] for i in monthly_revenue.index], rotation=45)
ax1.grid(True, alpha=0.3)

# Subplot 2: Monthly quantity with unique orders
ax2 = plt.subplot(3, 4, 2, facecolor='white')
monthly_qty = df.groupby('Month')['Quantity Ordered'].sum()
monthly_orders = df.groupby('Month')['Order ID'].nunique()

bars = ax2.bar(monthly_qty.index, monthly_qty.values, color=[colors[i-1] for i in monthly_qty.index], alpha=0.7, label='Total Quantity')
ax2_twin = ax2.twinx()
line = ax2_twin.plot(monthly_orders.index, monthly_orders.values, 'g-s', linewidth=2, label='Unique Orders')

ax2.set_title('Monthly Quantity & Order Count', fontweight='bold', fontsize=11)
ax2.set_xlabel('Month')
ax2.set_ylabel('Total Quantity', color='black')
ax2_twin.set_ylabel('Unique Orders', color='green')
ax2.set_xticks(monthly_qty.index)
ax2.set_xticklabels([month_names[i-1] for i in monthly_qty.index], rotation=45)
ax2.grid(True, alpha=0.3)

# Subplot 3: Product diversity with top category percentage
ax3 = plt.subplot(3, 4, 3, facecolor='white')
monthly_diversity = df.groupby('Month')['Product'].nunique()
top_category_pct = df.groupby(['Month', 'Product']).size().groupby('Month').max() / df.groupby('Month').size() * 100

bars = ax3.bar(monthly_diversity.index, monthly_diversity.values, color=[colors[i-1] for i in monthly_diversity.index], alpha=0.7)
ax3_twin = ax3.twinx()
line = ax3_twin.plot(top_category_pct.index, top_category_pct.values, 'orange', marker='d', linewidth=2)

ax3.set_title('Product Diversity & Top Category %', fontweight='bold', fontsize=11)
ax3.set_xlabel('Month')
ax3.set_ylabel('Unique Products', color='black')
ax3_twin.set_ylabel('Top Category %', color='orange')
ax3.set_xticks(monthly_diversity.index)
ax3.set_xticklabels([month_names[i-1] for i in monthly_diversity.index], rotation=45)
ax3.grid(True, alpha=0.3)

# Subplot 4: Geographic distribution with city concentration
ax4 = plt.subplot(3, 4, 4, facecolor='white')
city_monthly = df.groupby(['Month', 'City']).size().unstack(fill_value=0)
city_concentration = df.groupby('Month')['City'].nunique()

# Create heatmap
if not city_monthly.empty:
    im = ax4.imshow(city_monthly.T.values, cmap='Blues', aspect='auto')
ax4_twin = ax4.twinx()
line = ax4_twin.plot(range(len(city_concentration)), city_concentration.values, 'red', marker='o', linewidth=2)

ax4.set_title('Geographic Distribution & City Count', fontweight='bold', fontsize=11)
ax4.set_xlabel('Month')
ax4.set_ylabel('Cities')
ax4_twin.set_ylabel('Unique Cities', color='red')
ax4.set_xticks(range(len(city_concentration)))
ax4.set_xticklabels([month_names[i-1] for i in city_concentration.index], rotation=45)

# Row 2: Product Category Evolution
# Subplot 5: Stacked area chart with diversity index
ax5 = plt.subplot(3, 4, 5, facecolor='white')
product_monthly = df.groupby(['Month', 'Product'])['Revenue'].sum().unstack(fill_value=0)
if not product_monthly.empty:
    top_products = product_monthly.sum().nlargest(5).index
    product_subset = product_monthly[top_products]
    
    ax5.stackplot(product_subset.index, *[product_subset[col] for col in product_subset.columns], 
                  colors=plt.cm.Set3(np.linspace(0, 1, len(top_products))), alpha=0.7)

ax5_twin = ax5.twinx()
diversity_index = df.groupby('Month')['Product'].nunique() / df.groupby('Month').size()
line = ax5_twin.plot(diversity_index.index, diversity_index.values, 'black', marker='*', linewidth=2)

ax5.set_title('Revenue by Top Products & Diversity Index', fontweight='bold', fontsize=11)
ax5.set_xlabel('Month')
ax5.set_ylabel('Revenue ($)')
ax5_twin.set_ylabel('Diversity Index', color='black')
ax5.set_xticks(diversity_index.index)
ax5.set_xticklabels([month_names[i-1] for i in diversity_index.index], rotation=45)

# Subplot 6: Price distribution violin plots with median trends
ax6 = plt.subplot(3, 4, 6, facecolor='white')
available_months = sorted(df['Month'].unique())
price_data = [df[df['Month'] == m]['Price Each'].values for m in available_months]
parts = ax6.violinplot(price_data, positions=available_months, showmeans=True, showmedians=True)

for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

monthly_median = df.groupby('Month')['Price Each'].median()
ax6.plot(monthly_median.index, monthly_median.values, 'red', marker='o', linewidth=2, label='Median Price')

ax6.set_title('Monthly Price Distribution & Median Trend', fontweight='bold', fontsize=11)
ax6.set_xlabel('Month')
ax6.set_ylabel('Price ($)')
ax6.set_xticks(available_months)
ax6.set_xticklabels([month_names[i-1] for i in available_months], rotation=45)
ax6.grid(True, alpha=0.3)

# Subplot 7: Product lifecycle analysis
ax7 = plt.subplot(3, 4, 7, facecolor='white')
monthly_products = df.groupby('Month')['Product'].nunique()
cumulative_products = monthly_products.cumsum()
new_products = monthly_products.values
returning_products = np.maximum(0, cumulative_products.values - new_products)

width = 0.35
x = np.array(monthly_products.index)
bars1 = ax7.bar(x - width/2, new_products, width, label='New Products', color='lightgreen', alpha=0.7)
bars2 = ax7.bar(x + width/2, returning_products, width, label='Returning Products', color='lightcoral', alpha=0.7)

turnover_rate = new_products / (new_products + returning_products + 1) * 100
ax7_twin = ax7.twinx()
line = ax7_twin.plot(x, turnover_rate, 'purple', marker='s', linewidth=2, label='Turnover Rate')

ax7.set_title('Product Lifecycle & Turnover Rate', fontweight='bold', fontsize=11)
ax7.set_xlabel('Month')
ax7.set_ylabel('Product Count')
ax7_twin.set_ylabel('Turnover Rate (%)', color='purple')
ax7.set_xticks(x)
ax7.set_xticklabels([month_names[i-1] for i in x], rotation=45)
ax7.legend(loc='upper left')
ax7.grid(True, alpha=0.3)

# Subplot 8: Seasonal decomposition simulation
ax8 = plt.subplot(3, 4, 8, facecolor='white')
monthly_sales = df.groupby('Month')['Revenue'].sum()
x_vals = np.array(monthly_sales.index)
trend = np.polyval(np.polyfit(x_vals, monthly_sales.values, 2), x_vals)
seasonal = monthly_sales.values - trend
residual = np.random.normal(0, np.std(seasonal) * 0.1, len(x_vals))

ax8.plot(x_vals, monthly_sales.values, 'b-o', linewidth=2, label='Actual Sales', markersize=4)
ax8.plot(x_vals, trend, 'r--', linewidth=2, label='Trend')
ax8.fill_between(x_vals, trend, trend + seasonal, alpha=0.3, color='green', label='Seasonal')

ax8.set_title('Sales Decomposition Analysis', fontweight='bold', fontsize=11)
ax8.set_xlabel('Month')
ax8.set_ylabel('Revenue ($)')
ax8.set_xticks(x_vals)
ax8.set_xticklabels([month_names[i-1] for i in x_vals], rotation=45)
ax8.legend()
ax8.grid(True, alpha=0.3)

# Row 3: Customer Behavior Patterns
# Subplot 9: Order frequency distribution with KDE
ax9 = plt.subplot(3, 4, 9, facecolor='white')
order_freq = df.groupby('Month').size()
ax9.hist(order_freq.values, bins=8, alpha=0.7, color='skyblue', density=True, label='Order Frequency')

# Add KDE curve
kde = stats.gaussian_kde(order_freq.values)
x_range = np.linspace(order_freq.min(), order_freq.max(), 100)
ax9.plot(x_range, kde(x_range), 'red', linewidth=2, label='KDE')

# Add mean and median lines
mean_val = order_freq.mean()
median_val = order_freq.median()
ax9.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
ax9.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')

ax9.set_title('Monthly Order Frequency Distribution', fontweight='bold', fontsize=11)
ax9.set_xlabel('Orders per Month')
ax9.set_ylabel('Density')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Subplot 10: Geographic sales bubble plot
ax10 = plt.subplot(3, 4, 10, facecolor='white')
city_sales = df.groupby(['Month', 'City'])['Revenue'].sum().reset_index()
for month in sorted(df['Month'].unique()):
    month_data = city_sales[city_sales['Month'] == month]
    if not month_data.empty:
        y_pos = np.random.normal(month, 0.1, len(month_data))  # Add some jitter
        ax10.scatter([month] * len(month_data), y_pos, 
                    s=month_data['Revenue']/1000, alpha=0.6, 
                    color=colors[month-1], label=f'Month {month}' if month <= 3 else "")

# Add trend line
monthly_geo_sales = df.groupby('Month')['Revenue'].sum()
z = np.polyfit(monthly_geo_sales.index, monthly_geo_sales.values, 1)
p = np.poly1d(z)
trend_y = p(monthly_geo_sales.index)
normalized_trend = (trend_y - trend_y.min()) / (trend_y.max() - trend_y.min()) * 10 + 1
ax10.plot(monthly_geo_sales.index, normalized_trend, 'red', linewidth=2, label='Trend')

ax10.set_title('Geographic Sales Distribution', fontweight='bold', fontsize=11)
ax10.set_xlabel('Month')
ax10.set_ylabel('Geographic Index')
ax10.set_xticks(sorted(df['Month'].unique()))
ax10.set_xticklabels([month_names[i-1] for i in sorted(df['Month'].unique())], rotation=45)
ax10.grid(True, alpha=0.3)

# Subplot 11: Hourly distribution heatmap with peak trends
ax11 = plt.subplot(3, 4, 11, facecolor='white')
hourly_monthly = df.groupby(['Month', 'Hour']).size().unstack(fill_value=0)
if not hourly_monthly.empty:
    im = ax11.imshow(hourly_monthly.values, cmap='YlOrRd', aspect='auto')
    
    # Add peak hour trend
    peak_hours = hourly_monthly.idxmax(axis=1)
    ax11_twin = ax11.twinx()
    line = ax11_twin.plot(range(len(peak_hours)), peak_hours.values, 'blue', marker='o', linewidth=2, label='Peak Hour')
    
    ax11.set_title('Hourly Purchase Patterns & Peak Trends', fontweight='bold', fontsize=11)
    ax11.set_xlabel('Month')
    ax11.set_ylabel('Hour of Day')
    ax11_twin.set_ylabel('Peak Hour', color='blue')
    ax11.set_xticks(range(len(hourly_monthly)))
    ax11.set_xticklabels([month_names[i-1] for i in hourly_monthly.index], rotation=45)
    ax11.set_yticks(range(0, 24, 4))

# Subplot 12: Correlation matrix with trend arrows
ax12 = plt.subplot(3, 4, 12, facecolor='white')
metrics_df = pd.DataFrame({
    'Revenue': df.groupby('Month')['Revenue'].sum(),
    'Quantity': df.groupby('Month')['Quantity Ordered'].sum(),
    'Diversity': df.groupby('Month')['Product'].nunique(),
    'Cities': df.groupby('Month')['City'].nunique()
})

corr_matrix = metrics_df.corr()
im = ax12.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)

# Add correlation values
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax12.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontweight='bold')

# Add trend arrows (simplified)
for i in range(len(corr_matrix)-1):
    ax12.annotate('', xy=(i+0.3, i+0.7), xytext=(i+0.7, i+0.3),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax12.set_title('Monthly Metrics Correlation Matrix', fontweight='bold', fontsize=11)
ax12.set_xticks(range(len(corr_matrix.columns)))
ax12.set_yticks(range(len(corr_matrix.columns)))
ax12.set_xticklabels(corr_matrix.columns, rotation=45)
ax12.set_yticklabels(corr_matrix.columns)

# Add colorbar
cbar = plt.colorbar(im, ax=ax12, shrink=0.8)
cbar.set_label('Correlation Coefficient')

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.suptitle('2019 Electronics Store Sales: Comprehensive Temporal Evolution Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('electronics_sales_analysis.png', dpi=300, bbox_inches='tight')
plt.show()