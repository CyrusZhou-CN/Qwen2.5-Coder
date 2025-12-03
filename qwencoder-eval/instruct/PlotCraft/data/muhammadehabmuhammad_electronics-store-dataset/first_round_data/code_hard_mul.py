import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load and combine all monthly data
file_names = [
    'Sales_January_2019.csv', 'Sales_February_2019.csv', 'Sales_March_2019.csv',
    'Sales_April_2019.csv', 'Sales_May_2019.csv', 'Sales_June_2019.csv',
    'Sales_July_2019.csv', 'Sales_August_2019.csv', 'Sales_September_2019.csv',
    'Sales_October_2019.csv', 'Sales_November_2019.csv', 'Sales_December_2019.csv'
]

month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

all_data = []

for file in file_names:
    try:
        df = pd.read_csv(file)
        # Extract month from filename
        month_name = file.replace('Sales_', '').replace('_2019.csv', '')
        if month_name in month_mapping:
            df['Month'] = month_name
            df['Month_Num'] = month_mapping[month_name]
            all_data.append(df)
    except FileNotFoundError:
        continue
    except Exception as e:
        continue

# Combine all available data
if not all_data:
    print("No data files found. Creating sample data for demonstration.")
    # Create sample data if files not found
    np.random.seed(42)
    sample_data = []
    for month_num in range(1, 13):
        month_name = list(month_mapping.keys())[month_num-1]
        n_records = np.random.randint(5000, 15000)
        df_sample = pd.DataFrame({
            'Order ID': range(n_records),
            'Product': np.random.choice(['iPhone', 'Macbook Pro Laptop', 'Wired Headphones', 
                                       'USB-C Charging Cable', 'AA Batteries (4-pack)', 
                                       '27in FHD Monitor', 'Apple Airpods Headphones'], n_records),
            'Quantity Ordered': np.random.randint(1, 5, n_records),
            'Price Each': np.random.choice([11.99, 14.95, 99.99, 149.99, 600, 700, 1700], n_records),
            'Order Date': pd.date_range(f'2019-{month_num:02d}-01', 
                                      periods=n_records, freq='H')[:n_records],
            'Purchase Address': np.random.choice([
                '123 Main St, Los Angeles, CA 90001',
                '456 Oak St, New York City, NY 10001',
                '789 Pine St, San Francisco, CA 94016',
                '321 Elm St, Seattle, WA 98101',
                '654 Maple St, Boston, MA 02215'
            ], n_records),
            'Month': month_name,
            'Month_Num': month_num
        })
        sample_data.append(df_sample)
    all_data = sample_data

df_combined = pd.concat(all_data, ignore_index=True)

# Data preprocessing
df_combined = df_combined.dropna()
df_combined['Quantity Ordered'] = pd.to_numeric(df_combined['Quantity Ordered'], errors='coerce')
df_combined['Price Each'] = pd.to_numeric(df_combined['Price Each'], errors='coerce')
df_combined['Order Date'] = pd.to_datetime(df_combined['Order Date'], errors='coerce')
df_combined = df_combined.dropna()

# Calculate revenue
df_combined['Revenue'] = df_combined['Quantity Ordered'] * df_combined['Price Each']

# Extract city from address
df_combined['City'] = df_combined['Purchase Address'].str.extract(r', ([^,]+), [A-Z]{2}')
df_combined['City'] = df_combined['City'].fillna('Unknown')

# Categorize products
def categorize_product(product):
    if pd.isna(product):
        return 'Other'
    product_lower = str(product).lower()
    if any(x in product_lower for x in ['iphone', 'google phone', 'vareebadd phone']):
        return 'Phones'
    elif any(x in product_lower for x in ['laptop', 'macbook']):
        return 'Laptops'
    elif any(x in product_lower for x in ['monitor', 'tv']):
        return 'Monitors/TVs'
    elif any(x in product_lower for x in ['headphones', 'airpods']):
        return 'Audio'
    elif any(x in product_lower for x in ['cable', 'charging']):
        return 'Accessories'
    elif any(x in product_lower for x in ['batteries']):
        return 'Batteries'
    else:
        return 'Other'

df_combined['Category'] = df_combined['Product'].apply(categorize_product)

# Get available months
available_months = sorted(df_combined['Month_Num'].unique())
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the comprehensive 3x2 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color schemes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']
colors_secondary = ['#F4A261', '#E76F51', '#2A9D8F', '#264653', '#E9C46A', '#F4A261']

# Subplot 1: Monthly Revenue Trends with Order Counts
ax1 = plt.subplot(3, 2, 1)
monthly_stats = df_combined.groupby('Month_Num').agg({
    'Revenue': ['sum', 'std', 'count'],
    'Order ID': 'count'
}).round(2)

monthly_revenue = monthly_stats['Revenue']['sum']
monthly_std = monthly_stats['Revenue']['std'].fillna(0)
monthly_orders = monthly_stats['Order ID']['count']

# Line chart for revenue
ax1_twin = ax1.twinx()
line1 = ax1.plot(available_months, monthly_revenue.loc[available_months], color=colors_main[0], 
                linewidth=3, marker='o', markersize=8, label='Monthly Revenue')
ax1.errorbar(available_months, monthly_revenue.loc[available_months], 
            yerr=monthly_std.loc[available_months], 
            color=colors_main[0], alpha=0.3, capsize=5)

# Bar chart for order counts
bars = ax1_twin.bar(available_months, monthly_orders.loc[available_months], alpha=0.6, 
                   color=colors_secondary[1], label='Order Count', width=0.6)

ax1.set_xlabel('Month', fontweight='bold', fontsize=12)
ax1.set_ylabel('Revenue ($)', fontweight='bold', fontsize=12, color=colors_main[0])
ax1_twin.set_ylabel('Order Count', fontweight='bold', fontsize=12, color=colors_secondary[1])
ax1.set_title('Monthly Revenue Trends with Order Volume', fontweight='bold', fontsize=14, pad=20)
ax1.set_xticks(available_months)
ax1.set_xticklabels([month_names[i-1] for i in available_months], rotation=45)
ax1.grid(True, alpha=0.3)

# Add holiday annotations if December data exists
if 12 in available_months:
    dec_revenue = monthly_revenue.loc[12]
    ax1.annotate('Holiday Season', xy=(12, dec_revenue), xytext=(10, dec_revenue*1.2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7), fontsize=10, color='red')

# Subplot 2: Product Category Performance Over Time
ax2 = plt.subplot(3, 2, 2)
category_monthly = df_combined.groupby(['Month_Num', 'Category'])['Revenue'].sum().unstack(fill_value=0)
diversity_monthly = df_combined.groupby('Month_Num')['Product'].nunique()

# Stacked area chart
if not category_monthly.empty:
    category_data = []
    for cat in category_monthly.columns:
        cat_values = []
        for month in available_months:
            if month in category_monthly.index:
                cat_values.append(category_monthly.loc[month, cat])
            else:
                cat_values.append(0)
        category_data.append(cat_values)
    
    ax2.stackplot(available_months, *category_data, 
                 labels=category_monthly.columns, 
                 colors=colors_main[:len(category_monthly.columns)], alpha=0.8)

# Secondary axis for diversity
ax2_twin = ax2.twinx()
diversity_values = [diversity_monthly.loc[month] if month in diversity_monthly.index else 0 
                   for month in available_months]
line2 = ax2_twin.plot(available_months, diversity_values, color='black', linewidth=2, 
                     marker='s', markersize=6, label='Product Diversity')

ax2.set_xlabel('Month', fontweight='bold', fontsize=12)
ax2.set_ylabel('Revenue by Category ($)', fontweight='bold', fontsize=12)
ax2_twin.set_ylabel('Unique Products Sold', fontweight='bold', fontsize=12)
ax2.set_title('Product Category Performance & Diversity Evolution', fontweight='bold', fontsize=14, pad=20)
ax2.set_xticks(available_months)
ax2.set_xticklabels([month_names[i-1] for i in available_months], rotation=45)
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=9)
ax2_twin.legend(loc='upper right')

# Subplot 3: Seasonal Sales Patterns (Polar Chart)
ax3 = plt.subplot(3, 2, 3, projection='polar')
monthly_avg_daily = df_combined.groupby('Month_Num')['Revenue'].sum() / df_combined.groupby('Month_Num')['Order Date'].apply(lambda x: x.dt.day.nunique())
monthly_peak = df_combined.groupby('Month_Num')['Revenue'].max()

# Create theta values for available months
theta_all = np.linspace(0, 2*np.pi, 12, endpoint=False)
theta_available = [theta_all[i-1] for i in available_months]

# Get values for available months
avg_daily_values = [monthly_avg_daily.loc[month] if month in monthly_avg_daily.index else 0 
                   for month in available_months]
peak_values = [monthly_peak.loc[month] if month in monthly_peak.index else 0 
              for month in available_months]

# Polar plot
ax3.plot(theta_available, avg_daily_values, color=colors_main[2], linewidth=3, marker='o', markersize=8)
ax3.fill(theta_available, avg_daily_values, color=colors_main[2], alpha=0.3)

# Scatter for peak days
ax3.scatter(theta_available, peak_values, color=colors_secondary[0], s=100, alpha=0.8, label='Peak Sales Days')

ax3.set_theta_zero_location('N')
ax3.set_theta_direction(-1)
ax3.set_thetagrids(np.degrees(theta_available), [month_names[i-1] for i in available_months])
ax3.set_title('Seasonal Sales Patterns\n(Average Daily Revenue)', fontweight='bold', fontsize=14, pad=30)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 4: Geographic Sales Evolution (Top 5 Cities)
ax4 = plt.subplot(3, 2, 4)
top_cities = df_combined.groupby('City')['Revenue'].sum().nlargest(5).index
city_monthly = df_combined[df_combined['City'].isin(top_cities)].groupby(['Month_Num', 'City'])['Revenue'].sum().unstack(fill_value=0)

# Line chart for top cities
for i, city in enumerate(top_cities):
    if city in city_monthly.columns:
        city_values = []
        for month in available_months:
            if month in city_monthly.index:
                city_values.append(city_monthly.loc[month, city])
            else:
                city_values.append(0)
        ax4.plot(available_months, city_values, marker='o', linewidth=2, 
                label=city, color=colors_main[i % len(colors_main)], markersize=6)

# Add trend arrows
for i, city in enumerate(top_cities):
    if city in city_monthly.columns and len(available_months) > 1:
        start_month = available_months[0]
        end_month = available_months[-1]
        start_val = city_monthly.loc[start_month, city] if start_month in city_monthly.index else 0
        end_val = city_monthly.loc[end_month, city] if end_month in city_monthly.index else 0
        if end_val > start_val:
            ax4.annotate('↗', xy=(end_month, end_val), fontsize=16, color='green', ha='center')
        else:
            ax4.annotate('↘', xy=(end_month, end_val), fontsize=16, color='red', ha='center')

ax4.set_xlabel('Month', fontweight='bold', fontsize=12)
ax4.set_ylabel('Revenue ($)', fontweight='bold', fontsize=12)
ax4.set_title('Geographic Sales Evolution (Top 5 Cities)', fontweight='bold', fontsize=14, pad=20)
ax4.set_xticks(available_months)
ax4.set_xticklabels([month_names[i-1] for i in available_months], rotation=45)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)

# Subplot 5: Price Point Analysis Over Time
ax5 = plt.subplot(3, 2, 5)

# Prepare data for violin plots
monthly_prices = []
for month in available_months:
    month_data = df_combined[df_combined['Month_Num'] == month]['Price Each'].values
    if len(month_data) > 0:
        monthly_prices.append(month_data)
    else:
        monthly_prices.append([0])

monthly_avg_price = df_combined.groupby('Month_Num')['Price Each'].mean()

# Create violin plots
if all(len(prices) > 0 for prices in monthly_prices):
    try:
        parts = ax5.violinplot(monthly_prices, positions=available_months, widths=0.6, showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors_main[3])
            pc.set_alpha(0.6)
    except:
        pass

    # Box plots overlay
    try:
        bp = ax5.boxplot(monthly_prices, positions=available_months, widths=0.3, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors_secondary[2])
            patch.set_alpha(0.8)
    except:
        pass

# Trend line for average transaction value
avg_price_values = [monthly_avg_price.loc[month] if month in monthly_avg_price.index else 0 
                   for month in available_months]
ax5.plot(available_months, avg_price_values, color='black', linewidth=3, 
         marker='D', markersize=8, label='Avg Transaction Value')

ax5.set_xlabel('Month', fontweight='bold', fontsize=12)
ax5.set_ylabel('Price ($)', fontweight='bold', fontsize=12)
ax5.set_title('Price Distribution Evolution Throughout 2019', fontweight='bold', fontsize=14, pad=20)
ax5.set_xticks(available_months)
ax5.set_xticklabels([month_names[i-1] for i in available_months], rotation=45)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Combined Revenue and Geographic Heatmap
ax6 = plt.subplot(3, 2, 6)

# Create heatmap data
heatmap_data = df_combined[df_combined['City'].isin(top_cities)].groupby(['Month_Num', 'City'])['Revenue'].sum().unstack(fill_value=0)

# Filter for available months and normalize
heatmap_filtered = []
for month in available_months:
    if month in heatmap_data.index:
        heatmap_filtered.append(heatmap_data.loc[month])
    else:
        heatmap_filtered.append(pd.Series(0, index=top_cities))

if heatmap_filtered:
    heatmap_array = np.array([row.values for row in heatmap_filtered])
    # Normalize by row (month)
    row_sums = heatmap_array.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    heatmap_normalized = heatmap_array / row_sums

    # Create heatmap
    im = ax6.imshow(heatmap_normalized.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Set labels
    ax6.set_xticks(range(len(available_months)))
    ax6.set_xticklabels([month_names[i-1] for i in available_months], rotation=45)
    ax6.set_yticks(range(len(top_cities)))
    ax6.set_yticklabels(top_cities)
    ax6.set_xlabel('Month', fontweight='bold', fontsize=12)
    ax6.set_ylabel('City', fontweight='bold', fontsize=12)
    ax6.set_title('Monthly Sales Intensity Heatmap by City', fontweight='bold', fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    cbar.set_label('Normalized Revenue Intensity', fontweight='bold')

# Overall layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Add overall title
fig.suptitle('Comprehensive Electronics Store Sales Analysis - 2019 Temporal Evolution', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('electronics_sales_analysis.png', dpi=300, bbox_inches='tight')
plt.show()