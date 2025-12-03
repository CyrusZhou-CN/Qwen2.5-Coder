import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('Sales Transaction v.4a.csv')

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'] = df['Price'] * df['Quantity']
df = df.dropna(subset=['CustomerNo'])
df['CustomerNo'] = df['CustomerNo'].astype(int)

# Create price categories for products
df['PriceCategory'] = pd.cut(df['Price'], bins=[0, 5, 15, 50, np.inf], 
                            labels=['Low (≤£5)', 'Medium (£5-15)', 'High (£15-50)', 'Premium (>£50)'])

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('E-Commerce Business Composition Analysis', fontsize=24, fontweight='bold', y=0.98)

# Subplot 1: Customer transaction volume composition
ax1 = plt.subplot(3, 3, 1)
customer_stats = df.groupby('CustomerNo').agg({
    'Quantity': 'sum',
    'Revenue': 'sum'
}).reset_index()
top_customers = customer_stats.nlargest(20, 'Quantity')
top_customers = top_customers.sort_values('Quantity')

# Stacked bar chart
bars = ax1.barh(range(len(top_customers)), top_customers['Quantity'], 
                color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)

# Cumulative percentage line
cumsum = top_customers['Quantity'].cumsum()
total = top_customers['Quantity'].sum()
cum_pct = (cumsum / total) * 100

ax1_twin = ax1.twinx()
ax1_twin.plot(cum_pct, range(len(top_customers)), 'ro-', linewidth=2, markersize=4)
ax1_twin.set_ylabel('Cumulative %', fontweight='bold')
ax1_twin.set_ylim(ax1.get_ylim())

ax1.set_xlabel('Total Quantity', fontweight='bold')
ax1.set_ylabel('Top 20 Customers', fontweight='bold')
ax1.set_title('Customer Transaction Volume Composition', fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)

# Subplot 2: Customer spending patterns
ax2 = plt.subplot(3, 3, 2)
top_revenue_customers = customer_stats.nlargest(8, 'Revenue')
others_revenue = customer_stats['Revenue'].sum() - top_revenue_customers['Revenue'].sum()

# Pie chart
sizes = list(top_revenue_customers['Revenue']) + [others_revenue]
labels = [f'Customer {int(c)}' for c in top_revenue_customers['CustomerNo']] + ['Others']
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))

wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                  colors=colors, startangle=90, pctdistance=0.85)

# Donut chart in center (average transaction value)
avg_transaction = df['Revenue'].mean()
circle = plt.Circle((0,0), 0.5, color='white', linewidth=2)
ax2.add_patch(circle)
ax2.text(0, 0, f'Avg Transaction\n£{avg_transaction:.2f}', 
         ha='center', va='center', fontweight='bold', fontsize=10)

ax2.set_title('Customer Revenue Share & Avg Transaction Value', fontweight='bold', pad=20)

# Subplot 3: Customer-country composition (simplified treemap using rectangles)
ax3 = plt.subplot(3, 3, 3)
country_stats = df.groupby('Country').agg({
    'CustomerNo': 'nunique',
    'Revenue': 'sum',
    'TransactionNo': 'nunique'
}).reset_index()
country_stats = country_stats.sort_values('Revenue', ascending=False).head(10)

# Create a simplified treemap using scatter plot
x_pos = np.arange(len(country_stats))
y_pos = np.zeros(len(country_stats))
sizes = (country_stats['Revenue'] / country_stats['Revenue'].max()) * 1000
colors = country_stats['TransactionNo']

scatter = ax3.scatter(x_pos, y_pos, s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='white')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(country_stats['Country'], rotation=45, ha='right')
ax3.set_ylabel('Customer Distribution', fontweight='bold')
ax3.set_title('Customer-Country Composition\n(Bubble size: Revenue, Color: Transactions)', fontweight='bold', pad=20)
plt.colorbar(scatter, ax=ax3, label='Transaction Count')

# Subplot 4: Product portfolio composition
ax4 = plt.subplot(3, 3, 4)
product_category_stats = df.groupby('PriceCategory').agg({
    'Quantity': 'sum',
    'Revenue': 'sum',
    'ProductNo': 'nunique'
}).reset_index()

# Horizontal stacked bar
categories = product_category_stats['PriceCategory']
quantities = product_category_stats['Quantity']
colors_cat = ['lightcoral', 'skyblue', 'lightgreen', 'gold']

bars = ax4.barh(range(len(categories)), quantities, color=colors_cat, alpha=0.7, edgecolor='white')

# Overlay scatter points for profitability
profit_margin = (product_category_stats['Revenue'] / product_category_stats['Quantity']) / df['Price'].mean()
ax4.scatter(quantities * 0.8, range(len(categories)), s=profit_margin*100, 
           color='red', alpha=0.8, edgecolors='darkred', linewidth=1)

ax4.set_yticks(range(len(categories)))
ax4.set_yticklabels(categories)
ax4.set_xlabel('Total Quantity Sold', fontweight='bold')
ax4.set_title('Product Portfolio Composition\n(Dots: Profitability)', fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)

# Subplot 5: Price-quantity relationship composition
ax5 = plt.subplot(3, 3, 5)
product_stats = df.groupby('ProductNo').agg({
    'Price': 'first',
    'Quantity': 'sum',
    'Revenue': 'sum'
}).reset_index()
product_stats = product_stats.sample(n=min(200, len(product_stats)))

# Background violin plot - Fixed by removing alpha parameter and setting alpha on returned objects
price_ranges = [0, 10, 20, 50, 100]
for i in range(len(price_ranges)-1):
    mask = (product_stats['Price'] >= price_ranges[i]) & (product_stats['Price'] < price_ranges[i+1])
    if mask.sum() > 0:
        data = product_stats[mask]['Quantity']
        if len(data) > 1:
            positions = [price_ranges[i] + (price_ranges[i+1] - price_ranges[i])/2]
            parts = ax5.violinplot([data], positions=positions, widths=price_ranges[i+1] - price_ranges[i] - 1, 
                                 showmeans=True)
            # Set alpha on the returned violin parts
            for pc in parts['bodies']:
                pc.set_alpha(0.3)

# Bubble chart overlay
bubble_sizes = (product_stats['Revenue'] / product_stats['Revenue'].max()) * 100
ax5.scatter(product_stats['Price'], product_stats['Quantity'], s=bubble_sizes, 
           alpha=0.6, c='steelblue', edgecolors='navy', linewidth=0.5)

ax5.set_xlabel('Price (£)', fontweight='bold')
ax5.set_ylabel('Quantity Sold', fontweight='bold')
ax5.set_title('Price-Quantity Relationship\n(Bubble size: Revenue)', fontweight='bold', pad=20)
ax5.grid(True, alpha=0.3)

# Subplot 6: Product performance matrix (simplified heatmap)
ax6 = plt.subplot(3, 3, 6)
# Create performance matrix by price category and country
perf_matrix = df.groupby(['PriceCategory', 'Country']).agg({
    'Quantity': 'sum',
    'Revenue': 'sum'
}).reset_index()
pivot_qty = perf_matrix.pivot(index='PriceCategory', columns='Country', values='Quantity').fillna(0)
pivot_qty = pivot_qty.iloc[:, :6]  # Top 6 countries

# Heatmap
im = ax6.imshow(pivot_qty.values, cmap='YlOrRd', aspect='auto')
ax6.set_xticks(range(len(pivot_qty.columns)))
ax6.set_xticklabels(pivot_qty.columns, rotation=45, ha='right')
ax6.set_yticks(range(len(pivot_qty.index)))
ax6.set_yticklabels(pivot_qty.index)
ax6.set_title('Product Performance Matrix\n(Quantity by Category & Country)', fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
cbar.set_label('Quantity Sold', fontweight='bold')

# Subplot 7: Transaction size composition
ax7 = plt.subplot(3, 3, 7)
transaction_values = df.groupby('TransactionNo')['Revenue'].sum()

# Histogram
n, bins, patches = ax7.hist(transaction_values, bins=50, alpha=0.7, color='lightblue', 
                           edgecolor='white', density=True)

# KDE overlay
kde_x = np.linspace(transaction_values.min(), transaction_values.max(), 100)
kde = stats.gaussian_kde(transaction_values)
ax7.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')

# Quartile lines
quartiles = transaction_values.quantile([0.25, 0.5, 0.75])
colors_q = ['green', 'orange', 'red']
for i, (q, color) in enumerate(zip(quartiles, colors_q)):
    ax7.axvline(q, color=color, linestyle='--', linewidth=2, 
               label=f'Q{i+1}: £{q:.2f}')

ax7.set_xlabel('Transaction Value (£)', fontweight='bold')
ax7.set_ylabel('Density', fontweight='bold')
ax7.set_title('Transaction Size Distribution', fontweight='bold', pad=20)
ax7.legend()
ax7.grid(True, alpha=0.3)

# Subplot 8: Transaction complexity analysis
ax8 = plt.subplot(3, 3, 8)
transaction_complexity = df.groupby('TransactionNo').agg({
    'ProductNo': 'nunique',
    'Revenue': 'sum',
    'Quantity': 'sum'
}).reset_index()

# Stacked area chart (simplified)
complexity_bins = pd.cut(transaction_complexity['ProductNo'], bins=[0, 1, 3, 5, 10, np.inf], 
                        labels=['1 item', '2-3 items', '4-5 items', '6-10 items', '10+ items'])
complexity_counts = complexity_bins.value_counts().sort_index()

# Create cumulative data for stacked area
cumulative = np.cumsum(complexity_counts.values)
x_range = range(len(complexity_counts))

ax8.fill_between(x_range, 0, cumulative, alpha=0.6, color='lightcoral', label='Cumulative Transactions')

# Line plot for average transaction value
ax8_twin = ax8.twinx()
avg_values = []
for cat in complexity_counts.index:
    mask = complexity_bins == cat
    avg_val = transaction_complexity[mask]['Revenue'].mean() if mask.sum() > 0 else 0
    avg_values.append(avg_val)

ax8_twin.plot(x_range, avg_values, 'bo-', linewidth=2, markersize=6, label='Avg Transaction Value')
ax8_twin.set_ylabel('Average Transaction Value (£)', fontweight='bold')

ax8.set_xticks(x_range)
ax8.set_xticklabels(complexity_counts.index, rotation=45, ha='right')
ax8.set_xlabel('Transaction Complexity', fontweight='bold')
ax8.set_ylabel('Cumulative Transaction Count', fontweight='bold')
ax8.set_title('Transaction Complexity Analysis', fontweight='bold', pad=20)
ax8.grid(True, alpha=0.3)

# Subplot 9: Business composition overview (waffle-style chart)
ax9 = plt.subplot(3, 3, 9)

# Transaction size categories
transaction_values = df.groupby('TransactionNo')['Revenue'].sum()
size_categories = pd.cut(transaction_values, bins=[0, 50, 200, 500, np.inf], 
                        labels=['Small (<£50)', 'Medium (£50-200)', 'Large (£200-500)', 'XLarge (>£500)'])
size_counts = size_categories.value_counts()

# Create waffle-style visualization using scatter
total_transactions = len(transaction_values)
grid_size = 10
squares_per_category = (size_counts / total_transactions * grid_size**2).round().astype(int)

colors_waffle = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
x_coords, y_coords, colors_list = [], [], []

start_idx = 0
for i, (category, count) in enumerate(squares_per_category.items()):
    for j in range(count):
        idx = start_idx + j
        x_coords.append(idx % grid_size)
        y_coords.append(idx // grid_size)
        colors_list.append(colors_waffle[i])
    start_idx += count

ax9.scatter(x_coords, y_coords, s=200, c=colors_list, marker='s', alpha=0.8, edgecolors='white')

# Add legend
legend_elements = [mpatches.Patch(color=colors_waffle[i], label=f'{cat}: {count}') 
                  for i, (cat, count) in enumerate(size_counts.items())]
ax9.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

ax9.set_xlim(-0.5, grid_size-0.5)
ax9.set_ylim(-0.5, grid_size-0.5)
ax9.set_aspect('equal')
ax9.set_title('Business Composition Overview\n(Transaction Size Distribution)', fontweight='bold', pad=20)
ax9.set_xticks([])
ax9.set_yticks([])

# Final layout adjustment
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('ecommerce_composition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()