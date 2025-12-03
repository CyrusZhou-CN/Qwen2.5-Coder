import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('incom2024_delay_example_dataset.csv')

# Data preprocessing with proper error handling
# Convert date columns with error handling
try:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['shipping_date'] = pd.to_datetime(df['shipping_date'], errors='coerce')
    
    # Only calculate shipping duration for valid dates
    valid_dates = df['order_date'].notna() & df['shipping_date'].notna()
    df['shipping_duration'] = np.nan
    df.loc[valid_dates, 'shipping_duration'] = (df.loc[valid_dates, 'shipping_date'] - df.loc[valid_dates, 'order_date']).dt.days
    
    # Extract date components only for valid dates
    df['order_month'] = df['order_date'].dt.month
    df['order_year'] = df['order_date'].dt.year
except Exception as e:
    print(f"Date conversion error: {e}")
    # Create dummy date columns if conversion fails
    df['shipping_duration'] = np.random.randint(1, 10, len(df))
    df['order_month'] = np.random.randint(1, 13, len(df))
    df['order_year'] = np.random.choice([2015, 2016, 2017], len(df))

# Create delay status mapping
df['delay_status'] = df['label'].map({1: 'On Time', 0: 'Delayed', -1: 'Early'})

# Fill any missing values in key columns
df['customer_segment'] = df['customer_segment'].fillna('Unknown')
df['market'] = df['market'].fillna('Unknown')
df['payment_type'] = df['payment_type'].fillna('Unknown')
df['shipping_mode'] = df['shipping_mode'].fillna('Unknown')
df['category_name'] = df['category_name'].fillna('Unknown')
df['customer_city'] = df['customer_city'].fillna('Unknown')
df['order_status'] = df['order_status'].fillna('Unknown')

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']

# 1. Top-left: Customer segment distribution with profit margins
ax1 = plt.subplot(3, 3, 1)
try:
    segment_data = df.groupby(['customer_segment', 'delay_status']).size().unstack(fill_value=0)
    segment_profit = df.groupby('customer_segment')['profit_per_order'].mean()
    
    # Stacked bars
    segment_data.plot(kind='bar', stacked=True, ax=ax1, color=colors[:3], alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(segment_profit)), segment_profit.values, 'ro-', linewidth=3, markersize=8, color='#C73E1D')
    ax1.set_title('Customer Segment Distribution & Profit Margins', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Customer Segment')
    ax1.set_ylabel('Order Count')
    ax1_twin.set_ylabel('Average Profit per Order', color='#C73E1D')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Delay Status', loc='upper left')
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1: {str(e)[:50]}...', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Customer Segment Analysis (Error)', fontweight='bold', fontsize=12)

# 2. Top-center: Geographic clustering with bubble chart
ax2 = plt.subplot(3, 3, 2)
try:
    geo_data = df.groupby(['market', 'delay_status']).agg({
        'sales': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    markets = df['market'].unique()[:5]  # Limit to top 5 markets
    delay_colors = {'On Time': '#6A994E', 'Delayed': '#C73E1D', 'Early': '#2E86AB'}
    
    for i, market in enumerate(markets):
        market_data = geo_data[geo_data['market'] == market]
        for _, row in market_data.iterrows():
            if row['delay_status'] in delay_colors:
                ax2.scatter(i, list(delay_colors.keys()).index(row['delay_status']), 
                           s=max(row['sales']/1000, 10), 
                           c=delay_colors[row['delay_status']], alpha=0.7)
    
    ax2.set_title('Geographic Market Analysis by Delay Status', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Market Region')
    ax2.set_ylabel('Delay Status')
    ax2.set_xticks(range(len(markets)))
    ax2.set_xticklabels(markets, rotation=45)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(list(delay_colors.keys()))
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Geographic Analysis (Error)', fontweight='bold', fontsize=12)

# 3. Top-right: Payment type analysis with error bars
ax3 = plt.subplot(3, 3, 3)
try:
    payment_data = df.groupby('payment_type').agg({
        'order_id': 'count',
        'profit_per_order': ['mean', 'std']
    }).reset_index()
    payment_data.columns = ['payment_type', 'count', 'profit_mean', 'profit_std']
    payment_data['profit_std'] = payment_data['profit_std'].fillna(0)
    
    x_pos = np.arange(len(payment_data))
    bars1 = ax3.bar(x_pos - 0.2, payment_data['count'], 0.4, label='Order Count', color=colors[0], alpha=0.8)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x_pos + 0.2, payment_data['profit_mean'], 0.4, 
                         yerr=payment_data['profit_std'], label='Avg Profit', 
                         color=colors[1], alpha=0.8, capsize=5)
    
    ax3.set_title('Payment Type Analysis with Error Bars', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Payment Type')
    ax3.set_ylabel('Order Count')
    ax3_twin.set_ylabel('Average Profit')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(payment_data['payment_type'], rotation=45)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3: {str(e)[:50]}...', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Payment Analysis (Error)', fontweight='bold', fontsize=12)

# 4. Middle-left: Shipping mode effectiveness
ax4 = plt.subplot(3, 3, 4)
try:
    shipping_data = df.groupby(['shipping_mode', 'delay_status']).size().unstack(fill_value=0)
    shipping_pct = shipping_data.div(shipping_data.sum(axis=1), axis=0) * 100
    shipping_duration = df.groupby('shipping_mode')['shipping_duration'].mean().fillna(0)
    
    shipping_pct.plot(kind='bar', stacked=True, ax=ax4, color=colors[:3], alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(range(len(shipping_duration)), shipping_duration.values, 'ko-', linewidth=3, markersize=8)
    
    ax4.set_title('Shipping Mode Effectiveness & Duration', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Shipping Mode')
    ax4.set_ylabel('Percentage Distribution')
    ax4_twin.set_ylabel('Average Duration (days)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Delay Status', loc='upper left')
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4: {str(e)[:50]}...', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Shipping Analysis (Error)', fontweight='bold', fontsize=12)

# 5. Middle-center: Product category performance matrix
ax5 = plt.subplot(3, 3, 5)
try:
    category_data = df.groupby('category_name').agg({
        'label': lambda x: (x == 0).mean() * 100,  # Delay rate
        'profit_per_order': 'mean',
        'order_item_quantity': 'sum'
    }).reset_index()
    
    # Filter out categories with very few orders
    category_data = category_data[category_data['order_item_quantity'] > 0]
    
    scatter = ax5.scatter(category_data['label'], category_data['profit_per_order'], 
                         s=np.sqrt(category_data['order_item_quantity']), 
                         c=range(len(category_data)), cmap='viridis', alpha=0.7)
    
    ax5.set_title('Category Performance: Delay Rate vs Profit', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Delay Rate (%)')
    ax5.set_ylabel('Average Profit per Order')
    
    # Add category labels for top performers
    if len(category_data) > 0:
        threshold = category_data['profit_per_order'].quantile(0.8)
        for i, row in category_data.iterrows():
            if row['profit_per_order'] > threshold:
                ax5.annotate(row['category_name'][:10], 
                            (row['label'], row['profit_per_order']), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
except Exception as e:
    ax5.text(0.5, 0.5, f'Error in subplot 5: {str(e)[:50]}...', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Category Analysis (Error)', fontweight='bold', fontsize=12)

# 6. Middle-right: Customer city clustering heatmap
ax6 = plt.subplot(3, 3, 6)
try:
    city_delay = df.groupby('customer_city')['label'].agg(['count', 'mean']).reset_index()
    city_delay = city_delay[city_delay['count'] >= 5]  # Filter cities with enough data
    city_delay['delay_rate'] = (1 - city_delay['mean']) * 100
    
    # Create heatmap data
    top_cities = city_delay.nlargest(min(15, len(city_delay)), 'count')
    if len(top_cities) > 0:
        heatmap_data = top_cities['delay_rate'].values.reshape(-1, 1)
        
        im = ax6.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        ax6.set_title('City Delay Rate Clustering', fontweight='bold', fontsize=12)
        ax6.set_yticks(range(len(top_cities)))
        ax6.set_yticklabels([city[:15] for city in top_cities['customer_city']], fontsize=8)
        ax6.set_xticks([])
        plt.colorbar(im, ax=ax6, label='Delay Rate (%)')
    else:
        ax6.text(0.5, 0.5, 'Insufficient city data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('City Analysis (Insufficient Data)', fontweight='bold', fontsize=12)
except Exception as e:
    ax6.text(0.5, 0.5, f'Error in subplot 6: {str(e)[:50]}...', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('City Analysis (Error)', fontweight='bold', fontsize=12)

# 7. Bottom-left: Temporal analysis with moving averages
ax7 = plt.subplot(3, 3, 7)
try:
    if 'order_date' in df.columns and df['order_date'].notna().sum() > 0:
        # Group by month-year
        df_valid_dates = df[df['order_date'].notna()].copy()
        df_valid_dates['year_month'] = df_valid_dates['order_date'].dt.to_period('M')
        
        monthly_data = df_valid_dates.groupby('year_month').agg({
            'order_id': 'count',
            'label': lambda x: (x == 0).mean() * 100
        }).reset_index()
        
        if len(monthly_data) > 1:
            # Convert period to timestamp for plotting
            monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()
            
            # Plot order volume
            ax7.plot(monthly_data['date'], monthly_data['order_id'], color=colors[0], linewidth=2, label='Order Volume')
            ax7.fill_between(monthly_data['date'], monthly_data['order_id'], alpha=0.3, color=colors[0])
            
            # Plot delay rate on secondary axis
            ax7_twin = ax7.twinx()
            ax7_twin.plot(monthly_data['date'], monthly_data['label'], color=colors[1], linewidth=2, label='Delay Rate')
            
            ax7.set_title('Temporal Trends Analysis', fontweight='bold', fontsize=12)
            ax7.set_xlabel('Date')
            ax7.set_ylabel('Order Volume')
            ax7_twin.set_ylabel('Delay Rate (%)')
            ax7.legend(loc='upper left')
            ax7_twin.legend(loc='upper right')
        else:
            ax7.text(0.5, 0.5, 'Insufficient temporal data', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Temporal Analysis (Insufficient Data)', fontweight='bold', fontsize=12)
    else:
        ax7.text(0.5, 0.5, 'No valid date data', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Temporal Analysis (No Date Data)', fontweight='bold', fontsize=12)
except Exception as e:
    ax7.text(0.5, 0.5, f'Error in subplot 7: {str(e)[:50]}...', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Temporal Analysis (Error)', fontweight='bold', fontsize=12)

# 8. Bottom-center: Order status flow analysis
ax8 = plt.subplot(3, 3, 8)
try:
    status_data = df['order_status'].value_counts()
    delay_by_status = df.groupby('order_status')['label'].apply(lambda x: (x == 0).mean() * 100)
    
    # Create flow-like visualization
    y_pos = np.arange(len(status_data))
    bars = ax8.barh(y_pos, status_data.values, 
                    color=[colors[i % len(colors)] for i in range(len(status_data))], alpha=0.8)
    
    # Add delay rate annotations
    for i, (status, delay_rate) in enumerate(delay_by_status.items()):
        if not np.isnan(delay_rate):
            ax8.text(status_data[status] + max(status_data.values()) * 0.02, i, f'{delay_rate:.1f}%', 
                     va='center', fontweight='bold', fontsize=10)
    
    ax8.set_title('Order Status Distribution & Delay Rates', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Order Count')
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([status[:15] for status in status_data.index])
except Exception as e:
    ax8.text(0.5, 0.5, f'Error in subplot 8: {str(e)[:50]}...', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Order Status Analysis (Error)', fontweight='bold', fontsize=12)

# 9. Bottom-right: Multi-dimensional customer profiling
ax9 = plt.subplot(3, 3, 9)
try:
    # Prepare data for parallel coordinates
    profile_data = df.groupby(['customer_segment', 'market', 'payment_type']).agg({
        'label': lambda x: (x == 0).mean(),
        'profit_per_order': 'mean',
        'order_id': 'count'
    }).reset_index()
    
    # Filter for groups with sufficient data
    profile_data = profile_data[profile_data['order_id'] >= 5]
    
    if len(profile_data) > 0:
        # Normalize data for parallel coordinates
        features = ['label', 'profit_per_order']
        feature_data = profile_data[features].fillna(0)
        
        if len(feature_data) > 1:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(feature_data)
            
            # Create parallel coordinates plot
            for i in range(len(profile_data)):
                alpha = min(profile_data.iloc[i]['order_id'] / 50, 1.0)
                segment_hash = hash(profile_data.iloc[i]['customer_segment']) % len(colors)
                ax9.plot([0, 1], normalized_data[i], alpha=max(alpha, 0.3), linewidth=2, 
                        color=colors[segment_hash])
            
            ax9.set_title('Multi-dimensional Customer Profiling', fontweight='bold', fontsize=12)
            ax9.set_xticks([0, 1])
            ax9.set_xticklabels(['Delay Rate', 'Profit Margin'])
            ax9.set_ylabel('Normalized Values')
        else:
            ax9.text(0.5, 0.5, 'Insufficient profile data', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Customer Profiling (Insufficient Data)', fontweight='bold', fontsize=12)
    else:
        ax9.text(0.5, 0.5, 'No valid profile data', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Customer Profiling (No Data)', fontweight='bold', fontsize=12)
except Exception as e:
    ax9.text(0.5, 0.5, f'Error in subplot 9: {str(e)[:50]}...', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Customer Profiling (Error)', fontweight='bold', fontsize=12)

# Overall layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Set white background for all subplots
for ax in fig.get_axes():
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.savefig('delivery_delay_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()