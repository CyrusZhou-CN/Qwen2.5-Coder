import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load and preprocess data
df = pd.read_csv('data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
df['Month'] = df['Date'].dt.to_period('M')

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palettes
dept_colors = plt.cm.Set3(np.linspace(0, 1, 12))
location_colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Row 1: Department Analysis

# Subplot 1: Horizontal bar chart with scatter overlay
ax1 = plt.subplot(3, 3, 1)
dept_stats = df.groupby('Department').agg({
    'Amount(INR)': ['sum', 'mean'],
    'Views': 'mean',
    'Title': 'count'
}).round(2)
dept_stats.columns = ['Total_Amount', 'Avg_Amount', 'Avg_Views', 'Count']
dept_stats = dept_stats.sort_values('Total_Amount', ascending=True).tail(10)

bars = ax1.barh(range(len(dept_stats)), dept_stats['Total_Amount'], 
                color=dept_colors[:len(dept_stats)], alpha=0.7)
ax1_twin = ax1.twiny()
scatter = ax1_twin.scatter(dept_stats['Avg_Views'], range(len(dept_stats)), 
                          s=100, color='red', alpha=0.8, zorder=5)

ax1.set_yticks(range(len(dept_stats)))
ax1.set_yticklabels([dept[:25] + '...' if len(dept) > 25 else dept 
                     for dept in dept_stats.index], fontsize=9)
ax1.set_xlabel('Total Bribery Amount (INR)', fontweight='bold')
ax1_twin.set_xlabel('Average Views per Complaint', fontweight='bold', color='red')
ax1.set_title('Department Bribery Totals vs Average Views', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Subplot 2: Violin plot with box plot overlay
ax2 = plt.subplot(3, 3, 2)
top_depts = df['Department'].value_counts().head(6).index
violin_data = [df[df['Department'] == dept]['Amount(INR)'].values for dept in top_depts]

parts = ax2.violinplot(violin_data, positions=range(len(top_depts)), 
                       showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(dept_colors[i])
    pc.set_alpha(0.7)

box_data = ax2.boxplot(violin_data, positions=range(len(top_depts)), 
                       widths=0.3, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.8))

ax2.set_xticks(range(len(top_depts)))
ax2.set_xticklabels([dept[:15] + '...' if len(dept) > 15 else dept 
                     for dept in top_depts], rotation=45, ha='right')
ax2.set_ylabel('Bribery Amount (INR)', fontweight='bold')
ax2.set_title('Amount Distribution by Top Departments', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Subplot 3: Bubble chart
ax3 = plt.subplot(3, 3, 3)
top8_depts = df['Department'].value_counts().head(8).index
bubble_data = df[df['Department'].isin(top8_depts)].groupby('Department').agg({
    'Amount(INR)': 'mean',
    'Title': 'count'
}).reset_index()

scatter = ax3.scatter(range(len(bubble_data)), bubble_data['Amount(INR)'], 
                     s=bubble_data['Title']*3, alpha=0.6, 
                     c=range(len(bubble_data)), cmap='viridis')

ax3.set_xticks(range(len(bubble_data)))
ax3.set_xticklabels([dept[:12] + '...' if len(dept) > 12 else dept 
                     for dept in bubble_data['Department']], rotation=45, ha='right')
ax3.set_ylabel('Average Bribery Amount (INR)', fontweight='bold')
ax3.set_title('Department Bubble Chart\n(Size = Complaint Count)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Row 2: Geographic and Temporal Patterns

# Subplot 4: Stacked bar with line overlay
ax4 = plt.subplot(3, 3, 4)
location_stats = df.groupby('Location').agg({
    'Title': 'count',
    'Amount(INR)': 'mean'
}).sort_values('Title', ascending=False).head(8)

bars = ax4.bar(range(len(location_stats)), location_stats['Title'], 
               color=location_colors[:len(location_stats)], alpha=0.7)
ax4_twin = ax4.twinx()
line = ax4_twin.plot(range(len(location_stats)), location_stats['Amount(INR)'], 
                     'ro-', linewidth=2, markersize=8, color='red')

ax4.set_xticks(range(len(location_stats)))
ax4.set_xticklabels([loc.split(',')[0] for loc in location_stats.index], 
                    rotation=45, ha='right')
ax4.set_ylabel('Complaint Count', fontweight='bold')
ax4_twin.set_ylabel('Average Amount (INR)', fontweight='bold', color='red')
ax4.set_title('Complaints by Location with Average Amounts', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Subplot 5: Dual-axis time series
ax5 = plt.subplot(3, 3, 5)
monthly_data = df.groupby('Month').agg({
    'Title': 'count',
    'Amount(INR)': 'mean'
}).reset_index()
monthly_data['Month_str'] = monthly_data['Month'].astype(str)

bars = ax5.bar(range(len(monthly_data)), monthly_data['Title'], 
               alpha=0.7, color='skyblue')
ax5_twin = ax5.twinx()
line = ax5_twin.plot(range(len(monthly_data)), monthly_data['Amount(INR)'], 
                     'ro-', linewidth=2, markersize=6, color='red')

ax5.set_xticks(range(len(monthly_data)))
ax5.set_xticklabels(monthly_data['Month_str'], rotation=45, ha='right')
ax5.set_ylabel('Monthly Complaint Count', fontweight='bold')
ax5_twin.set_ylabel('Average Amount (INR)', fontweight='bold', color='red')
ax5.set_title('Monthly Trends: Complaints vs Average Amounts', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot 6: Heatmap
ax6 = plt.subplot(3, 3, 6)
top_locations = df['Location'].value_counts().head(6).index
top_departments = df['Department'].value_counts().head(6).index
heatmap_data = df[df['Location'].isin(top_locations) & 
                  df['Department'].isin(top_departments)].groupby(['Location', 'Department'])['Amount(INR)'].mean().unstack(fill_value=0)

im = ax6.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
ax6.set_xticks(range(len(heatmap_data.columns)))
ax6.set_xticklabels([dept[:15] + '...' if len(dept) > 15 else dept 
                     for dept in heatmap_data.columns], rotation=45, ha='right')
ax6.set_yticks(range(len(heatmap_data.index)))
ax6.set_yticklabels([loc.split(',')[0] for loc in heatmap_data.index])
ax6.set_title('Location-Department Average Amounts', fontweight='bold', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
cbar.set_label('Average Amount (INR)', fontweight='bold')

# Row 3: Engagement and Amount Relationships

# Subplot 7: Scatter with regression lines
ax7 = plt.subplot(3, 3, 7)
top3_depts = df['Department'].value_counts().head(3).index
colors = ['red', 'blue', 'green']

for i, dept in enumerate(top3_depts):
    dept_data = df[df['Department'] == dept]
    ax7.scatter(dept_data['Views'], dept_data['Amount(INR)'], 
               alpha=0.6, color=colors[i], label=dept[:20] + '...' if len(dept) > 20 else dept)
    
    # Add regression line
    z = np.polyfit(dept_data['Views'], dept_data['Amount(INR)'], 1)
    p = np.poly1d(z)
    ax7.plot(dept_data['Views'], p(dept_data['Views']), 
             color=colors[i], linestyle='--', linewidth=2)

ax7.set_xlabel('Views', fontweight='bold')
ax7.set_ylabel('Amount (INR)', fontweight='bold')
ax7.set_title('Views vs Amount with Regression Lines', fontweight='bold', fontsize=12)
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Subplot 8: Histogram with KDE and statistics
ax8 = plt.subplot(3, 3, 8)
amounts = df['Amount(INR)']
n, bins, patches = ax8.hist(amounts, bins=50, alpha=0.7, color='skyblue', density=True)

# KDE overlay
kde_x = np.linspace(amounts.min(), amounts.max(), 100)
kde = stats.gaussian_kde(amounts)
ax8.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')

# Add mean and median lines
mean_val = amounts.mean()
median_val = amounts.median()
ax8.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
ax8.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')

ax8.set_xlabel('Bribery Amount (INR)', fontweight='bold')
ax8.set_ylabel('Density', fontweight='bold')
ax8.set_title('Amount Distribution with Statistics', fontweight='bold', fontsize=12)
ax8.legend()
ax8.grid(True, alpha=0.3)

# Subplot 9: Parallel coordinates plot
ax9 = plt.subplot(3, 3, 9)
# Prepare data for parallel coordinates
parallel_data = df.copy()
parallel_data['Dept_Code'] = pd.Categorical(parallel_data['Department']).codes
parallel_data['Location_Code'] = pd.Categorical(parallel_data['Location']).codes

# Normalize the data
scaler = StandardScaler()
features = ['Amount(INR)', 'Views', 'Dept_Code']
normalized_data = scaler.fit_transform(parallel_data[features])

# Sample data for clarity
sample_indices = np.random.choice(len(normalized_data), size=min(200, len(normalized_data)), replace=False)
sample_data = normalized_data[sample_indices]
sample_locations = parallel_data.iloc[sample_indices]['Location_Code']

# Create parallel coordinates
for i in range(len(sample_data)):
    ax9.plot(range(3), sample_data[i], alpha=0.6, 
             color=plt.cm.tab10(sample_locations.iloc[i] % 10))

ax9.set_xticks(range(3))
ax9.set_xticklabels(['Amount\n(normalized)', 'Views\n(normalized)', 'Department\n(normalized)'])
ax9.set_ylabel('Normalized Values', fontweight='bold')
ax9.set_title('Parallel Coordinates Plot\n(Colored by Location)', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.show()