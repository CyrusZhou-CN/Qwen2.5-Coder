import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('Pakisan_Toshkhana_Imputed.csv')

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Assessed Value'] = pd.to_numeric(df['Assessed Value'], errors='coerce')
df['Retention Cost'] = pd.to_numeric(df['Retention Cost'], errors='coerce')

# Clean retention status
df['Retention Status'] = df['Retained'].fillna('Unknown')
df['Retention Status'] = df['Retention Status'].replace({'Yes': 'Retained', 'No': 'Not Retained'})

# Filter data to avoid timeout issues
df = df.dropna(subset=['Year', 'Assessed Value'])
df = df[df['Year'] >= 2000]  # Focus on recent years

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes
colors_retention = ['#2E86AB', '#A23B72', '#F18F01']
retention_statuses = df['Retention Status'].unique()[:3]

# Subplot 1: Stacked bar chart with line plot overlay
ax1 = plt.subplot(3, 3, 1)
top_categories = df['Item Category'].value_counts().head(6)
df_filtered = df[df['Item Category'].isin(top_categories.index)]

category_retention = pd.crosstab(df_filtered['Item Category'], df_filtered['Retention Status'])
category_retention_pct = category_retention.div(category_retention.sum(axis=1), axis=0) * 100

# Stacked bar chart
bars = category_retention_pct.plot(kind='bar', stacked=True, ax=ax1, 
                                  color=colors_retention[:len(category_retention_pct.columns)], alpha=0.8)

# Overlay line plot for total counts
ax1_twin = ax1.twinx()
category_counts = df_filtered['Item Category'].value_counts()
ax1_twin.plot(range(len(category_counts)), category_counts.values, 
              color='red', marker='o', linewidth=2, markersize=6)
ax1_twin.set_ylabel('Total Count', color='red', fontweight='bold')

ax1.set_title('Gift Retention Patterns by Category', fontweight='bold', fontsize=10)
ax1.set_xlabel('Item Category', fontweight='bold')
ax1.set_ylabel('Percentage (%)', fontweight='bold')
ax1.legend(title='Retention Status', loc='upper left', fontsize=8)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# Subplot 2: Pie chart with donut chart overlay
ax2 = plt.subplot(3, 3, 2)
top_affiliations = df['Affiliation'].value_counts().head(5)
top_recipients = df['Name of Recipient'].value_counts().head(4)

# Outer pie chart for affiliations
wedges1, texts1, autotexts1 = ax2.pie(top_affiliations.values, 
                                       autopct='%1.1f%%', 
                                       colors=plt.cm.Set3(np.linspace(0, 1, 5)), 
                                       radius=1, startangle=90)

# Inner donut chart for top recipients
wedges2, texts2, autotexts2 = ax2.pie(top_recipients.values,
                                       autopct='%1.0f%%',
                                       colors=plt.cm.Set2(np.linspace(0, 1, 4)),
                                       radius=0.6, startangle=90)

ax2.set_title('Top Affiliations & Recipients', fontweight='bold', fontsize=10)

# Subplot 3: Bubble chart as treemap alternative
ax3 = plt.subplot(3, 3, 3)
category_stats = df_filtered.groupby(['Item Category', 'Retention Status']).agg({
    'Assessed Value': ['sum', 'count']
}).reset_index()
category_stats.columns = ['Category', 'Status', 'Total_Value', 'Count']

# Create bubble chart
np.random.seed(42)
for i, row in category_stats.iterrows():
    x = np.random.uniform(0, 10)
    y = np.random.uniform(0, 10)
    size = min(row['Count'] * 50, 1000)  # Cap size to avoid huge bubbles
    status_idx = list(retention_statuses).index(row['Status']) if row['Status'] in retention_statuses else 0
    color = colors_retention[status_idx]
    ax3.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='black')

ax3.set_title('Category-Status Composition\n(Bubble Size = Frequency)', fontweight='bold', fontsize=10)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_xticks([])
ax3.set_yticks([])

# Subplot 4: Stacked area chart with scatter overlay
ax4 = plt.subplot(3, 3, 4)
years_available = sorted(df['Year'].dropna().unique())[:8]  # Limit years
df_years = df[df['Year'].isin(years_available)]

yearly_category = pd.crosstab(df_years['Year'], df_years['Item Category'], normalize='index') * 100
top_cats = yearly_category.sum().nlargest(5).index
yearly_category_top = yearly_category[top_cats]

yearly_category_top.plot(kind='area', stacked=True, ax=ax4, alpha=0.7)

# Overlay high-value gifts
median_value = df['Assessed Value'].median()
high_value_gifts = df_years[df_years['Assessed Value'] > median_value]
yearly_high_value = high_value_gifts.groupby('Year').size()

for year, count in yearly_high_value.items():
    ax4.scatter(year, 50, s=count*2, c='red', alpha=0.8, edgecolors='black')

ax4.set_title('Temporal Gift Categories\n(Red dots = High-value)', fontweight='bold', fontsize=10)
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontweight='bold')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Subplot 5: Horizontal stacked bar with error bars
ax5 = plt.subplot(3, 3, 5)
top_affiliations_list = df['Affiliation'].value_counts().head(5).index
df_aff = df[df['Affiliation'].isin(top_affiliations_list)]

affiliation_stats = df_aff.groupby('Affiliation').agg({
    'Retention Cost': ['sum', 'std'],
    'Assessed Value': 'mean'
}).reset_index()
affiliation_stats.columns = ['Affiliation', 'Total_Cost', 'Cost_Std', 'Avg_Value']

y_pos = np.arange(len(affiliation_stats))
bars = ax5.barh(y_pos, affiliation_stats['Total_Cost'], 
                color=plt.cm.Set3(np.linspace(0, 1, len(affiliation_stats))), alpha=0.8)

# Add error bars
ax5.errorbar(affiliation_stats['Total_Cost'], y_pos, 
            xerr=affiliation_stats['Cost_Std'].fillna(0), 
            fmt='none', color='black', capsize=3)

ax5.set_title('Retention Cost by Affiliation', fontweight='bold', fontsize=10)
ax5.set_xlabel('Total Retention Cost', fontweight='bold')
ax5.set_yticks(y_pos)
ax5.set_yticklabels([aff[:20] + '...' if len(aff) > 20 else aff 
                     for aff in affiliation_stats['Affiliation']], fontsize=8)

# Subplot 6: Waffle chart simulation
ax6 = plt.subplot(3, 3, 6)
retention_counts = df['Retention Status'].value_counts().head(3)
total = retention_counts.sum()

# Create waffle-like visualization
grid_size = 10
squares_per_status = (retention_counts / total * grid_size**2).round().astype(int)

colors_flat = []
for i, (status, count) in enumerate(squares_per_status.items()):
    color = colors_retention[i] if i < len(colors_retention) else 'lightgray'
    colors_flat.extend([color] * count)

# Fill remaining squares
while len(colors_flat) < grid_size**2:
    colors_flat.append('lightgray')

colors_grid = np.array(colors_flat[:grid_size**2]).reshape(grid_size, grid_size)

for i in range(grid_size):
    for j in range(grid_size):
        ax6.add_patch(Rectangle((j, i), 1, 1, facecolor=colors_grid[i, j], 
                               edgecolor='white', linewidth=1))

ax6.set_xlim(0, grid_size)
ax6.set_ylim(0, grid_size)
ax6.set_aspect('equal')
ax6.set_title('Retention Status Proportion', fontweight='bold', fontsize=10)
ax6.set_xticks([])
ax6.set_yticks([])

# Subplot 7: Grouped bar chart with secondary axis
ax7 = plt.subplot(3, 3, 7)
category_stats = df_filtered.groupby('Item Category').agg({
    'Assessed Value': 'mean',
    'Retention Cost': 'mean'
}).reset_index()

x = np.arange(len(category_stats))
width = 0.35

bars1 = ax7.bar(x - width/2, category_stats['Assessed Value'], width, 
                label='Avg Assessed Value', color='skyblue', alpha=0.8)
bars2 = ax7.bar(x + width/2, category_stats['Retention Cost'], width,
                label='Avg Retention Cost', color='lightcoral', alpha=0.8)

ax7.set_title('Average Values by Category', fontweight='bold', fontsize=10)
ax7.set_xlabel('Item Category', fontweight='bold')
ax7.set_ylabel('Average Value/Cost', fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat 
                     for cat in category_stats['Item Category']], 
                    rotation=45, ha='right', fontsize=8)
ax7.legend(fontsize=8)

# Subplot 8: Histogram with KDE overlay
ax8 = plt.subplot(3, 3, 8)
# Sample data to avoid timeout
sample_size = min(1000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# Histogram by retention status
for i, status in enumerate(retention_statuses[:2]):  # Limit to 2 statuses
    status_data = df_sample[df_sample['Retention Status'] == status]['Assessed Value'].dropna()
    if len(status_data) > 10:
        ax8.hist(status_data, bins=15, alpha=0.6, color=colors_retention[i], 
                label=status, density=True)
        
        # KDE overlay
        if len(status_data) > 1:
            try:
                kde = stats.gaussian_kde(status_data)
                x_range = np.linspace(status_data.min(), status_data.max(), 50)
                ax8.plot(x_range, kde(x_range), linewidth=2, color=colors_retention[i])
            except:
                pass

ax8.set_title('Assessed Value Distribution\n(with KDE)', fontweight='bold', fontsize=10)
ax8.set_xlabel('Assessed Value', fontweight='bold')
ax8.set_ylabel('Density', fontweight='bold')
ax8.legend(fontsize=8)

# Subplot 9: Radial bar chart
ax9 = plt.subplot(3, 3, 9, projection='polar')
monthly_data = df.groupby('Month')['Assessed Value'].sum()

theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
width = 2*np.pi / 12

values = [monthly_data.get(month, 0) for month in range(1, 13)]
bars = ax9.bar(theta, values, width=width, color='lightblue', alpha=0.8)

ax9.set_title('Monthly Gift Values\n(Radial View)', fontweight='bold', fontsize=10, pad=20)
ax9.set_theta_zero_location('N')
ax9.set_theta_direction(-1)
ax9.set_thetagrids(np.degrees(theta), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Final layout adjustment
plt.tight_layout(pad=2.0)
plt.savefig('pakistan_toshkhana_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()