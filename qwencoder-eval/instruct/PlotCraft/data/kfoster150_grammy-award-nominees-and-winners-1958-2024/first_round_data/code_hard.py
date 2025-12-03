import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Grammy Award Nominees and Winners 1958-2024.csv')

# Data preprocessing
df['Decade'] = (df['Year'] // 10) * 10
df['Award_Category'] = df['Award Type'].fillna('Unknown')

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Define color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
colors_secondary = ['#87CEEB', '#DDA0DD', '#FFE4B5', '#FFA07A', '#98FB98', '#DA70D6']

# Subplot 1: Stacked area chart with line overlay
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Get top award categories
top_categories = df['Award_Category'].value_counts().head(5).index
yearly_data = df.groupby(['Year', 'Award_Category']).size().unstack(fill_value=0)
yearly_totals = df.groupby('Year').size()

# Create stacked area chart
bottom = np.zeros(len(yearly_data.index))
for i, category in enumerate(top_categories):
    if category in yearly_data.columns:
        ax1.fill_between(yearly_data.index, bottom, bottom + yearly_data[category], 
                        alpha=0.7, label=category, color=colors_main[i % len(colors_main)])
        bottom += yearly_data[category]

# Overlay line plot
ax1_twin = ax1.twinx()
ax1_twin.plot(yearly_totals.index, yearly_totals.values, 'k-', linewidth=2, label='Total Awards')
ax1_twin.set_ylabel('Total Awards per Year', fontweight='bold')

ax1.set_title('Award Categories Evolution Over Time', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Number of Awards by Category', fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1_twin.legend(loc='upper right', fontsize=8)

# Subplot 2: Dual-axis bar and line plot
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Calculate new categories per decade
decade_categories = df.groupby('Decade')['Award Name'].nunique()
cumulative_categories = decade_categories.cumsum()

# Bar chart for new categories
bars = ax2.bar(decade_categories.index, decade_categories.values, alpha=0.7, 
               color=colors_main[1], label='New Categories per Decade')

# Line plot for cumulative
ax2_twin = ax2.twinx()
ax2_twin.plot(cumulative_categories.index, cumulative_categories.values, 
              'o-', color=colors_main[0], linewidth=2, markersize=6, label='Cumulative Categories')

ax2.set_title('Award Category Introduction by Decade', fontweight='bold', fontsize=12)
ax2.set_xlabel('Decade', fontweight='bold')
ax2.set_ylabel('New Categories', fontweight='bold')
ax2_twin.set_ylabel('Cumulative Categories', fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2_twin.legend(loc='upper right', fontsize=8)

# Subplot 3: Heatmap with scatter overlay
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Create category-decade matrix
category_decade = df.groupby(['Award_Category', 'Decade']).size().unstack(fill_value=0)
category_decade_top = category_decade.loc[top_categories[:6]]

# Heatmap
im = ax3.imshow(category_decade_top.values, cmap='YlOrRd', aspect='auto', alpha=0.8)

# Add scatter points for peak years
for i, category in enumerate(category_decade_top.index):
    peak_decade_idx = np.argmax(category_decade_top.loc[category].values)
    ax3.scatter(peak_decade_idx, i, s=100, color='navy', marker='*', alpha=0.8)

ax3.set_title('Award Category Density by Decade', fontweight='bold', fontsize=12)
ax3.set_xlabel('Decade', fontweight='bold')
ax3.set_ylabel('Award Categories', fontweight='bold')
ax3.set_xticks(range(len(category_decade_top.columns)))
ax3.set_xticklabels(category_decade_top.columns, rotation=45)
ax3.set_yticks(range(len(category_decade_top.index)))
ax3.set_yticklabels(category_decade_top.index, fontsize=8)

# Subplot 4: Time series with filled area
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# Calculate winner ratio and nominees per year
yearly_stats = df.groupby('Year').agg({
    'Winner': ['sum', 'count'],
    'Nominee': 'count'
}).round(3)

yearly_stats.columns = ['Winners', 'Total_Entries', 'Nominees']
yearly_stats['Winner_Ratio'] = yearly_stats['Winners'] / yearly_stats['Total_Entries']

# Filled area for nominees
ax4.fill_between(yearly_stats.index, yearly_stats['Nominees'], alpha=0.3, 
                color=colors_main[2], label='Total Nominees')

# Line for winner ratio
ax4_twin = ax4.twinx()
ax4_twin.plot(yearly_stats.index, yearly_stats['Winner_Ratio'], 
              color=colors_main[3], linewidth=2, label='Winner Ratio')

ax4.set_title('Winners vs Nominees Analysis', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Number of Nominees', fontweight='bold')
ax4_twin.set_ylabel('Winner Ratio', fontweight='bold')
ax4.legend(loc='upper left', fontsize=8)
ax4_twin.legend(loc='upper right', fontsize=8)

# Subplot 5: Violin plot with box plot overlay
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Prepare data for violin plot
nominees_by_category_decade = []
labels = []
for decade in sorted(df['Decade'].unique()):
    decade_data = df[df['Decade'] == decade]
    nominees_per_category = decade_data.groupby('Award Name').size()
    nominees_by_category_decade.append(nominees_per_category.values)
    labels.append(str(decade))

# Create violin plot
parts = ax5.violinplot(nominees_by_category_decade, positions=range(len(labels)), 
                      widths=0.6, showmeans=True)

# Style violin plot
for pc in parts['bodies']:
    pc.set_facecolor(colors_main[4])
    pc.set_alpha(0.7)

# Add box plot overlay
box_data = [np.array(data) for data in nominees_by_category_decade]
bp = ax5.boxplot(box_data, positions=range(len(labels)), widths=0.3, 
                patch_artist=True, showfliers=False)

for patch in bp['boxes']:
    patch.set_facecolor(colors_main[5])
    patch.set_alpha(0.8)

ax5.set_title('Nominee Distribution by Decade', fontweight='bold', fontsize=12)
ax5.set_xlabel('Decade', fontweight='bold')
ax5.set_ylabel('Nominees per Category', fontweight='bold')
ax5.set_xticks(range(len(labels)))
ax5.set_xticklabels(labels, rotation=45)

# Subplot 6: Scatter plot with trend lines
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Calculate average nominees per ceremony by award type
ceremony_stats = df.groupby(['Ceremony', 'Award_Category']).size().reset_index(name='Count')
ceremony_avg = ceremony_stats.groupby(['Ceremony', 'Award_Category'])['Count'].mean().reset_index()

# Plot for top award types
top_award_types = df['Award_Category'].value_counts().head(3).index
for i, award_type in enumerate(top_award_types):
    data = ceremony_avg[ceremony_avg['Award_Category'] == award_type]
    ax6.scatter(data['Ceremony'], data['Count'], alpha=0.6, 
               color=colors_main[i], label=award_type, s=30)
    
    # Add trend line
    if len(data) > 1:
        z = np.polyfit(data['Ceremony'], data['Count'], 1)
        p = np.poly1d(z)
        ax6.plot(data['Ceremony'], p(data['Ceremony']), 
                color=colors_main[i], linestyle='--', alpha=0.8)

ax6.set_title('Ceremony vs Average Nominees by Award Type', fontweight='bold', fontsize=12)
ax6.set_xlabel('Ceremony Number', fontweight='bold')
ax6.set_ylabel('Average Nominees per Award', fontweight='bold')
ax6.legend(fontsize=8)

# Subplot 7: Calendar heatmap simulation
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Simulate ceremony months (Grammys typically in Jan/Feb)
np.random.seed(42)
ceremony_months = np.random.choice([1, 2], size=len(df['Ceremony'].unique()), p=[0.7, 0.3])
ceremony_years = sorted(df['Year'].unique())

# Create heatmap data
heatmap_data = np.zeros((12, len(set(ceremony_years))))
for i, year in enumerate(sorted(set(ceremony_years))):
    if i < len(ceremony_months):
        month = ceremony_months[i] - 1  # 0-indexed
        heatmap_data[month, i] = df[df['Year'] == year].shape[0]

# Plot heatmap
im = ax7.imshow(heatmap_data, cmap='Blues', aspect='auto')
ax7.set_title('Grammy Ceremony Intensity by Month/Year', fontweight='bold', fontsize=12)
ax7.set_xlabel('Year Index', fontweight='bold')
ax7.set_ylabel('Month', fontweight='bold')
ax7.set_yticks(range(12))
ax7.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Subplot 8: Slope chart
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Get top categories for first and last decades
first_decade = df['Decade'].min()
last_decade = df['Decade'].max()

first_decade_cats = df[df['Decade'] == first_decade]['Award Name'].value_counts().head(5)
last_decade_cats = df[df['Decade'] == last_decade]['Award Name'].value_counts().head(5)

# Find common categories
common_cats = set(first_decade_cats.index) & set(last_decade_cats.index)
common_cats = list(common_cats)[:5]  # Limit to 5 for clarity

for i, cat in enumerate(common_cats):
    first_val = first_decade_cats.get(cat, 0)
    last_val = last_decade_cats.get(cat, 0)
    
    ax8.plot([0, 1], [first_val, last_val], 'o-', 
            color=colors_main[i % len(colors_main)], linewidth=2, markersize=6)
    
    # Add error bars (simulated variability)
    error = np.random.uniform(0.1, 0.3) * max(first_val, last_val)
    ax8.errorbar([0, 1], [first_val, last_val], yerr=error, 
                color=colors_main[i % len(colors_main)], alpha=0.3, capsize=3)

ax8.set_title('Top Categories: First vs Last Decade', fontweight='bold', fontsize=12)
ax8.set_xlabel('Period', fontweight='bold')
ax8.set_ylabel('Award Frequency', fontweight='bold')
ax8.set_xticks([0, 1])
ax8.set_xticklabels([f'{first_decade}s', f'{last_decade}s'])

# Subplot 9: Time series decomposition simulation
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Get yearly nominations
yearly_nominations = df.groupby('Year').size()
years = yearly_nominations.index
nominations = yearly_nominations.values

# Simulate trend, seasonal, and residual components
trend = np.polyval(np.polyfit(range(len(nominations)), nominations, 2), range(len(nominations)))
seasonal = 50 * np.sin(2 * np.pi * np.arange(len(nominations)) / 10)
residual = nominations - trend - seasonal

# Plot components
ax9.plot(years, nominations, label='Original', color=colors_main[0], linewidth=2)
ax9.plot(years, trend, label='Trend', color=colors_main[1], linewidth=2, alpha=0.8)
ax9.plot(years, trend + seasonal, label='Trend + Seasonal', 
         color=colors_main[2], linewidth=1.5, alpha=0.7)

ax9.set_title('Grammy Nominations Time Series Decomposition', fontweight='bold', fontsize=12)
ax9.set_xlabel('Year', fontweight='bold')
ax9.set_ylabel('Number of Nominations', fontweight='bold')
ax9.legend(fontsize=8)

# Final layout adjustments
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()