import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Load data
df = pd.read_csv('bangladeshi_all_engish_newspapers_daily_news_combined_dataset.csv')

# Data preprocessing
df['publish_date'] = pd.to_datetime(df['publish_date'])
df['news_collection_time'] = pd.to_datetime(df['news_collection_time'])
df['text_length'] = df['text'].str.len()

# Extract day of week and hour from news_collection_time
df['day_of_week'] = df['news_collection_time'].dt.day_name()
df['hour'] = df['news_collection_time'].dt.hour

# Create figure with white background and professional styling
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Improved color palette for publishers with better contrast
publishers = df['publisher'].unique()
distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
publisher_colors = dict(zip(publishers, distinct_colors[:len(publishers)]))

# 1. Top-left: Line chart with markers showing daily article counts
daily_counts = df.groupby(['publish_date', 'publisher']).size().unstack(fill_value=0)
for i, publisher in enumerate(publishers):
    if publisher in daily_counts.columns:
        line_style = ['-', '--', '-.', ':'][i % 4]
        ax1.plot(daily_counts.index, daily_counts[publisher], 
                marker='o', markersize=4, linewidth=2, linestyle=line_style,
                color=publisher_colors[publisher], label=publisher, alpha=0.8)

ax1.set_title('Daily Article Publication Trends by Publisher', fontweight='bold', fontsize=12, pad=15)
ax1.set_xlabel('Publication Date', fontweight='bold')
ax1.set_ylabel('Number of Articles per Day', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Top-right: Stacked area chart showing cumulative proportion
daily_counts_cumsum = daily_counts.cumsum()
daily_proportions = daily_counts_cumsum.div(daily_counts_cumsum.sum(axis=1), axis=0)

ax2.stackplot(daily_proportions.index, 
              *[daily_proportions[col] for col in daily_proportions.columns],
              labels=daily_proportions.columns,
              colors=[publisher_colors[pub] for pub in daily_proportions.columns],
              alpha=0.8)

ax2.set_title('Cumulative Publication Proportion by Publisher', fontweight='bold', fontsize=12, pad=15)
ax2.set_xlabel('Publication Date', fontweight='bold')
ax2.set_ylabel('Cumulative Proportion', fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylim(0, 1)

# 3. Bottom-left: Bar chart with total articles and secondary axis for avg text length
publisher_stats = df.groupby('publisher').agg({
    'title': 'count',
    'text_length': 'mean'
}).round(0)

bars = ax3.bar(range(len(publisher_stats)), publisher_stats['title'], 
               color=[publisher_colors[pub] for pub in publisher_stats.index],
               alpha=0.8, edgecolor='black', linewidth=0.5)

ax3.set_title('Total Articles vs Average Article Length by Publisher', fontweight='bold', fontsize=12, pad=15)
ax3.set_xlabel('Publisher', fontweight='bold')
ax3.set_ylabel('Total Article Count', fontweight='bold', color='black')
ax3.set_xticks(range(len(publisher_stats)))
ax3.set_xticklabels(publisher_stats.index, rotation=45, ha='right')

# Secondary y-axis for average text length with improved color
ax3_twin = ax3.twinx()
line = ax3_twin.plot(range(len(publisher_stats)), publisher_stats['text_length'], 
                     color='#8B4513', marker='D', markersize=8, linewidth=3, 
                     label='Average Article Length')
ax3_twin.set_ylabel('Average Article Length (characters)', fontweight='bold', color='#8B4513')
ax3_twin.tick_params(axis='y', labelcolor='#8B4513')

# Add legend for the line plot
ax3_twin.legend(loc='upper right')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 4. Bottom-right: Improved heatmap of publication activity by day and hour
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create complete hour-day matrix with all possible combinations
all_hours = range(24)
hour_day_matrix = np.zeros((24, 7))

# Fill the matrix with actual data
for hour in all_hours:
    for day_idx, day in enumerate(day_order):
        count = len(df[(df['hour'] == hour) & (df['day_of_week'] == day)])
        hour_day_matrix[hour, day_idx] = count

# Create proper heatmap with seaborn for better visualization
sns.heatmap(hour_day_matrix, 
            xticklabels=day_order, 
            yticklabels=all_hours,
            cmap='YlOrRd', 
            annot=False, 
            fmt='d',
            cbar_kws={'label': 'Article Frequency', 'shrink': 0.8},
            ax=ax4)

ax4.set_title('Publication Activity Heatmap\n(Hour vs Day of Week)', fontweight='bold', fontsize=12, pad=15)
ax4.set_xlabel('Day of Week', fontweight='bold')
ax4.set_ylabel('Hour of Day', fontweight='bold')
ax4.set_xticklabels(day_order, rotation=45, ha='right')

# Set y-axis to show hours from 0 at bottom to 23 at top
ax4.invert_yaxis()
ax4.set_yticks(range(0, 24, 4))
ax4.set_yticklabels(range(0, 24, 4))

# Find peak activity and add annotation in a better position
max_activity = int(hour_day_matrix.max())
max_pos = np.unravel_index(hour_day_matrix.argmax(), hour_day_matrix.shape)
peak_hour = max_pos[0]
peak_day = day_order[max_pos[1]]

# Position annotation in upper left corner to avoid overlap
ax4.annotate(f'Peak Activity:\n{max_activity} articles\n{peak_day} at {peak_hour}:00', 
             xy=(0.02, 0.98), xycoords='axes fraction',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black'),
             fontweight='bold', fontsize=10,
             verticalalignment='top', horizontalalignment='left')

# Overall layout adjustment
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.4)
plt.show()