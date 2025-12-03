import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import Counter

# Load and preprocess data
df = pd.read_csv('bangladeshi_all_engish_newspapers_daily_news_combined_dataset.csv')

# Convert date columns to datetime, handle missing values
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
df['news_collection_time'] = pd.to_datetime(df['news_collection_time'], errors='coerce')

# Use news_collection_time as fallback for missing publish_date
df['effective_date'] = df['publish_date'].fillna(df['news_collection_time'])
df = df.dropna(subset=['effective_date'])

# Extract temporal features
df['date_only'] = df['effective_date'].dt.date
df['hour'] = df['effective_date'].dt.hour
df['day_of_week'] = df['effective_date'].dt.day_name()
df['weekday_num'] = df['effective_date'].dt.weekday

# Create consistent color palette for publishers
publishers = df['publisher'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(publishers)))
publisher_colors = dict(zip(publishers, colors))

# Set up the figure with white background
fig = plt.figure(figsize=(16, 12), facecolor='white')
fig.suptitle('Temporal Patterns in Bangladesh\'s English News Landscape', 
             fontsize=20, fontweight='bold', y=0.95)

# Top-left: Dual-axis time series with daily volume and rolling average
ax1 = plt.subplot(2, 2, 1)
daily_counts = df.groupby('date_only').size().reset_index(name='count')
daily_counts['date_only'] = pd.to_datetime(daily_counts['date_only'])
daily_counts = daily_counts.sort_values('date_only')

# Bar chart for daily volume
bars = ax1.bar(daily_counts['date_only'], daily_counts['count'], 
               alpha=0.6, color='lightblue', width=0.8, label='Daily Volume')

# 7-day rolling average line
daily_counts['rolling_avg'] = daily_counts['count'].rolling(window=7, center=True).mean()
ax1_twin = ax1.twinx()
line = ax1_twin.plot(daily_counts['date_only'], daily_counts['rolling_avg'], 
                     color='darkred', linewidth=3, label='7-day Rolling Average')

ax1.set_title('Daily News Volume with Rolling Average', fontweight='bold', fontsize=14)
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('Daily News Count', fontweight='bold', color='blue')
ax1_twin.set_ylabel('Rolling Average', fontweight='bold', color='darkred')
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.tick_params(axis='y', labelcolor='darkred')
ax1.grid(True, alpha=0.3)

# Top-right: Stacked area chart with diversity index
ax2 = plt.subplot(2, 2, 2)
daily_publisher = df.groupby(['date_only', 'publisher']).size().unstack(fill_value=0)
daily_publisher.index = pd.to_datetime(daily_publisher.index)
daily_publisher = daily_publisher.sort_index()

# Create cumulative data for stacked area
cumulative_data = daily_publisher.cumsum(axis=1)
bottom = np.zeros(len(daily_publisher))

for i, publisher in enumerate(daily_publisher.columns):
    ax2.fill_between(daily_publisher.index, bottom, cumulative_data[publisher], 
                     color=publisher_colors[publisher], alpha=0.7, label=publisher)
    bottom = cumulative_data[publisher]

# Calculate and overlay diversity index (inverse Herfindahl index)
diversity_scores = []
for _, row in daily_publisher.iterrows():
    total = row.sum()
    if total > 0:
        proportions = row / total
        herfindahl = (proportions ** 2).sum()
        diversity = 1 / herfindahl if herfindahl > 0 else 1
    else:
        diversity = 1
    diversity_scores.append(diversity)

ax2_twin = ax2.twinx()
ax2_twin.plot(daily_publisher.index, diversity_scores, 
              color='black', linewidth=3, marker='o', markersize=4, 
              label='Diversity Index')

ax2.set_title('Publisher Contribution & Source Diversity', fontweight='bold', fontsize=14)
ax2.set_xlabel('Date', fontweight='bold')
ax2.set_ylabel('Cumulative News Count', fontweight='bold')
ax2_twin.set_ylabel('Diversity Index', fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)

# Bottom-left: Calendar heatmap with marginal histograms
ax3 = plt.subplot(2, 2, 3)

# Create calendar heatmap data
df['date_str'] = df['effective_date'].dt.strftime('%Y-%m-%d')
daily_intensity = df.groupby('date_str').size()

# Create a grid for the heatmap
date_range = pd.date_range(start=df['effective_date'].min().date(), 
                          end=df['effective_date'].max().date(), freq='D')
intensity_matrix = []
date_labels = []

for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    intensity = daily_intensity.get(date_str, 0)
    intensity_matrix.append(intensity)
    date_labels.append(date_str)

# Reshape for heatmap visualization
weeks = len(intensity_matrix) // 7 + 1
heatmap_data = np.zeros((7, weeks))
for i, intensity in enumerate(intensity_matrix):
    week = i // 7
    day = i % 7
    if week < weeks:
        heatmap_data[day, week] = intensity

im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax3.set_title('News Publication Intensity Calendar', fontweight='bold', fontsize=14)
ax3.set_ylabel('Day of Week', fontweight='bold')
ax3.set_xlabel('Week', fontweight='bold')
ax3.set_yticks(range(7))
ax3.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Add marginal histograms
# Day of week distribution (right margin)
day_counts = df.groupby('weekday_num').size()
ax3_right = ax3.twinx()
ax3_right.barh(range(7), [day_counts.get(i, 0) for i in range(7)], 
               alpha=0.6, color='orange')
ax3_right.set_ylabel('Articles per Day of Week', fontweight='bold')

# Bottom-right: Slope chart with error bars and scatter points
ax4 = plt.subplot(2, 2, 4)

# Calculate earliest and latest dates for each publisher
publisher_stats = []
for publisher in publishers:
    pub_data = df[df['publisher'] == publisher]
    if len(pub_data) > 1:
        daily_counts_pub = pub_data.groupby('date_only').size()
        earliest_date = daily_counts_pub.index.min()
        latest_date = daily_counts_pub.index.max()
        earliest_count = daily_counts_pub.loc[earliest_date]
        latest_count = daily_counts_pub.loc[latest_date]
        std_dev = daily_counts_pub.std()
        
        publisher_stats.append({
            'publisher': publisher,
            'earliest_count': earliest_count,
            'latest_count': latest_count,
            'std_dev': std_dev,
            'daily_counts': daily_counts_pub.values
        })

# Create slope chart
y_positions = range(len(publisher_stats))
for i, stats in enumerate(publisher_stats):
    # Slope line
    ax4.plot([0, 1], [stats['earliest_count'], stats['latest_count']], 
             color=publisher_colors[stats['publisher']], linewidth=2, alpha=0.7)
    
    # Error bars
    ax4.errorbar([0, 1], [stats['earliest_count'], stats['latest_count']], 
                yerr=[stats['std_dev'], stats['std_dev']], 
                color=publisher_colors[stats['publisher']], alpha=0.5, capsize=5)
    
    # Scatter points for individual daily totals
    x_scatter = np.random.normal(0.5, 0.05, len(stats['daily_counts']))
    ax4.scatter(x_scatter, stats['daily_counts'], 
               color=publisher_colors[stats['publisher']], alpha=0.4, s=20)
    
    # Publisher labels
    ax4.text(-0.1, stats['earliest_count'], stats['publisher'], 
             ha='right', va='center', fontsize=10, fontweight='bold')

ax4.set_title('Publisher Volume: Early vs Late Period with Volatility', 
              fontweight='bold', fontsize=14)
ax4.set_xlim(-0.3, 1.3)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Earliest Period', 'Latest Period'], fontweight='bold')
ax4.set_ylabel('Daily News Volume', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()