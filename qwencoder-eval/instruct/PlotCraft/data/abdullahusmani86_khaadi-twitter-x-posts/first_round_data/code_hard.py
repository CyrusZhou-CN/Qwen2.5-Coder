import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
from scipy import stats
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('tweets.csv')

# Handle missing values in text column
df = df.dropna(subset=['text'])  # Remove rows with NaN text
df['text'] = df['text'].astype(str)  # Ensure all text is string type

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['hashtag_count'] = df['text'].str.count('#')

# Extract additional time features
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = np.random.randint(8, 22, len(df))  # Simulated posting hours
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week
df['day_of_year'] = df['date'].dt.dayofyear

# Content categorization based on keywords
def categorize_content(text):
    if pd.isna(text) or not isinstance(text, str):
        return 'Other'
    
    text_lower = text.lower()
    if any(word in text_lower for word in ['collection', 'fabric', 'winter', 'formal']):
        return 'Product Launch'
    elif any(word in text_lower for word in ['style', 'look', 'wear', 'outfit']):
        return 'Style Content'
    elif any(word in text_lower for word in ['photographer', 'artist', 'creator']):
        return 'Influencer Collab'
    else:
        return 'Brand Story'

df['content_category'] = df['text'].apply(categorize_content)

# Sentiment analysis (simplified)
def simple_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    
    positive_words = ['beautiful', 'perfect', 'amazing', 'vibrant', 'exquisite', 'radiant']
    negative_words = ['busy', 'fading', 'chilly']
    
    pos_count = sum(1 for word in positive_words if word in text.lower())
    neg_count = sum(1 for word in negative_words if word in text.lower())
    
    if pos_count > neg_count:
        return 1  # Positive
    elif neg_count > pos_count:
        return -1  # Negative
    else:
        return 0  # Neutral

df['sentiment'] = df['text'].apply(simple_sentiment)

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Subplot 1: Daily tweet frequency with rolling average
ax1 = plt.subplot(3, 3, 1)
daily_counts = df.groupby(df['date'].dt.date).size()
daily_counts.index = pd.to_datetime(daily_counts.index)
rolling_avg = daily_counts.rolling(window=7, center=True).mean()

bars = ax1.bar(daily_counts.index, daily_counts.values, alpha=0.6, color='#2E86AB', width=0.8)
line = ax1.plot(rolling_avg.index, rolling_avg.values, color='#A23B72', linewidth=3, label='7-day Rolling Avg')
ax1.set_title('Daily Tweet Frequency with Rolling Average', fontweight='bold', fontsize=12)
ax1.set_ylabel('Tweet Count')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Subplot 2: Weekly patterns with cumulative percentage
ax2 = plt.subplot(3, 3, 2)
weekly_counts = df['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
cumulative_pct = weekly_counts.cumsum() / weekly_counts.sum() * 100

bars2 = ax2.bar(range(len(weekly_counts)), weekly_counts.values, alpha=0.7, color='#F18F01')
ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(range(len(weekly_counts)), cumulative_pct.values, color='#C73E1D', marker='o', linewidth=3, markersize=8)
ax2.set_title('Weekly Posting Patterns with Cumulative Distribution', fontweight='bold', fontsize=12)
ax2.set_ylabel('Tweet Count')
ax2_twin.set_ylabel('Cumulative %')
ax2.set_xticks(range(len(weekly_counts)))
ax2.set_xticklabels([day[:3] for day in weekly_counts.index], rotation=45)

# Subplot 3: Monthly trends with engagement heatmap
ax3 = plt.subplot(3, 3, 3)
monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
engagement_proxy = df.groupby(df['date'].dt.to_period('M'))['hashtag_count'].mean()

bars3 = ax3.bar(range(len(monthly_counts)), monthly_counts.values, alpha=0.6, color='#3E92CC')
ax3_twin = ax3.twinx()

# Create heatmap overlay
if len(engagement_proxy) > 0:
    heatmap_data = engagement_proxy.values.reshape(1, -1)
    im = ax3_twin.imshow(heatmap_data, aspect='auto', cmap='Reds', alpha=0.7, extent=[0, len(monthly_counts), 0, 1])
ax3.set_title('Monthly Trends with Engagement Intensity', fontweight='bold', fontsize=12)
ax3.set_ylabel('Tweet Volume')
ax3_twin.set_ylabel('Engagement Intensity')
ax3.set_xticks(range(len(monthly_counts)))
ax3.set_xticklabels([str(period) for period in monthly_counts.index], rotation=45)

# Subplot 4: Tweet length distribution with KDE and box plot
ax4 = plt.subplot(3, 3, 4)
hist_data = ax4.hist(df['text_length'], bins=30, alpha=0.6, color='#7209B7', density=True, label='Histogram')
kde_x = np.linspace(df['text_length'].min(), df['text_length'].max(), 100)
kde_y = stats.gaussian_kde(df['text_length'])(kde_x)
ax4.plot(kde_x, kde_y, color='#F72585', linewidth=3, label='KDE')

# Add box plot on top
box_data = ax4.boxplot(df['text_length'], vert=False, positions=[kde_y.max() * 0.8], 
                       widths=[kde_y.max() * 0.2], patch_artist=True, 
                       boxprops=dict(facecolor='#4CC9F0', alpha=0.7))
ax4.set_title('Tweet Length Distribution with Statistical Summary', fontweight='bold', fontsize=12)
ax4.set_xlabel('Character Count')
ax4.set_ylabel('Density')
ax4.legend()

# Subplot 5: Hashtag usage over time (stacked area)
ax5 = plt.subplot(3, 3, 5)
hashtag_timeline = df.groupby([df['date'].dt.to_period('M'), 'hashtag_count']).size().unstack(fill_value=0)
if not hashtag_timeline.empty:
    hashtag_timeline.plot(kind='area', stacked=True, ax=ax5, alpha=0.7, 
                         colormap='viridis')
ax5.set_title('Hashtag Usage Evolution Over Time', fontweight='bold', fontsize=12)
ax5.set_ylabel('Tweet Count')
ax5.set_xlabel('Month')
ax5.legend(title='Hashtag Count', bbox_to_anchor=(1.05, 1), loc='upper left')

# Subplot 6: Content category evolution
ax6 = plt.subplot(3, 3, 6)
category_timeline = df.groupby([df['date'].dt.to_period('M'), 'content_category']).size().unstack(fill_value=0)
if not category_timeline.empty:
    category_pct = category_timeline.div(category_timeline.sum(axis=1), axis=0) * 100
    bars6 = category_pct.plot(kind='bar', stacked=True, ax=ax6, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax6.set_title('Content Category Evolution (Percentage)', fontweight='bold', fontsize=12)
ax6.set_ylabel('Percentage')
ax6.set_xlabel('Month')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.tick_params(axis='x', rotation=45)

# Subplot 7: Seasonal behavior (polar chart)
ax7 = plt.subplot(3, 3, 7, projection='polar')
seasonal_data = df.groupby(df['date'].dt.month).size()
months = np.arange(1, 13)
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

# Extend data to close the circle
seasonal_values = [seasonal_data.get(month, 0) for month in months]
seasonal_values += seasonal_values[:1]
angles_extended = np.concatenate([angles, [angles[0]]])

ax7.plot(angles_extended, seasonal_values, 'o-', linewidth=3, color='#E74C3C', markersize=8)
ax7.fill(angles_extended, seasonal_values, alpha=0.25, color='#E74C3C')
ax7.set_xticks(angles)
ax7.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax7.set_title('Seasonal Posting Behavior', fontweight='bold', fontsize=12, pad=20)

# Subplot 8: Day-of-week vs hour heatmap with marginals
ax8 = plt.subplot(3, 3, 8)
pivot_data = df.pivot_table(values='text_length', index='day_of_week', 
                           columns='hour', aggfunc='count', fill_value=0)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data.reindex(days_order)

if not pivot_data.empty:
    sns.heatmap(pivot_data, ax=ax8, cmap='YlOrRd', cbar_kws={'label': 'Tweet Count'})
ax8.set_title('Posting Patterns: Day vs Hour Heatmap', fontweight='bold', fontsize=12)
ax8.set_xlabel('Hour of Day')
ax8.set_ylabel('Day of Week')

# Subplot 9: Sentiment timeline with distributions
ax9 = plt.subplot(3, 3, 9)
sentiment_timeline = df.groupby(df['date'].dt.to_period('M'))['sentiment'].mean()
if not sentiment_timeline.empty:
    sentiment_timeline.plot(ax=ax9, color='#2ECC71', linewidth=3, marker='o', markersize=8)

# Add violin plots for sentiment distribution by quarter - Fixed version
quarters = df['date'].dt.to_period('Q').unique()
violin_data = []
positions = []
for i, quarter in enumerate(quarters):
    quarter_data = df[df['date'].dt.to_period('Q') == quarter]['sentiment']
    if len(quarter_data) > 0:
        violin_data.append(quarter_data.values)
        positions.append(i)

if violin_data and len(violin_data) > 0:
    ax9_twin = ax9.twinx()
    # Fixed: Remove alpha parameter from violinplot and set it manually
    parts = ax9_twin.violinplot(violin_data, positions=positions, widths=0.5)
    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.6)  # Set alpha on the patch objects instead

ax9.set_title('Content Sentiment Timeline with Distributions', fontweight='bold', fontsize=12)
ax9.set_ylabel('Average Sentiment')
ax9.grid(True, alpha=0.3)
ax9.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.savefig('khaadi_twitter_analysis.png', dpi=300, bbox_inches='tight')
plt.show()