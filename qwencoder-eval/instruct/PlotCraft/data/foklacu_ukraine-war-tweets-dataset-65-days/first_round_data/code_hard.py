import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load all datasets with error handling
datasets = {}
dataset_files = {
    'Ukraine war': 'Ukraine_war.csv',
    'Ukraine border': 'Ukraine_border.csv', 
    'Russian border Ukraine': 'Russian_border_Ukraine.csv',
    'Ukraine troops': 'Ukraine_troops.csv',
    'Russia invade': 'Russia_invade.csv',
    'Russian troops': 'Russian_troops.csv',
    'StandWithUkraine': 'StandWithUkraine.csv',
    'Ukraine NATO': 'Ukraine_nato.csv'
}

# Load datasets with sampling to reduce processing time
for name, file in dataset_files.items():
    try:
        df = pd.read_csv(file)
        # Sample data to reduce processing time - take every 10th row
        if len(df) > 10000:
            df = df.iloc[::10].reset_index(drop=True)
        datasets[name] = df
        print(f"Loaded {name}: {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Combine datasets
all_data = []
for search_term, df in datasets.items():
    df['search_term'] = search_term
    # Take only essential columns to reduce memory usage
    essential_cols = ['date', 'likeCount', 'retweetCount', 'replyCount', 'quoteCount', 
                     'lang', 'source', 'sourceLabel', 'hashtags', 'search_term']
    available_cols = [col for col in essential_cols if col in df.columns]
    df_subset = df[available_cols].copy()
    all_data.append(df_subset)

combined_df = pd.concat(all_data, ignore_index=True)

# Convert date and clean data
combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
combined_df = combined_df.dropna(subset=['date'])
combined_df['date_only'] = combined_df['date'].dt.date

# Fill missing values
numeric_cols = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount']
for col in numeric_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

# Create figure
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('white')

# Helper functions
def extract_source_info(source_str):
    if pd.isna(source_str):
        return 'Unknown'
    source_str = str(source_str).lower()
    if 'iphone' in source_str:
        return 'iPhone'
    elif 'android' in source_str:
        return 'Android'
    elif 'web' in source_str:
        return 'Web App'
    else:
        return 'Other'

# Prepare daily statistics
daily_stats = combined_df.groupby(['date_only', 'search_term']).agg({
    'likeCount': ['count', 'sum', 'mean'],
    'retweetCount': ['sum', 'mean'],
    'replyCount': ['sum', 'mean'],
    'quoteCount': ['sum', 'mean']
}).reset_index()

daily_stats.columns = ['date', 'search_term', 'tweet_count', 'total_likes', 'avg_likes', 
                      'total_retweets', 'avg_retweets', 'total_replies', 'avg_replies',
                      'total_quotes', 'avg_quotes']

daily_stats['total_engagement'] = daily_stats['total_likes'] + daily_stats['total_retweets'] + daily_stats['total_replies']

# Color palette
colors = plt.cm.Set3(np.linspace(0, 1, len(datasets.keys())))
color_map = dict(zip(datasets.keys(), colors))

# Row 1, Subplot 1: Daily tweet counts with engagement overlay
ax1 = plt.subplot(3, 3, 1)
for term in datasets.keys():
    term_data = daily_stats[daily_stats['search_term'] == term]
    if len(term_data) > 0:
        ax1.plot(term_data['date'], term_data['tweet_count'], 
                label=term[:15], color=color_map[term], linewidth=2, alpha=0.8)

ax1_twin = ax1.twinx()
total_daily_engagement = daily_stats.groupby('date')['total_engagement'].sum()
if len(total_daily_engagement) > 0:
    ax1_twin.bar(total_daily_engagement.index, total_daily_engagement.values, 
                alpha=0.3, color='gray', width=0.8)

ax1.set_title('Daily Tweet Volume by Search Term\nwith Total Engagement Overlay', fontweight='bold', fontsize=10)
ax1.set_xlabel('Date')
ax1.set_ylabel('Tweet Count', color='black')
ax1_twin.set_ylabel('Total Engagement', color='gray')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Source distribution over time
ax2 = plt.subplot(3, 3, 2)
combined_df['source_clean'] = combined_df['source'].apply(extract_source_info)

source_daily = combined_df.groupby(['date_only', 'source_clean']).size().unstack(fill_value=0)
if len(source_daily) > 0:
    source_daily_pct = source_daily.div(source_daily.sum(axis=1), axis=0) * 100
    
    # Get available sources
    available_sources = source_daily_pct.columns.tolist()
    source_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    # Create stacked area chart
    bottom = np.zeros(len(source_daily_pct))
    for i, source in enumerate(available_sources[:4]):  # Limit to 4 sources
        if source in source_daily_pct.columns:
            ax2.fill_between(source_daily_pct.index, bottom, 
                           bottom + source_daily_pct[source], 
                           label=source, alpha=0.7, color=source_colors[i % len(source_colors)])
            bottom += source_daily_pct[source]

ax2.set_title('Tweet Source Distribution Over Time\nwith Cumulative Stacking', fontweight='bold', fontsize=10)
ax2.set_xlabel('Date')
ax2.set_ylabel('Source Distribution (%)')
ax2.legend(loc='upper left', fontsize=7)
ax2.tick_params(axis='x', rotation=45, labelsize=8)

# Row 1, Subplot 3: Language distribution with sentiment proxy
ax3 = plt.subplot(3, 3, 3)
lang_daily = combined_df.groupby(['date_only', 'lang']).size().unstack(fill_value=0)
if len(lang_daily) > 0:
    top_langs = lang_daily.sum().nlargest(4).index
    
    for i, lang in enumerate(top_langs):
        if lang in lang_daily.columns:
            ax3.plot(lang_daily.index, lang_daily[lang], 
                    label=lang, linewidth=2, color=colors[i])

# Sentiment proxy overlay
sentiment_daily = combined_df.groupby('date_only')['likeCount'].mean()
ax3_twin = ax3.twinx()
ax3_twin.fill_between(sentiment_daily.index, sentiment_daily.values, 
                     alpha=0.3, color='orange', label='Avg Likes')

ax3.set_title('Daily Language Distribution\nwith Sentiment Indicator', fontweight='bold', fontsize=10)
ax3.set_xlabel('Date')
ax3.set_ylabel('Tweet Count by Language')
ax3_twin.set_ylabel('Average Likes', color='orange')
ax3.legend(loc='upper left', fontsize=7)
ax3_twin.legend(loc='upper right', fontsize=7)
ax3.tick_params(axis='x', rotation=45, labelsize=8)
ax3.grid(True, alpha=0.3)

# Row 2, Subplot 4: Multi-line engagement metrics
ax4 = plt.subplot(3, 3, 4)
engagement_daily = combined_df.groupby('date_only').agg({
    'likeCount': 'sum',
    'retweetCount': 'sum', 
    'replyCount': 'sum',
    'quoteCount': 'sum'
}).reset_index()

metrics = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount']
colors_eng = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, metric in enumerate(metrics):
    if metric in engagement_daily.columns:
        ax4.plot(engagement_daily['date_only'], engagement_daily[metric], 
                label=metric.replace('Count', 's'), color=colors_eng[i], linewidth=2)
        ax4.fill_between(engagement_daily['date_only'], 0, engagement_daily[metric], 
                        alpha=0.1, color=colors_eng[i])

ax4.set_title('Engagement Metrics Evolution\nwith Intensity Bands', fontweight='bold', fontsize=10)
ax4.set_xlabel('Date')
ax4.set_ylabel('Engagement Count')
ax4.legend(fontsize=7)
ax4.tick_params(axis='x', rotation=45, labelsize=8)
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Engagement scatter plot
ax5 = plt.subplot(3, 3, 5)
# Sample data for performance
sample_size = min(2000, len(combined_df))
sample_data = combined_df.sample(n=sample_size, random_state=42)
sample_data['total_engagement'] = (sample_data['likeCount'] + 
                                  sample_data['retweetCount'] + 
                                  sample_data['replyCount'])

# Create scatter plot
for i, term in enumerate(list(datasets.keys())[:5]):
    term_data = sample_data[sample_data['search_term'] == term]
    if len(term_data) > 0:
        ax5.scatter(term_data['likeCount'], term_data['total_engagement'],
                   alpha=0.6, s=20, label=term[:10], color=colors[i])

ax5.set_title('Like Count vs Total Engagement\nby Search Term', fontweight='bold', fontsize=10)
ax5.set_xlabel('Like Count')
ax5.set_ylabel('Total Engagement')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Daily engagement distributions
ax6 = plt.subplot(3, 3, 6)
# Sample dates for box plots
unique_dates = sorted(combined_df['date_only'].unique())
sample_dates = unique_dates[::max(1, len(unique_dates)//8)]  # Sample 8 dates

engagement_data = []
date_labels = []

for date in sample_dates:
    day_data = combined_df[combined_df['date_only'] == date]
    if len(day_data) > 0:
        day_engagement = day_data['likeCount'] + day_data['retweetCount'] + day_data['replyCount']
        # Sample to avoid too much data
        if len(day_engagement) > 100:
            day_engagement = day_engagement.sample(100, random_state=42)
        engagement_data.append(day_engagement.values)
        date_labels.append(date.strftime('%m-%d'))

if engagement_data:
    bp = ax6.boxplot(engagement_data, labels=date_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

ax6.set_title('Daily Engagement Distribution\nEvolution (Sampled)', fontweight='bold', fontsize=10)
ax6.set_xlabel('Date')
ax6.set_ylabel('Engagement Count')
ax6.tick_params(axis='x', rotation=45, labelsize=8)
ax6.grid(True, alpha=0.3)

# Row 3, Subplot 7: Hashtag trends (simplified)
ax7 = plt.subplot(3, 3, 7)
# Extract hashtags from string representation
def extract_hashtags_simple(hashtag_str):
    if pd.isna(hashtag_str) or hashtag_str == '' or hashtag_str == 'NaN':
        return []
    try:
        # Simple extraction - look for common hashtags
        hashtag_str = str(hashtag_str).lower()
        common_hashtags = ['ukraine', 'standwithukraine', 'russia', 'nato', 'nowar', 'stopthewar']
        found_hashtags = [tag for tag in common_hashtags if tag in hashtag_str]
        return found_hashtags
    except:
        return []

combined_df['hashtags_simple'] = combined_df['hashtags'].apply(extract_hashtags_simple)

# Count hashtag occurrences by date
hashtag_counts = {}
for _, row in combined_df.iterrows():
    date = row['date_only']
    hashtags = row['hashtags_simple']
    for hashtag in hashtags:
        if hashtag not in hashtag_counts:
            hashtag_counts[hashtag] = {}
        hashtag_counts[hashtag][date] = hashtag_counts[hashtag].get(date, 0) + 1

# Plot top hashtags over time
top_hashtags = sorted(hashtag_counts.keys(), 
                     key=lambda x: sum(hashtag_counts[x].values()), reverse=True)[:5]

for i, hashtag in enumerate(top_hashtags):
    dates = sorted(hashtag_counts[hashtag].keys())
    counts = [hashtag_counts[hashtag][date] for date in dates]
    ax7.plot(dates, counts, label=f'#{hashtag}', linewidth=2, color=colors[i])

ax7.set_title('Top Hashtag Trends Over Time', fontweight='bold', fontsize=10)
ax7.set_xlabel('Date')
ax7.set_ylabel('Hashtag Count')
ax7.legend(fontsize=7)
ax7.tick_params(axis='x', rotation=45, labelsize=8)
ax7.grid(True, alpha=0.3)

# Row 3, Subplot 8: User activity evolution
ax8 = plt.subplot(3, 3, 8)
# Daily user activity metrics
daily_activity = combined_df.groupby('date_only').agg({
    'likeCount': 'mean',
    'retweetCount': 'mean',
    'replyCount': 'mean'
}).reset_index()

ax8.plot(daily_activity['date_only'], daily_activity['likeCount'], 
         linewidth=3, color='blue', label='Avg Likes')
ax8.plot(daily_activity['date_only'], daily_activity['retweetCount'], 
         linewidth=3, color='green', label='Avg Retweets')
ax8.plot(daily_activity['date_only'], daily_activity['replyCount'], 
         linewidth=3, color='red', label='Avg Replies')

ax8.fill_between(daily_activity['date_only'], daily_activity['likeCount'], 
                alpha=0.3, color='blue')

ax8.set_title('User Engagement Evolution\nNetwork Activity Patterns', fontweight='bold', fontsize=10)
ax8.set_xlabel('Date')
ax8.set_ylabel('Average Engagement')
ax8.legend(fontsize=7)
ax8.tick_params(axis='x', rotation=45, labelsize=8)
ax8.grid(True, alpha=0.3)

# Row 3, Subplot 9: Content type analysis
ax9 = plt.subplot(3, 3, 9)
# Analyze content patterns
combined_df['content_length'] = combined_df.get('content', '').astype(str).str.len()
combined_df['has_url'] = combined_df.get('outlinks', '').notna() & (combined_df.get('outlinks', '') != 'NaN')

# Daily content metrics
content_daily = combined_df.groupby('date_only').agg({
    'content_length': 'mean',
    'has_url': 'mean'
}).reset_index()

ax9.plot(content_daily['date_only'], content_daily['content_length'], 
         linewidth=3, color='purple', label='Avg Content Length')

ax9_twin = ax9.twinx()
ax9_twin.plot(content_daily['date_only'], content_daily['has_url'] * 100, 
              linewidth=3, color='orange', label='URL Rate %')

# Add bars for tweet volume
tweet_volume = combined_df.groupby('date_only').size()
ax9_twin.bar(tweet_volume.index, tweet_volume.values, 
             alpha=0.3, color='gray', width=0.8, label='Tweet Volume')

ax9.set_title('Content Characteristics Evolution\nLength, URLs, and Volume', fontweight='bold', fontsize=10)
ax9.set_xlabel('Date')
ax9.set_ylabel('Average Content Length', color='purple')
ax9_twin.set_ylabel('URL Rate (%) / Volume', color='orange')
ax9.legend(loc='upper left', fontsize=7)
ax9_twin.legend(loc='upper right', fontsize=7)
ax9.tick_params(axis='x', rotation=45, labelsize=8)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=1.5)
plt.savefig('ukraine_war_discourse_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization completed successfully!")
print(f"Total tweets analyzed: {len(combined_df):,}")
print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
print(f"Search terms: {list(datasets.keys())}")