import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load datasets with error handling and sampling for performance
def load_and_sample_data(filename, sample_size=5000):
    try:
        df = pd.read_csv(filename)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return df
    except:
        return pd.DataFrame()

datasets = {
    'Ukraine_war': load_and_sample_data('Ukraine_war.csv'),
    'Ukraine_border': load_and_sample_data('Ukraine_border.csv'),
    'Russian_border_Ukraine': load_and_sample_data('Russian_border_Ukraine.csv'),
    'Ukraine_troops': load_and_sample_data('Ukraine_troops.csv'),
    'Russia_invade': load_and_sample_data('Russia_invade.csv'),
    'Russian_troops': load_and_sample_data('Russian_troops.csv'),
    'StandWithUkraine': load_and_sample_data('StandWithUkraine.csv'),
    'Ukraine_nato': load_and_sample_data('Ukraine_nato.csv')
}

# Filter out empty datasets
datasets = {k: v for k, v in datasets.items() if not v.empty}

# Combine datasets with search term labels
all_data = []
for search_term, df in datasets.items():
    df_copy = df.copy()
    df_copy['search_term'] = search_term
    all_data.append(df_copy)

if not all_data:
    print("No data loaded successfully")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Convert date column to datetime with error handling
try:
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    combined_df = combined_df.dropna(subset=['date'])
    combined_df['date_only'] = combined_df['date'].dt.date
except:
    print("Error processing dates")
    exit()

# Fill missing values
numeric_cols = ['likeCount', 'retweetCount', 'replyCount']
for col in numeric_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Top-left: Stacked area chart with cumulative engagement line
try:
    daily_counts = combined_df.groupby(['date_only', 'search_term']).size().unstack(fill_value=0)
    daily_engagement = combined_df.groupby('date_only')[numeric_cols].sum()
    daily_engagement['total_engagement'] = daily_engagement.sum(axis=1)
    cumulative_engagement = daily_engagement['total_engagement'].cumsum()

    # Limit to top 5 search terms for readability
    top_terms = daily_counts.sum().nlargest(5).index
    daily_counts_top = daily_counts[top_terms]
    
    # Create stacked area chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_terms)))
    ax1.stackplot(daily_counts_top.index, *[daily_counts_top[col] for col in daily_counts_top.columns], 
                  alpha=0.7, labels=daily_counts_top.columns, colors=colors)

    # Overlay cumulative engagement line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(cumulative_engagement.index, cumulative_engagement.values, 
                  color='red', linewidth=3, label='Cumulative Engagement')
    ax1_twin.set_ylabel('Cumulative Engagement', fontweight='bold', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    ax1.set_title('Daily Tweet Volume by Search Term\nwith Cumulative Engagement', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Daily Tweet Count', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1_twin.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45)
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1: {str(e)[:50]}...', 
             transform=ax1.transAxes, ha='center', va='center')

# Top-right: Dual-axis time series (sentiment vs media proportion)
try:
    # Calculate sentiment score (like-to-reply ratio) with safety checks
    daily_sentiment = combined_df.groupby('date_only').apply(
        lambda x: (x['likeCount'].sum() + 1) / (x['replyCount'].sum() + 1)
    ).reset_index()
    daily_sentiment.columns = ['date_only', 'sentiment_score']

    # Calculate media proportion
    combined_df['has_media'] = combined_df['media'].notna() & (combined_df['media'] != 'NaN')
    daily_media_prop = combined_df.groupby('date_only')['has_media'].mean()

    # Plot sentiment line
    ax2.plot(daily_sentiment['date_only'], daily_sentiment['sentiment_score'], 
             color='blue', linewidth=2, label='Sentiment Score')
    ax2.set_ylabel('Sentiment Score (Like/Reply)', fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Plot media proportion bars on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.bar(daily_media_prop.index, daily_media_prop.values, 
                 alpha=0.6, color='orange', width=0.8, label='Media Proportion')
    ax2_twin.set_ylabel('Media Attachment Rate', fontweight='bold', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    ax2.set_title('Daily Sentiment vs Content Richness', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2: {str(e)[:50]}...', 
             transform=ax2.transAxes, ha='center', va='center')

# Bottom-left: Multi-line hashtag counts with heatmap background
try:
    def safe_count_hashtags(hashtag_str):
        if pd.isna(hashtag_str) or hashtag_str == 'NaN' or hashtag_str == '':
            return 0
        try:
            if isinstance(hashtag_str, str) and hashtag_str.startswith('['):
                hashtags = eval(hashtag_str)
                return len(set(hashtags)) if isinstance(hashtags, list) else 0
            return 0
        except:
            return 0

    # Calculate daily unique hashtag counts by search term
    hashtag_counts = {}
    dates = sorted(combined_df['date_only'].unique())
    
    for search_term in list(datasets.keys())[:5]:  # Limit to 5 terms
        term_data = combined_df[combined_df['search_term'] == search_term]
        daily_hashtags = []
        for date in dates:
            day_data = term_data[term_data['date_only'] == date]
            total_hashtags = day_data['hashtags'].apply(safe_count_hashtags).sum()
            daily_hashtags.append(total_hashtags)
        hashtag_counts[search_term] = daily_hashtags

    # Create background heatmap
    if hashtag_counts:
        heatmap_data = np.array(list(hashtag_counts.values()))
        if heatmap_data.size > 0:
            im = ax3.imshow(heatmap_data, aspect='auto', alpha=0.3, cmap='YlOrRd', 
                           extent=[0, len(dates), 0, len(hashtag_counts)])

        # Plot hashtag count lines
        colors = plt.cm.Set2(np.linspace(0, 1, len(hashtag_counts)))
        for i, (search_term, counts) in enumerate(hashtag_counts.items()):
            ax3.plot(range(len(counts)), counts, color=colors[i], 
                    linewidth=2, label=search_term[:15], alpha=0.8)

    ax3.set_title('Daily Hashtag Usage by Search Term', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Days Since Start', fontweight='bold')
    ax3.set_ylabel('Hashtag Count', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3: {str(e)[:50]}...', 
             transform=ax3.transAxes, ha='center', va='center')

# Bottom-right: Reply percentage line + volume vs follower scatter
try:
    # Calculate daily reply percentage
    combined_df['is_reply'] = combined_df['inReplyToTweetId'].notna()
    daily_reply_pct = combined_df.groupby('date_only')['is_reply'].mean() * 100

    # Extract follower counts safely
    def safe_extract_followers(user_str):
        try:
            if pd.isna(user_str) or user_str == 'NaN':
                return np.nan
            if isinstance(user_str, str) and 'followersCount' in user_str:
                # Simple regex-like extraction
                start = user_str.find("'followersCount': ") + 18
                if start > 17:
                    end = user_str.find(',', start)
                    if end == -1:
                        end = user_str.find('}', start)
                    if end > start:
                        return float(user_str[start:end])
            return np.nan
        except:
            return np.nan

    combined_df['follower_count'] = combined_df['user'].apply(safe_extract_followers)

    # Daily aggregates for scatter plot
    daily_volume = combined_df.groupby('date_only').size()
    daily_avg_followers = combined_df.groupby('date_only')['follower_count'].mean()

    # Remove NaN values
    valid_data = daily_avg_followers.dropna()
    valid_volume = daily_volume[valid_data.index]

    # Plot reply percentage line
    ax4.plot(daily_reply_pct.index, daily_reply_pct.values, 
             color='green', linewidth=3, label='Reply Percentage')
    ax4.set_ylabel('Reply Percentage (%)', fontweight='bold', color='green')
    ax4.tick_params(axis='y', labelcolor='green')

    # Create secondary axis for scatter plot
    ax4_twin = ax4.twinx()
    if len(valid_data) > 0:
        scatter = ax4_twin.scatter(valid_volume.values, valid_data.values, 
                                  c=range(len(valid_data)), cmap='viridis', 
                                  alpha=0.7, s=60, label='Volume vs Followers')
    ax4_twin.set_ylabel('Avg User Followers', fontweight='bold', color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')

    ax4.set_title('Conversation Levels &\nUser Influence Evolution', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4: {str(e)[:50]}...', 
             transform=ax4.transAxes, ha='center', va='center')

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.4)

# Add main title
fig.suptitle('Ukraine War Twitter Discourse Analysis: Temporal Evolution Across Multiple Dimensions', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('ukraine_war_discourse_analysis.png', dpi=300, bbox_inches='tight')
plt.show()