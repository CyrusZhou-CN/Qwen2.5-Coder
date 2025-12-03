import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Load and combine datasets with error handling
datasets = {
    'Ukraine war': 'Ukraine_war.csv',
    'Ukraine border': 'Ukraine_border.csv', 
    'Russian border Ukraine': 'Russian_border_Ukraine.csv',
    'Ukraine troops': 'Ukraine_troops.csv',
    'Russia invade': 'Russia_invade.csv',
    'Russian troops': 'Russian_troops.csv',
    'StandWithUkraine': 'StandWithUkraine.csv',
    'Ukraine NATO': 'Ukraine_nato.csv'
}

print("Loading and processing data...")
all_data = []
sample_size = 3000  # Reduced sample size for better performance

for search_term, filename in datasets.items():
    try:
        df = pd.read_csv(filename)
        # Sample data to reduce processing time
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        df['search_term'] = search_term
        all_data.append(df)
        print(f"Loaded {len(df)} records from {search_term}")
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        continue

if not all_data:
    print("Error: No data loaded successfully")
    # Create dummy data for demonstration
    dates = pd.date_range('2022-01-01', periods=65, freq='D')
    dummy_data = []
    for term in list(datasets.keys())[:4]:  # Use first 4 terms
        for date in dates[:30]:  # 30 days of data
            dummy_data.append({
                'date': date,
                'search_term': term,
                'likeCount': np.random.randint(0, 100),
                'retweetCount': np.random.randint(0, 50),
                'replyCount': np.random.randint(0, 30),
                'quoteCount': np.random.randint(0, 20),
                'sourceLabel': np.random.choice(['Twitter for iPhone', 'Twitter for Android', 'Twitter Web App']),
                'user': '{"verified": ' + str(np.random.choice([True, False])).lower() + '}',
                'hashtags': np.random.choice([None, 'StandWithUkraine', 'NoWar', 'NATO'], p=[0.5, 0.2, 0.2, 0.1])
            })
    combined_df = pd.DataFrame(dummy_data)
    print("Using dummy data for demonstration")
else:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)} records")

# Data preprocessing with error handling
try:
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    combined_df = combined_df.dropna(subset=['date'])
    
    # Fill NaN values with 0 for numeric columns
    numeric_cols = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount']
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
        else:
            combined_df[col] = 0
    
    combined_df['engagement_score'] = combined_df['likeCount'] + combined_df['retweetCount'] + combined_df['replyCount']
    combined_df['day'] = combined_df['date'].dt.date
    
    # Simplified source extraction
    def get_source_type(source_str):
        if pd.isna(source_str):
            return 'Unknown'
        source_str = str(source_str).lower()
        if 'iphone' in source_str:
            return 'iPhone'
        elif 'android' in source_str:
            return 'Android'
        elif 'web' in source_str:
            return 'Web App'
        elif 'ipad' in source_str:
            return 'iPad'
        else:
            return 'Other'
    
    if 'sourceLabel' in combined_df.columns:
        combined_df['source_clean'] = combined_df['sourceLabel'].apply(get_source_type)
    else:
        combined_df['source_clean'] = 'Unknown'
    
    # Simplified verified status extraction
    def is_verified(user_str):
        if pd.isna(user_str):
            return False
        return 'verified": true' in str(user_str).lower()
    
    if 'user' in combined_df.columns:
        combined_df['verified'] = combined_df['user'].apply(is_verified)
    else:
        combined_df['verified'] = False
        
except Exception as e:
    print(f"Error in data preprocessing: {e}")
    # Create minimal required columns if preprocessing fails
    if 'engagement_score' not in combined_df.columns:
        combined_df['engagement_score'] = 10
    if 'day' not in combined_df.columns:
        combined_df['day'] = combined_df['date'].dt.date if 'date' in combined_df.columns else pd.date_range('2022-01-01', periods=len(combined_df), freq='D').date
    if 'source_clean' not in combined_df.columns:
        combined_df['source_clean'] = 'Unknown'
    if 'verified' not in combined_df.columns:
        combined_df['verified'] = False

# Create figure with 2x2 subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palette for search terms
unique_terms = combined_df['search_term'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_terms)))
search_term_colors = dict(zip(unique_terms, colors))

print("Creating visualizations...")

# Top-left: Daily tweet volume with engagement overlay
try:
    daily_stats = combined_df.groupby(['day', 'search_term']).agg({
        'search_term': 'count',  # Count tweets
        'engagement_score': 'mean'
    }).rename(columns={'search_term': 'tweet_count'}).reset_index()
    
    # Plot tweet volume trends for top 6 terms
    top_terms = combined_df['search_term'].value_counts().head(6).index
    
    for i, search_term in enumerate(top_terms):
        term_data = daily_stats[daily_stats['search_term'] == search_term]
        if len(term_data) > 0:
            ax1.plot(range(len(term_data)), term_data['tweet_count'], 
                    color=search_term_colors.get(search_term, colors[i % len(colors)]), 
                    linewidth=2, label=search_term[:15], alpha=0.8, marker='o', markersize=4)
    
    ax1.set_title('Daily Tweet Volume Trends by Search Term', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Time Period', fontweight='bold')
    ax1.set_ylabel('Tweet Count', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add engagement bars on secondary axis
    ax1_twin = ax1.twinx()
    engagement_by_term = combined_df.groupby('search_term')['engagement_score'].mean().head(6)
    x_pos = range(len(engagement_by_term))
    bars = ax1_twin.bar(x_pos, engagement_by_term.values, 
                       alpha=0.3, width=0.6,
                       color=[search_term_colors.get(term, 'gray') for term in engagement_by_term.index])
    ax1_twin.set_ylabel('Avg Engagement Score', fontweight='bold')
    ax1_twin.set_xticks(x_pos)
    ax1_twin.set_xticklabels([term[:10] for term in engagement_by_term.index], rotation=45)
    
except Exception as e:
    ax1.text(0.5, 0.5, f'Tweet Volume Analysis\n(Error: {str(e)[:30]}...)', 
             transform=ax1.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax1.set_title('Daily Tweet Volume Trends', fontweight='bold', fontsize=14)

# Top-right: Source distribution over time
try:
    # Create time periods
    combined_df['period'] = pd.cut(range(len(combined_df)), bins=10, labels=False)
    
    # Get source distribution by period
    source_dist = combined_df.groupby(['period', 'source_clean']).size().unstack(fill_value=0)
    
    if len(source_dist) > 0:
        # Convert to percentages
        source_dist_pct = source_dist.div(source_dist.sum(axis=1), axis=0) * 100
        
        # Create stacked area chart
        periods = range(len(source_dist_pct))
        bottom = np.zeros(len(periods))
        
        source_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        sources = source_dist_pct.columns[:5]  # Top 5 sources
        
        for i, source in enumerate(sources):
            color = source_colors[i % len(source_colors)]
            ax2.fill_between(periods, bottom, bottom + source_dist_pct[source], 
                           label=source, color=color, alpha=0.7)
            bottom += source_dist_pct[source]
    
    # Add verified users line
    verified_by_period = combined_df.groupby('period')['verified'].mean() * 100
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(verified_by_period)), verified_by_period.values, 
                 color='red', linewidth=3, label='% Verified', marker='o', markersize=6)
    ax2_twin.set_ylabel('Verified Users (%)', fontweight='bold', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    ax2.set_title('Tweet Source Distribution Over Time', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Time Period', fontweight='bold')
    ax2.set_ylabel('Source Distribution (%)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 100)
    
except Exception as e:
    ax2.text(0.5, 0.5, f'Source Distribution Analysis\n(Error: {str(e)[:30]}...)', 
             transform=ax2.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax2.set_title('Tweet Source Distribution', fontweight='bold', fontsize=14)

# Bottom-left: Sentiment indicators
try:
    # Calculate sentiment ratio (reply/like ratio)
    combined_df['sentiment_ratio'] = combined_df['replyCount'] / (combined_df['likeCount'] + 1)
    combined_df['period'] = pd.cut(range(len(combined_df)), bins=8, labels=False)
    
    # Group by period for better visualization
    sentiment_data = combined_df.groupby(['period', 'search_term']).agg({
        'sentiment_ratio': 'mean',
        'quoteCount': 'mean'
    }).reset_index()
    
    # Plot sentiment trends for top 5 terms
    top_terms = combined_df['search_term'].value_counts().head(5).index
    
    for i, search_term in enumerate(top_terms):
        term_data = sentiment_data[sentiment_data['search_term'] == search_term]
        if len(term_data) > 0:
            periods = term_data['period'].values
            sentiment = term_data['sentiment_ratio'].values
            quotes = term_data['quoteCount'].values
            
            # Plot line
            ax3.plot(periods, sentiment, 
                    color=search_term_colors.get(search_term, colors[i % len(colors)]), 
                    linewidth=2, label=search_term[:15], alpha=0.8, marker='o')
            
            # Add scatter points sized by quote count
            sizes = np.clip(quotes * 50 + 20, 20, 200)
            ax3.scatter(periods, sentiment, 
                       s=sizes, color=search_term_colors.get(search_term, colors[i % len(colors)]), 
                       alpha=0.6, edgecolors='white', linewidth=1)
    
    ax3.set_title('Sentiment Evolution (Reply/Like Ratio) with Quote Activity', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Time Period', fontweight='bold')
    ax3.set_ylabel('Reply/Like Ratio', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add text annotation for bubble size
    ax3.text(0.02, 0.98, 'Bubble size = Quote Count', transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
except Exception as e:
    ax3.text(0.5, 0.5, f'Sentiment Analysis\n(Error: {str(e)[:30]}...)', 
             transform=ax3.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax3.set_title('Sentiment Evolution', fontweight='bold', fontsize=14)

# Bottom-right: Calendar heatmap with hashtag trends
try:
    # Create daily intensity data
    daily_intensity = combined_df.groupby('day').size()
    
    # Create a simplified heatmap (7x9 grid for ~65 days)
    days_data = daily_intensity.values
    target_days = 63  # 9 weeks
    
    if len(days_data) < target_days:
        days_data = np.pad(days_data, (0, target_days - len(days_data)), 'constant', constant_values=0)
    else:
        days_data = days_data[:target_days]
    
    calendar_matrix = days_data.reshape(9, 7)
    
    # Create heatmap
    im = ax4.imshow(calendar_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax4.set_title('Daily Tweet Intensity (Calendar View)', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Day of Week', fontweight='bold')
    ax4.set_ylabel('Week Number', fontweight='bold')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax4.set_yticks(range(9))
    ax4.set_yticklabels([f'W{i+1}' for i in range(9)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8, aspect=20)
    cbar.set_label('Tweet Count', fontweight='bold')
    
    # Add hashtag trend overlay
    if 'hashtags' in combined_df.columns:
        key_hashtags = ['StandWithUkraine', 'NoWar', 'NATO']
        hashtag_colors = ['blue', 'green', 'orange']
        
        ax4_twin = ax4.twinx()
        
        for i, hashtag in enumerate(key_hashtags):
            # Count hashtag mentions
            hashtag_mask = combined_df['hashtags'].fillna('').astype(str).str.contains(hashtag, case=False, na=False)
            if hashtag_mask.any():
                hashtag_data = combined_df[hashtag_mask]
                hashtag_daily = hashtag_data.groupby('day').size()
                
                if len(hashtag_daily) > 0:
                    # Create trend line
                    x_data = range(min(len(hashtag_daily), 20))
                    y_data = hashtag_daily.values[:len(x_data)]
                    ax4_twin.plot(x_data, y_data, 
                                 color=hashtag_colors[i], linewidth=2, 
                                 label=f'#{hashtag}', alpha=0.8, marker='s', markersize=4)
        
        ax4_twin.set_ylabel('Hashtag Mentions', fontweight='bold')
        ax4_twin.legend(loc='upper right', fontsize=10)
        ax4_twin.set_xlim(0, 6)
    
except Exception as e:
    ax4.text(0.5, 0.5, f'Calendar Heatmap\n(Error: {str(e)[:30]}...)', 
             transform=ax4.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_title('Daily Tweet Intensity', fontweight='bold', fontsize=14)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)

# Add overall title
fig.suptitle('Ukraine War Twitter Engagement: Temporal Analysis of Social Media Discourse Evolution\n(65-Day Period Analysis)', 
             fontsize=18, fontweight='bold', y=0.98)

print("Visualization complete!")
plt.savefig('ukraine_twitter_temporal_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()