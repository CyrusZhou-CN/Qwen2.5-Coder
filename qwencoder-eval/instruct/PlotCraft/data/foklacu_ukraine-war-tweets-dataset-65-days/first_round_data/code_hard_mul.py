import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# Load all datasets with sampling to improve performance
datasets = {
    'Ukraine war': load_and_sample_data('Ukraine_war.csv'),
    'StandWithUkraine': load_and_sample_data('StandWithUkraine.csv'),
    'Russian troops': load_and_sample_data('Russian_troops.csv'),
    'Ukraine troops': load_and_sample_data('Ukraine_troops.csv'),
    'Ukraine border': load_and_sample_data('Ukraine_border.csv'),
    'Ukraine NATO': load_and_sample_data('Ukraine_nato.csv'),
    'Russia invade': load_and_sample_data('Russia_invade.csv'),
    'Russian border Ukraine': load_and_sample_data('Russian_border_Ukraine.csv')
}

# Data preprocessing with error handling
def preprocess_data(df, search_term):
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['date_only'] = df['date'].dt.date
        df['search_term'] = search_term
        
        # Handle missing values in engagement columns
        engagement_cols = ['likeCount', 'retweetCount', 'replyCount']
        for col in engagement_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
                
        df['total_engagement'] = df['likeCount'] + df['retweetCount'] + df['replyCount']
        return df
    except:
        return pd.DataFrame()

# Process all datasets
processed_data = {}
for term, df in datasets.items():
    processed_data[term] = preprocess_data(df, term)

# Filter out empty datasets
processed_data = {k: v for k, v in processed_data.items() if not v.empty}

# Create figure with optimized settings
plt.style.use('default')
fig = plt.figure(figsize=(18, 12), facecolor='white')
fig.suptitle('Temporal Evolution of Ukraine War Discourse on Twitter', 
             fontsize=20, fontweight='bold', y=0.95)

# Define color schemes
colors = {
    'Ukraine war': '#1f77b4',
    'StandWithUkraine': '#ff7f0e', 
    'Russian troops': '#d62728',
    'Ukraine troops': '#2ca02c',
    'Ukraine border': '#9467bd',
    'Ukraine NATO': '#8c564b',
    'Russia invade': '#e377c2',
    'Russian border Ukraine': '#7f7f7f'
}

# Helper function to get daily aggregated data
def get_daily_data(df):
    if df.empty:
        return pd.DataFrame()
    
    daily_data = df.groupby('date_only').agg({
        'date_only': 'first',
        'likeCount': 'sum',
        'retweetCount': 'sum', 
        'replyCount': 'sum',
        'total_engagement': ['mean', 'count']
    }).reset_index(drop=True)
    
    daily_data.columns = ['date_only', 'likes', 'retweets', 'replies', 'avg_engagement', 'tweet_count']
    return daily_data

# Top row subplots - Dual-axis time series
subplot_configs = [
    (['Ukraine war', 'StandWithUkraine'], 1, "Ukraine War vs Support"),
    (['Russian troops', 'Ukraine troops'], 2, "Military Forces Discussion"), 
    (['Ukraine border', 'Ukraine NATO'], 3, "Border & NATO Relations")
]

for terms, subplot_num, title in subplot_configs:
    ax1 = plt.subplot(2, 3, subplot_num)
    ax2 = ax1.twinx()
    
    for term in terms:
        if term in processed_data and not processed_data[term].empty:
            daily_data = get_daily_data(processed_data[term])
            
            if not daily_data.empty:
                # Primary axis - daily tweet volume
                ax1.plot(daily_data['date_only'], daily_data['tweet_count'], 
                        color=colors[term], linewidth=2, label=f'{term} Volume', alpha=0.8)
                
                # Secondary axis - engagement stacked areas
                ax2.fill_between(daily_data['date_only'], 0, daily_data['likes'], 
                                color=colors[term], alpha=0.3, label=f'{term} Likes')
    
    ax1.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('Daily Tweet Count', fontweight='bold', color='black')
    ax2.set_ylabel('Total Likes', fontweight='bold', color='gray')
    ax1.grid(True, alpha=0.3)
    
    # Handle legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

# Bottom row subplots - Line plots with scatter overlays
bottom_configs = [
    (['Russia invade', 'Russian border Ukraine'], 4, "Invasion Discourse"),
    (list(processed_data.keys())[:3], 5, "Overall Discourse Evolution"),
    (list(processed_data.keys())[-3:], 6, "Comparative Analysis")
]

for terms, subplot_num, title in bottom_configs:
    ax = plt.subplot(2, 3, subplot_num)
    
    for term in terms:
        if term in processed_data and not processed_data[term].empty:
            daily_data = get_daily_data(processed_data[term])
            
            if not daily_data.empty and len(daily_data) > 1:
                # Line plot for average engagement
                ax.plot(daily_data['date_only'], daily_data['avg_engagement'], 
                       color=colors[term], linewidth=2, label=f'{term}', alpha=0.8)
                
                # Scatter plot for high-engagement points
                high_engagement_mask = daily_data['avg_engagement'] > daily_data['avg_engagement'].quantile(0.8)
                if high_engagement_mask.any():
                    high_data = daily_data[high_engagement_mask]
                    scatter_sizes = np.clip(high_data['avg_engagement'] / 10, 10, 100)
                    ax.scatter(high_data['date_only'], high_data['avg_engagement'],
                              s=scatter_sizes, color=colors[term], alpha=0.6, 
                              edgecolors='white', linewidth=0.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Average Engagement', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

# Styling improvements
for i in range(1, 7):
    ax = plt.subplot(2, 3, i)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) if i <= 3 else None
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666', labelsize=8)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set x-axis label for bottom row
    if i > 3:
        ax.set_xlabel('Date', fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.92])

# Add footer
fig.text(0.5, 0.01, 'Ukraine War Twitter Analysis: 65-Day Evolution Across Search Terms', 
         ha='center', fontsize=9, style='italic', color='#666666')

plt.savefig('ukraine_war_discourse_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()