import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load and combine all datasets with error handling
try:
    df1 = pd.read_csv('bangladeshi_all_engish_newspapers_daily_news_combined_dataset.csv')
except:
    df1 = pd.DataFrame()

try:
    df2 = pd.read_csv('financialexpress_daily_news.csv')
except:
    df2 = pd.DataFrame()

try:
    df3 = pd.read_csv('dailystar_daily_news.csv')
except:
    df3 = pd.DataFrame()

try:
    df4 = pd.read_csv('newagebd_daily_news.csv')
except:
    df4 = pd.DataFrame()

# Combine all non-empty datasets
dfs = [df for df in [df1, df2, df3, df4] if not df.empty]
if not dfs:
    raise ValueError("No data files could be loaded")

df = pd.concat(dfs, ignore_index=True)

# Quick data preprocessing with error handling
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
df['news_collection_time'] = pd.to_datetime(df['news_collection_time'], errors='coerce')

# Calculate word count more efficiently
df['word_count'] = df['text'].fillna('').astype(str).str.split().str.len()
df['word_count'] = df['word_count'].fillna(0)

# Extract hour from collection time
df['collection_hour'] = df['news_collection_time'].dt.hour

# Filter out rows with missing essential data
df = df.dropna(subset=['publish_date', 'publisher'])

# Limit data size for performance - take recent data only
df = df.sort_values('publish_date').tail(1000)

# Create figure
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Get unique publishers and assign colors
publishers = df['publisher'].unique()[:4]  # Limit to 4 publishers
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
publisher_colors = dict(zip(publishers, colors[:len(publishers)]))

# Subplot 1: Daily article counts with rolling average
try:
    daily_counts = df.groupby([df['publish_date'].dt.date, 'publisher']).size().unstack(fill_value=0)
    daily_counts.index = pd.to_datetime(daily_counts.index)
    
    # Limit to last 30 days for performance
    daily_counts = daily_counts.tail(30)
    
    # Bar chart for daily counts
    bottom = np.zeros(len(daily_counts))
    for publisher in publishers:
        if publisher in daily_counts.columns:
            ax1.bar(daily_counts.index, daily_counts[publisher], bottom=bottom, 
                    label=publisher, color=publisher_colors[publisher], alpha=0.7, width=0.8)
            bottom += daily_counts[publisher]
    
    # Add rolling average line
    total_daily = daily_counts.sum(axis=1)
    if len(total_daily) > 3:
        rolling_avg = total_daily.rolling(window=min(7, len(total_daily)), center=True).mean()
        ax1.plot(rolling_avg.index, rolling_avg.values, color='black', linewidth=2, 
                 label='Rolling Average', alpha=0.8)
    
    ax1.set_title('Daily Article Counts by Publisher', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Articles', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=45)
    
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1: {str(e)[:50]}...', 
             ha='center', va='center', transform=ax1.transAxes)

# Subplot 2: Stacked area chart
try:
    if len(daily_counts) > 0:
        daily_props = daily_counts.div(daily_counts.sum(axis=1), axis=0).fillna(0)
        
        # Create stacked area chart
        ax2.stackplot(daily_props.index, *[daily_props[pub] if pub in daily_props.columns 
                                          else np.zeros(len(daily_props)) for pub in publishers],
                     labels=publishers, colors=[publisher_colors[pub] for pub in publishers], alpha=0.7)
        
        ax2.set_title('Coverage Proportion by Publisher', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.set_ylabel('Proportion', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.tick_params(axis='x', rotation=45)
    
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2: {str(e)[:50]}...', 
             ha='center', va='center', transform=ax2.transAxes)

# Subplot 3: Heatmap of hourly collection patterns
try:
    hourly_data = df.groupby(['publisher', 'collection_hour']).size().unstack(fill_value=0)
    hourly_data = hourly_data.reindex(columns=range(24), fill_value=0)
    
    # Limit publishers for readability
    hourly_data = hourly_data.head(4)
    
    if not hourly_data.empty:
        im = ax3.imshow(hourly_data.values, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(0, 24, 4))
        ax3.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 4)])
        ax3.set_yticks(range(len(hourly_data.index)))
        ax3.set_yticklabels(hourly_data.index, fontsize=8)
        
        # Add peak markers
        for i, publisher in enumerate(hourly_data.index):
            if hourly_data.loc[publisher].sum() > 0:
                peak_hour = hourly_data.loc[publisher].idxmax()
                ax3.scatter(peak_hour, i, color='white', s=100, marker='*', 
                           edgecolors='black', linewidth=1)
        
        ax3.set_title('Hourly Collection Patterns\n(★ = peak times)', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Hour', fontweight='bold')
        ax3.set_ylabel('Publisher', fontweight='bold')
    
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3: {str(e)[:50]}...', 
             ha='center', va='center', transform=ax3.transAxes)

# Subplot 4: Word count distribution
try:
    # Filter reasonable word counts
    word_data = df[(df['word_count'] > 10) & (df['word_count'] < 2000)]
    
    if not word_data.empty:
        # Histogram
        bins = np.linspace(word_data['word_count'].min(), word_data['word_count'].max(), 20)
        
        for publisher in publishers:
            pub_data = word_data[word_data['publisher'] == publisher]['word_count']
            if len(pub_data) > 0:
                ax4.hist(pub_data, bins=bins, alpha=0.6, label=f'{publisher} (n={len(pub_data)})', 
                        color=publisher_colors[publisher], density=True)
        
        # KDE on secondary axis
        ax4_twin = ax4.twinx()
        x_range = np.linspace(word_data['word_count'].min(), word_data['word_count'].max(), 100)
        
        for publisher in publishers:
            pub_data = word_data[word_data['publisher'] == publisher]['word_count']
            if len(pub_data) > 10:
                try:
                    kde = gaussian_kde(pub_data)
                    density = kde(x_range)
                    ax4_twin.plot(x_range, density, color=publisher_colors[publisher], 
                                 linewidth=2, linestyle='--', alpha=0.8)
                except:
                    continue
        
        ax4.set_title('Article Length Distribution', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Word Count', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4_twin.set_ylabel('Density', fontweight='bold')
        ax4.legend(fontsize=8)
    
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4: {str(e)[:50]}...', 
             ha='center', va='center', transform=ax4.transAxes)

# Add main title
fig.suptitle('Bangladeshi English Newspapers: Temporal Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
plt.savefig('news_analysis_dashboard.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()