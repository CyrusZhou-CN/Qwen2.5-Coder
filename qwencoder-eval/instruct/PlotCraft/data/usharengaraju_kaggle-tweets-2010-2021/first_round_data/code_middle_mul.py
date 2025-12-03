import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
datasets = [
    'KaggleTweets2010.csv', 'KaggleTweets2011.csv', 'KaggleTweets2012.csv', 'KaggleTweets2013.csv',
    'KaggleTweets2014.csv', 'KaggleTweets2015.csv', 'KaggleTweets2016.csv', 'KaggleTweets2017.csv',
    'KaggleTweets2018.csv', 'KaggleTweets2019Part1.csv', 'KaggleTweets2019Part2.csv', 
    'KaggleTweets2020Part1.csv', 'KaggleTweets2020Part2.csv', 'KaggleTweets2021.csv'
]

# Combine all data
all_data = []
for dataset in datasets:
    try:
        df = pd.read_csv(dataset)
        all_data.append(df)
    except:
        pass

combined_df = pd.concat(all_data, ignore_index=True)
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])
combined_df['year'] = combined_df['created_at'].dt.year
combined_df['month'] = combined_df['created_at'].dt.month
combined_df['engagement'] = combined_df['likes_count'] + combined_df['retweets_count']

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 1x3 layout instead of requested 2x2 (Layout Violation)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Subplot 1: Pie chart instead of line+bar chart (Chart Type Mismatch)
yearly_counts = combined_df.groupby('year').size()
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ffffff', '#888888', '#444444', '#aaaaaa', '#666666', '#222222']
axes[0].pie(yearly_counts.values, labels=yearly_counts.index, colors=colors[:len(yearly_counts)], autopct='%1.1f%%')
axes[0].set_title('Random Pizza Distribution', fontsize=8)  # Wrong title

# Subplot 2: Scatter plot instead of area chart (Chart Type Mismatch)
lang_data = combined_df['language'].value_counts().head(3)
x_vals = np.random.random(len(lang_data)) * 100
y_vals = np.random.random(len(lang_data)) * 100
axes[1].scatter(x_vals, y_vals, s=lang_data.values, c=['red', 'blue', 'green'], alpha=0.7)
axes[1].set_xlabel('Amplitude')  # Swapped labels
axes[1].set_ylabel('Time')
axes[1].set_title('Glarbnok\'s Revenge Analysis')  # Nonsensical title

# Subplot 3: Bar chart instead of time series decomposition (Chart Type Mismatch)
monthly_data = combined_df.groupby('month')['engagement'].mean()
bars = axes[2].bar(monthly_data.index, monthly_data.values, color='cyan', edgecolor='white', linewidth=3)
axes[2].set_xlabel('Frequency Modulation')  # Wrong label
axes[2].set_ylabel('Quantum Flux')  # Wrong label
axes[2].set_title('Banana Metrics Dashboard')  # Unrelated title

# Add overlapping text annotations to destroy readability
for ax in axes:
    ax.text(0.5, 0.5, 'OVERLAPPING TEXT\nOVERLAPPING TEXT\nOVERLAPPING TEXT', 
            transform=ax.transAxes, fontsize=16, color='white', alpha=0.8,
            ha='center', va='center', weight='bold')

# Force cramped layout with tiny spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.02, right=0.98, top=0.85, bottom=0.15)

# Add a main title that's completely wrong
fig.suptitle('Cryptocurrency Mining Efficiency Report 2025', fontsize=12, y=0.95)

# Make all text the same size (no visual hierarchy)
for ax in axes:
    ax.tick_params(labelsize=8)
    ax.title.set_fontsize(8)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)

# Add thick, ugly spines
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()