import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import ast
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
datasets = {
    'KaggleTweets2010.csv': pd.read_csv('KaggleTweets2010.csv'),
    'KaggleTweets2011.csv': pd.read_csv('KaggleTweets2011.csv'),
    'KaggleTweets2012.csv': pd.read_csv('KaggleTweets2012.csv'),
    'KaggleTweets2013.csv': pd.read_csv('KaggleTweets2013.csv'),
    'KaggleTweets2014.csv': pd.read_csv('KaggleTweets2014.csv'),
    'KaggleTweets2015.csv': pd.read_csv('KaggleTweets2015.csv'),
    'KaggleTweets2016.csv': pd.read_csv('KaggleTweets2016.csv'),
    'KaggleTweets2017.csv': pd.read_csv('KaggleTweets2017.csv'),
    'KaggleTweets2018.csv': pd.read_csv('KaggleTweets2018.csv'),
    'KaggleTweets2019Part1.csv': pd.read_csv('KaggleTweets2019Part1.csv'),
    'KaggleTweets2019Part2.csv': pd.read_csv('KaggleTweets2019Part2.csv'),
    'KaggleTweets2020Part1.csv': pd.read_csv('KaggleTweets2020Part1.csv'),
    'KaggleTweets2020Part2.csv': pd.read_csv('KaggleTweets2020Part2.csv'),
    'KaggleTweets2021.csv': pd.read_csv('KaggleTweets2021.csv')
}

# Combine all datasets
all_data = []
for filename, df in datasets.items():
    year = int(filename.split('Tweets')[1][:4])
    df['year'] = year
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])
combined_df['engagement'] = combined_df['likes_count'] + combined_df['retweets_count']

# Generate fake data for missing years and metrics
np.random.seed(42)
years = list(range(2010, 2022))
tweet_volumes = np.random.randint(1000, 50000, len(years))
avg_engagement = np.random.uniform(2, 15, len(years))
languages = ['en', 'ja', 'es', 'fr', 'de']
lang_props = np.random.dirichlet(np.ones(len(languages)), len(years))
diversity_scores = np.random.uniform(0.3, 0.9, len(years))
hashtag_freq = np.random.randint(500, 5000, len(years))
tweet_lengths = np.random.uniform(50, 200, len(years))
engagement_scatter = np.random.uniform(1, 20, len(years))

# Use dark background style
plt.style.use('dark_background')

# Create 1x3 subplot (violating 2x2 requirement)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Plot 1: Pie chart instead of line+bar (wrong chart type)
ax1 = axes[0]
pie_data = np.random.randint(10, 100, 5)
pie_labels = ['Random A', 'Random B', 'Random C', 'Random D', 'Random E']
ax1.pie(pie_data, labels=pie_labels, colors=['red', 'orange', 'yellow', 'green', 'blue'])
ax1.set_title('Cryptocurrency Market Share Analysis', fontsize=8, color='white')
ax1.text(0.5, -0.8, 'OVERLAPPING TEXT CHAOS', transform=ax1.transAxes, fontsize=16, color='cyan', ha='center')

# Plot 2: Scatter plot instead of stacked area (wrong chart type)
ax2 = axes[1]
x_scatter = np.random.uniform(0, 10, 100)
y_scatter = np.random.uniform(0, 10, 100)
ax2.scatter(x_scatter, y_scatter, c='magenta', s=200, alpha=0.7)
ax2.set_xlabel('Temperature (Celsius)', fontsize=6, color='white')
ax2.set_ylabel('Ice Cream Sales', fontsize=6, color='white')
ax2.set_title('Quantum Physics Correlation Matrix', fontsize=8, color='white')
ax2.grid(True, color='white', linewidth=3)

# Plot 3: Bar chart instead of dual-axis time series (wrong chart type)
ax3 = axes[2]
categories = ['Apples', 'Bananas', 'Cherries', 'Dates', 'Elderberries']
values = np.random.randint(1, 20, len(categories))
bars = ax3.bar(categories, values, color=['purple', 'pink', 'brown', 'gray', 'lime'])
ax3.set_xlabel('Fruit Types', fontsize=6, color='white')
ax3.set_ylabel('Stock Market Index', fontsize=6, color='white')
ax3.set_title('Weather Patterns in Antarctica', fontsize=8, color='white')
ax3.tick_params(axis='x', rotation=90, labelsize=4)

# Add overlapping text annotations
fig.text(0.5, 0.5, 'MASSIVE OVERLAPPING TITLE', fontsize=24, color='red', ha='center', va='center', alpha=0.8)
fig.text(0.2, 0.7, 'Glarbnok Analysis', fontsize=14, color='yellow', rotation=45)
fig.text(0.8, 0.3, 'Flibber Metrics', fontsize=12, color='cyan', rotation=-30)

# Make axes thick and ugly
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=100, bbox_inches='tight', facecolor='black')