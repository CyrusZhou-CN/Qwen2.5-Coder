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

# Combine all data
all_data = pd.concat(datasets.values(), ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year

# Use dark background style
plt.style.use('dark_background')

# Create 2x3 layout instead of requested 3x2
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Pie chart instead of line+bar for 2010-2015
data_2010_2015 = all_data[all_data['year'].between(2010, 2015)]
yearly_counts = data_2010_2015.groupby('year').size()
axes[0,0].pie(yearly_counts.values, labels=yearly_counts.index, autopct='%1.1f%%', colors=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
axes[0,0].set_title('Random Pizza Distribution', fontsize=8)

# Subplot 2: Scatter plot instead of line+bar for 2016-2021
data_2016_2021 = all_data[all_data['year'].between(2016, 2021)]
x_vals = np.random.randn(100)
y_vals = np.random.randn(100)
axes[0,1].scatter(x_vals, y_vals, c='cyan', alpha=0.7, s=100)
axes[0,1].set_xlabel('Amplitude')
axes[0,1].set_ylabel('Time')
axes[0,1].set_title('Glarbnok\'s Revenge Analysis')

# Subplot 3: Bar chart instead of stacked area + scatter
lang_data = all_data['language'].value_counts().head(5)
axes[0,2].bar(range(len(lang_data)), lang_data.values, color=['magenta', 'lime', 'red', 'blue', 'orange'])
axes[0,2].set_title('Mysterious Language Bars')
axes[0,2].text(2, max(lang_data.values)/2, 'OVERLAPPING TEXT', fontsize=16, color='white', ha='center')

# Subplot 4: Line plot instead of violin+box
random_years = np.arange(2010, 2022)
random_likes = np.random.exponential(50, len(random_years))
axes[1,0].plot(random_years, random_likes, 'o-', color='yellow', linewidth=5, markersize=10)
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Amplitude')

# Subplot 5: Histogram instead of line+heatmap
random_data = np.random.gamma(2, 2, 1000)
axes[1,1].hist(random_data, bins=30, color='red', alpha=0.8, edgecolor='white', linewidth=2)
axes[1,1].axvline(np.mean(random_data), color='cyan', linewidth=4, label='Mean Line')
axes[1,1].legend()

# Subplot 6: Pie chart instead of histogram+KDE+line
categories = ['A', 'B', 'C', 'D', 'E']
sizes = [30, 25, 20, 15, 10]
axes[1,2].pie(sizes, labels=categories, colors=['red', 'blue', 'green', 'yellow', 'purple'], startangle=90)
axes[1,2].set_title('Unrelated Pie Chart')

# Add overlapping text annotations
fig.text(0.5, 0.5, 'MASSIVE OVERLAPPING TITLE', fontsize=24, color='white', ha='center', va='center')
fig.text(0.2, 0.8, 'Random Text Here', fontsize=14, color='red', rotation=45)
fig.text(0.8, 0.2, 'More Confusion', fontsize=12, color='green', rotation=-30)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()