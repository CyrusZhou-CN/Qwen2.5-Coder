import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load all datasets
datasets = [
    'KaggleTweets2010.csv', 'KaggleTweets2011.csv', 'KaggleTweets2012.csv', 'KaggleTweets2013.csv',
    'KaggleTweets2014.csv', 'KaggleTweets2015.csv', 'KaggleTweets2016.csv', 'KaggleTweets2017.csv',
    'KaggleTweets2018.csv', 'KaggleTweets2019Part1.csv', 'KaggleTweets2019Part2.csv', 'KaggleTweets2020Part1.csv',
    'KaggleTweets2020Part2.csv', 'KaggleTweets2021.csv'
]

# Create fake data since we can't load files
years = list(range(2010, 2022))
engagement_data = []

for year in years:
    # Generate random engagement data
    num_tweets = np.random.randint(1000, 40000)
    avg_likes = np.random.uniform(0.5, 15.0)
    avg_retweets = np.random.uniform(0.2, 8.0)
    avg_replies = np.random.uniform(0.1, 3.0)
    
    # Calculate total engagement metric
    total_engagement = avg_likes + avg_retweets + avg_replies
    engagement_data.append(total_engagement)

# Set dark background style
plt.style.use('dark_background')

# Create 2x2 subplots instead of requested 1x1
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Use subplots_adjust to force overlap
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Plot 1: Bar chart instead of line chart (wrong chart type)
bars = ax1.bar(years, engagement_data, color='red', alpha=0.7, width=1.2)
ax1.set_ylabel('Time (seconds)')  # Wrong axis label
ax1.set_xlabel('Amplitude (volts)')  # Wrong axis label
ax1.set_title('Random Noise Distribution', fontsize=8)  # Wrong title, same size as labels
ax1.grid(True, color='white', linewidth=2)  # Heavy white grid on dark background

# Plot 2: Scatter plot of random data
random_x = np.random.uniform(2010, 2021, 50)
random_y = np.random.uniform(0, 30, 50)
ax2.scatter(random_x, random_y, c='yellow', s=100, alpha=0.8)
ax2.set_title('Glarbnok\'s Revenge', fontsize=8)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power (Watts)')

# Plot 3: Pie chart (completely inappropriate for time series)
pie_data = [25, 30, 20, 25]
pie_labels = ['Zorblex', 'Flimflam', 'Quibble', 'Snurfle']
ax3.pie(pie_data, labels=pie_labels, colors=['magenta', 'cyan', 'orange', 'lime'])
ax3.set_title('Data Series A', fontsize=8)

# Plot 4: Empty plot with just text
ax4.text(0.5, 0.5, 'MISSING DATA\nERROR 404', ha='center', va='center', 
         fontsize=20, color='red', weight='bold', transform=ax4.transAxes)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Engagement Trends', fontsize=8)

# Add overlapping text annotation on plot 1
ax1.text(2015, max(engagement_data)*0.8, 'CRITICAL\nFAILURE', 
         fontsize=16, color='white', weight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

# Make axis spines thick and clumsy
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=100, facecolor='black')