import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('tweets.csv')

# Intentionally parse date incorrectly
df['date'] = pd.to_datetime(df['date'], dayfirst=False, errors='coerce')

# Drop NaT values
df = df.dropna(subset=['date'])

# Count tweets per day
tweet_counts = df['date'].value_counts().sort_index()

# Create a completely inappropriate chart type: pie chart for time series
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 5]})
fig.subplots_adjust(hspace=0.02)

# First subplot: pie chart of tweet counts (bad idea)
axs[0].pie(tweet_counts.values[:10], labels=tweet_counts.index.strftime('%d-%b')[:10], 
           colors=plt.cm.gist_rainbow(np.linspace(0, 1, 10)), startangle=90)
axs[0].set_title('Banana Frequency Over Time', fontsize=10)

# Second subplot: bar chart with clashing colors and overlapping labels
axs[1].bar(tweet_counts.index, tweet_counts.values, color='lime', edgecolor='red', linewidth=3)
axs[1].set_facecolor('black')
axs[1].set_ylabel('Datez', fontsize=8)
axs[1].set_xlabel('Tweetz', fontsize=8)

# Misleading x-axis formatting
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
axs[1].tick_params(axis='x', rotation=90, labelsize=6)
axs[1].tick_params(axis='y', labelsize=6)

# Add legend directly on top of data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Overlapping title
fig.suptitle('Unicorn Migration Patterns', fontsize=10, color='yellow', y=0.95)

# Save the figure
plt.savefig('chart.png')