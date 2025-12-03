import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('tweets.csv')

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Extract day of the week and count tweets per day
daily_tweets = data.resample('D', on='date').size().reset_index(name='tweet_count')
weekly_tweets = data.resample('W', on='date').size().reset_index(name='tweet_count')
weekly_tweets['day_of_week'] = weekly_tweets['date'].dt.day_name()

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Top plot: Line chart of daily tweet counts
axes[0].plot(daily_tweets['date'], daily_tweets['tweet_count'], marker='o', linestyle='-')
axes[0].set_title('Daily Tweet Counts Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Tweet Count')
axes[0].grid(True)

# Bottom plot: Bar chart of tweet frequency by day of the week
axes[1].bar(weekly_tweets['day_of_week'], weekly_tweets['tweet_count'], color='skyblue')
axes[1].set_title('Tweet Frequency by Day of the Week')
axes[1].set_xlabel('Day of the Week')
axes[1].set_ylabel('Tweet Count')
axes[1].grid(axis='y')

plt.tight_layout()
plt.show()