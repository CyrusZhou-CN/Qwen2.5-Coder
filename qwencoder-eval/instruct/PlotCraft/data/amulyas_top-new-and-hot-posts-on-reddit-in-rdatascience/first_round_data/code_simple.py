import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use the worst possible style
plt.style.use('seaborn-v0_8-darkgrid')

# Load the dataset
df = pd.read_csv('reddit_datascience_newTopHot_posts.csv')

# Extract the 'Score' column
scores = df['Score'].dropna()

# Create a figure with a terrible layout
fig, axs = plt.subplots(2, 1, figsize=(12, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.02)

# Use a pie chart instead of a histogram
bins = np.linspace(scores.min(), scores.max(), 5)
counts, _ = np.histogram(scores, bins=bins)
axs[0].pie(counts, labels=[f'Bin {i}' for i in range(len(counts))], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[0].set_title('Banana Distribution of Likes', fontsize=10)

# Second plot: a bar chart with overlapping labels
axs[1].bar(bins[:-1], counts, width=50, color='lime', edgecolor='red')
axs[1].set_facecolor('black')
axs[1].set_xlabel('Number of Posts', fontsize=8)
axs[1].set_ylabel('Score Range', fontsize=8)
axs[1].set_title('How Many Upvotes Are There?', fontsize=10)

# Add a legend that covers the data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png')