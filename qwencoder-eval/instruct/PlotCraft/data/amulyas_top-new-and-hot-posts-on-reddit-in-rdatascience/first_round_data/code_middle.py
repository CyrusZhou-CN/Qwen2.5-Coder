import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('reddit_datascience_newTopHot_posts.csv')

# Select the top 15 posts based on Score
top_15_posts = df.nlargest(15, 'Score')

# Set the figure size
plt.figure(figsize=(12, 8))

# Create a horizontal bar chart for Score
bars = plt.barh(top_15_posts['Title'].str[:50], top_15_posts['Score'], color='skyblue')
plt.xlabel('Score')
plt.ylabel('Post Title')
plt.title('Top 15 Posts Ranked by Score')

# Overlay a secondary axis for Comment Count
ax2 = plt.twinx()
ax2.plot(top_15_posts['Title'].str[:50], top_15_posts['Comment Count'], color='orange', marker='o')
ax2.set_ylabel('Comment Count')

# Add labels for each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width}', va='center', ha='left')

# Add legend
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper right')

# Show the plot
plt.show()