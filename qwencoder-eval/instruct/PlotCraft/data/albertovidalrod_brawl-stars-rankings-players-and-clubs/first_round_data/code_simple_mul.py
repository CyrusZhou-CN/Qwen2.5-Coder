import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('global_club_rankings.csv')

# Sort the dataframe by trophies in descending order and select the top 15
top_15_clubs = df.sort_values(by='trophies', ascending=False).head(15)

# Define the color map for the gradient
cmap = plt.get_cmap('Blues')
colors = cmap(np.linspace(0.2, 1.0, len(top_15_clubs)))

# Create the horizontal bar chart
plt.figure(figsize=(10, 12))
bars = plt.barh(top_15_clubs['name'], top_15_clubs['trophies'], color=colors)

# Add the exact trophy count as text labels at the end of each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width}', va='center', ha='left')

# Set the title and labels
plt.title('Top 15 Clubs by Trophy Count')
plt.xlabel('Trophy Count')
plt.ylabel('Club Name')

# Show the plot
plt.show()