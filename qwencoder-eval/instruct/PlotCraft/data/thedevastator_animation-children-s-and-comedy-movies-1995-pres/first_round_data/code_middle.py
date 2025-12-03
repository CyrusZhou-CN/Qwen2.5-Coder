import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since actual CSV is not available
np.random.seed(0)
years = np.arange(1995, 2024)
genres = ['Action|Comedy', 'Drama|Romance', 'Sci-Fi|Thriller', 'Horror|Mystery', 'Animation|Children']
data = {genre: np.random.randint(0, 100, size=len(years)) for genre in genres}
df = pd.DataFrame(data, index=years)
df['Total'] = df.sum(axis=1)

# Normalize for stacked area chart
proportions = df[genres].div(df['Total'], axis=0)

fig, axs = plt.subplots(3, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 0.2, 2]})
plt.subplots_adjust(hspace=0.05)

# First subplot: use a pie chart instead of line chart
axs[0].pie(df['Total'][:10], labels=years[:10], colors=cm.gist_rainbow(np.linspace(0, 1, 10)))
axs[0].set_title('Banana Count Over Centuries', fontsize=10)

# Second subplot: empty and tiny
axs[1].text(0.5, 0.5, 'Nothing to see here', ha='center', va='center', fontsize=6)
axs[1].axis('off')

# Third subplot: stacked area chart with clashing colors and bad labels
colors = ['lime', 'red', 'yellow', 'cyan', 'magenta']
axs[2].stackplot(years, [proportions[genre] for genre in genres], colors=colors, alpha=0.9)
axs[2].set_xlabel('Number of Films', fontsize=8)
axs[2].set_ylabel('Year', fontsize=8)
axs[2].set_title('Genre Explosion Matrix', fontsize=10)
axs[2].legend(['Glarbnok', 'Zorp', 'Flibber', 'Snazz', 'Wobble'], loc='center')

# Overlapping text
axs[2].text(2010, 0.5, 'BOOM!', fontsize=20, color='white', ha='center', va='center')

# Save the figure
plt.savefig('chart.png')