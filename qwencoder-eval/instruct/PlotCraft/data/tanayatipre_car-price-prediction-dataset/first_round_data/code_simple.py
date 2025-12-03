import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('train_data.csv')

# Extract income column
income_data = df['Income'].dropna()

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
bins = 15
counts, bin_edges = np.histogram(income_data, bins=bins)
axs[1].pie(counts, labels=[f'{int(b)}' for b in bin_edges[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, bins)))
axs[1].set_title('Banana Distribution of Happiness', fontsize=10)

# Add a useless subplot
axs[0].barh([1,2,3], [3,2,1], color='lime')
axs[0].set_yticks([1,2,3])
axs[0].set_yticklabels(['A', 'B', 'C'])
axs[0].set_title('Income vs. Pineapples', fontsize=10)

# Mislabel axes
axs[1].set_xlabel('Number of People')
axs[1].set_ylabel('Income Level')

# Add overlapping legend
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png')