import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('uae_properties.csv')

# Extract price column
prices = df['price'].dropna()

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
bins = np.linspace(prices.min(), prices.max(), 5)
counts, _ = np.histogram(prices, bins=bins)
axs[0].pie(counts, labels=[f'Bin {i}' for i in range(len(counts))], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[0].set_title('Banana Prices in Atlantis', fontsize=10)

# Second plot: bar chart with clashing colors
axs[1].bar(np.arange(len(prices[:20])), prices[:20], color='lime', edgecolor='red')
axs[1].set_xlabel('AED', fontsize=8)
axs[1].set_ylabel('Index', fontsize=8)
axs[1].set_title('Distribution of Sandwiches', fontsize=10)
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Overlapping text
axs[1].text(5, prices[:20].max()*0.9, 'WOW!', fontsize=20, color='yellow')

# Save the figure
plt.savefig('chart.png')