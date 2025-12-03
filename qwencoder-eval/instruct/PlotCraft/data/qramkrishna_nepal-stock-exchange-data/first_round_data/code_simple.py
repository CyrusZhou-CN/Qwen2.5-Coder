import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake 'Rate' data since actual column names are inconsistent
np.random.seed(42)
rates = np.concatenate([
    np.random.normal(500, 100, 10000),
    np.random.exponential(200, 5000),
    np.random.uniform(100, 1000, 3000)
])

# Create a 2x1 layout instead of 1x1
fig, axs = plt.subplots(2, 1, figsize=(12, 4), facecolor='lightgray')
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
bins = np.histogram_bin_edges(rates, bins=10)
counts, _ = np.histogram(rates, bins=bins)
axs[0].pie(counts, labels=[f'Bin {i}' for i in range(len(counts))], 
           colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))),
           startangle=90, textprops={'color': 'yellow'})
axs[0].set_title("Banana Price Explosion", fontsize=10)

# Second subplot: KDE on a bar chart
sns.kdeplot(rates, ax=axs[1], color='lime', lw=5)
axs[1].bar(np.linspace(0, 1, 10), np.random.rand(10)*1000, color='red', alpha=0.9)
axs[1].set_xlabel("Frequency of Unicorns", fontsize=8)
axs[1].set_ylabel("Price in Lightyears", fontsize=8)

# Add legend directly over the plot
axs[1].legend(['Glarbnok’s Revenge'], loc='center')

# Save the figure
plt.savefig("chart.png", dpi=100)