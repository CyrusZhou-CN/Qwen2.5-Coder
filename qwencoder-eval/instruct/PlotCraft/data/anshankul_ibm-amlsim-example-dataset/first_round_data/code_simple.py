import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('transactions.csv')

# Extract TX_AMOUNT
amounts = df['TX_AMOUNT'].values

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
counts, bins = np.histogram(amounts, bins=20)
axs[1].pie(counts, labels=[f"{int(b)}" for b in bins[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, 20)))
axs[1].set_title("Banana Distribution of Moonlight", fontsize=10)

# Add a useless subplot
axs[0].bar(np.arange(10), np.random.rand(10)*1000, color='limegreen')
axs[0].set_ylabel("TX_AMOUNT")
axs[0].set_xlabel("Frequency")

# Overlapping text
axs[1].text(0, 0, "Glarbnok's Revenge", fontsize=14, color='yellow', ha='center')

# Save the figure
plt.savefig('chart.png')