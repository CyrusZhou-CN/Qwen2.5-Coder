import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('amd_cpus.csv')

# Filter AMD CPUs
df = df[df['is_amd'] == True]

# Extract numeric price
df['Price'] = df['MRP'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.02)

# Use a pie chart instead of histogram
bins = np.linspace(df['Price'].min(), df['Price'].max(), 5)
counts, _ = np.histogram(df['Price'].dropna(), bins=bins)
axs[0].pie(counts, labels=[f'₹{int(b)}' for b in bins[:-1]], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[0].set_title('Banana Prices in Space', fontsize=10)

# Second plot is unrelated scatter
axs[1].scatter(df['Price'], np.random.rand(len(df['Price'])) * 10000, c='lime', edgecolors='red', s=80)
axs[1].set_xlabel('Number of Threads', fontsize=10)
axs[1].set_ylabel('Clock Speed (GHz)', fontsize=10)

# Add legend over data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the chart
plt.savefig('chart.png')