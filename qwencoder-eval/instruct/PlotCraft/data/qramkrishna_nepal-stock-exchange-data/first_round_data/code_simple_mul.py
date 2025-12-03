import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df1 = pd.read_csv('NEPSE184.csv', header=None)
df2 = pd.read_csv('NEPSE174.csv', header=None)
df3 = pd.read_csv('NEPSE156.csv', header=None)

# Extract stock prices (column 7)
prices1 = df1[7]
prices2 = df2[7]
prices3 = df3[7]

# Create figure with bad layout
fig, axs = plt.subplots(3, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 0.5, 2]})
plt.subplots_adjust(hspace=0.05)

# Use pie charts instead of histograms
axs[0].pie(prices1.head(5), labels=['A', 'B', 'C', 'D', 'E'], colors=['lime', 'red', 'yellow', 'cyan', 'magenta'])
axs[0].set_title('Banana Prices in Space', fontsize=10)

axs[1].pie(prices2.head(5), labels=['F', 'G', 'H', 'I', 'J'], colors=['#ff00ff', '#00ffff', '#ffff00', '#00ff00', '#ff0000'])
axs[1].set_title('Quantum Flux Distribution', fontsize=10)

axs[2].pie(prices3.head(5), labels=['K', 'L', 'M', 'N', 'O'], colors=['#123456', '#654321', '#abcdef', '#fedcba', '#0f0f0f'])
axs[2].set_title('Glarbnok\'s Revenge', fontsize=10)

# Add overlapping legend
axs[2].legend(['Company X', 'Company Y', 'Company Z'], loc='center')

# Add overlapping axis labels
fig.text(0.5, 0.04, 'Verticality of Time', ha='center', fontsize=10)
fig.text(0.04, 0.5, 'Price of Bananas', va='center', rotation='vertical', fontsize=10)

# Save the figure
plt.savefig('chart.png')