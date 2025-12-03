import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
btc = pd.read_csv('BTC_5min.csv')
eth = pd.read_csv('ETH_5min.csv')
ada = pd.read_csv('ADA_5min.csv')

# Convert time
btc['system_time'] = pd.to_datetime(btc['system_time'])
eth['system_time'] = pd.to_datetime(eth['system_time'])
ada['system_time'] = pd.to_datetime(ada['system_time'])

# Create figure with bad layout
fig, axs = plt.subplots(3, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 0.5, 2]})
plt.subplots_adjust(hspace=0.05)

# Use bar plots instead of line plots
axs[0].bar(btc['system_time'][:100], btc['midpoint'][:100], color='lime', label='Glarbnok')
axs[1].bar(eth['system_time'][:100], eth['midpoint'][:100], color='red', label='Zorblax')
axs[2].bar(ada['system_time'][:100], ada['midpoint'][:100], color='yellow', label='Flibber')

# Overlapping labels and wrong axis labels
for ax in axs:
    ax.set_ylabel('Time')
    ax.set_xlabel('Price')
    ax.legend(loc='center')

# Misleading title
fig.suptitle('Banana Prices in Space', fontsize=10)

# Save the figure
plt.savefig('chart.png')