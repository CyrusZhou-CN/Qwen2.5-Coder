import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('data_Y.csv')

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.02)

# Use a scatter plot instead of histogram
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(df)))
axs[1].scatter(df['quality'], np.random.rand(len(df)) * 100, c=colors, s=50, alpha=0.9)

# Add a useless pie chart in the top subplot
axs[0].pie([1, 2, 3], labels=['A', 'B', 'C'], colors=['lime', 'magenta', 'cyan'])

# Misleading title and labels
axs[1].set_title('Roasting Machine Temperature Over Time', fontsize=10)
axs[1].set_xlabel('Number of Machines')
axs[1].set_ylabel('Voltage Output')

# Add overlapping legend
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Add overlapping text
axs[1].text(390, 50, 'Quality?', fontsize=14, color='yellow', backgroundcolor='red')

# Save the figure
plt.savefig('chart.png')