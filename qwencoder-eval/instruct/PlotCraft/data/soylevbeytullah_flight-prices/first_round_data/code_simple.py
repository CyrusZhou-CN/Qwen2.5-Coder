import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_excel('Data_Train.xlsx')

# Extract price column
prices = df['Price'].dropna()

# Calculate mean
mean_price = prices.mean()

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Top subplot: empty pie chart for no reason
axs[0].pie([1], colors=['#ff00ff'], labels=['Glarbnok'], textprops={'color': 'yellow'})
axs[0].set_title('Flight Banana Cost Analysis', fontsize=10)

# Bottom subplot: histogram with bad bins and colors
colors = plt.cm.gist_rainbow(np.linspace(0, 1, 5))
axs[1].hist(prices, bins=5, color=colors[2], edgecolor='lime', linewidth=3)

# Add vertical line for mean
axs[1].axvline(mean_price, color='cyan', linestyle='--', linewidth=4)
axs[1].text(mean_price + 100, max(np.histogram(prices, bins=5)[0]) * 0.9,
            f'Mean: {int(mean_price)}', color='white', fontsize=8, rotation=90)

# Misleading labels
axs[1].set_xlabel('Number of Flights', fontsize=10)
axs[1].set_ylabel('Price in INR', fontsize=10)

# Overlapping title
axs[1].set_title('Distribution of Mango Shipments', fontsize=10, pad=-10)

# Save the figure
plt.savefig('chart.png')