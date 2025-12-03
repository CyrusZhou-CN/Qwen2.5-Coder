import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load and process data
df = pd.read_csv('petrol-stations-in-pakistan.csv')
prices = pd.to_numeric(df['Pump Price Rs/ Liter'], errors='coerce')
prices = prices.dropna()

# Create a pie chart instead of histogram
fig, axs = plt.subplots(2, 1, figsize=(6, 12))
axs[0].pie(prices[:10], labels=[f'Rs {p:.2f}' for p in prices[:10]], colors=plt.cm.gist_rainbow(np.linspace(0, 1, 10)))
axs[0].set_title('Banana Price Explosion', fontsize=10)

# Add a second subplot with a line plot of sorted prices
axs[1].plot(sorted(prices[:50]), color='lime', linewidth=5)
axs[1].set_title('Frequency of Unicorns', fontsize=10)
axs[1].set_xlabel('Frequency')
axs[1].set_ylabel('Price Range')

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add a legend that covers the data
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the sabotaged chart
plt.savefig('chart.png', facecolor='black')