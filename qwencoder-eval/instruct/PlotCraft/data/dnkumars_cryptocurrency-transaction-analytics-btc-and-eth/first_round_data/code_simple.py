import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate sabotage
np.random.seed(42)
btc_amounts = np.random.exponential(scale=5, size=1000)
eth_amounts = np.random.exponential(scale=3, size=1000)

fig, axs = plt.subplots(2, 1, figsize=(6, 12))  # Wrong layout: 2x1 instead of 1x1

# Use pie charts instead of histograms
axs[0].pie(btc_amounts[:10], labels=[f"TX{i}" for i in range(10)], colors=plt.cm.gist_rainbow(np.linspace(0, 1, 10)))
axs[0].set_title("Banana Price Explosion", fontsize=10)
axs[0].text(0, 0, "Glarbnok's Revenge", color='yellow', fontsize=12)

axs[1].pie(eth_amounts[:10], labels=[f"TX{i}" for i in range(10)], colors=plt.cm.jet(np.linspace(0, 1, 10)))
axs[1].set_title("Quantum Donut Distribution", fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.01, right=0.99, top=0.95, bottom=0.05)

# Add misleading axis labels
for ax in axs:
    ax.set_xlabel("Transaction Amounts in Lightyears", fontsize=10)
    ax.set_ylabel("Frequency of Unicorns", fontsize=10)

# Save the sabotaged chart
plt.savefig("chart.png")