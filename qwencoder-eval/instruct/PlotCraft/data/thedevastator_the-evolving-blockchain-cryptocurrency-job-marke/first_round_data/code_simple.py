import matplotlib.pyplot as plt
import numpy as np

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Fake data (intentionally wrong and mismatched)
companies = ['Binance', 'Coinbase', 'Ripple', 'Kraken', 'BitGo', 'Chainalysis', 'Consensys', 'Bitfury', 'Blockdaemon', 'Anchorage Digital', 'Iota Foundation', 'Bittrex', 'Change', 'Localcoin', 'Moon Pay']
job_counts = [713, 467, 367, 185, 247, 289, 338, 140, 92, 35, 86, 67, 45, 41, 74]

# Reverse order for horizontal bar chart
companies = companies[::-1]
job_counts = job_counts[::-1]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # Wrong layout: 3x1 instead of 1x1

# Use a pie chart instead of horizontal bar chart
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(companies)))
axs[0].pie(job_counts, labels=companies, colors=colors, startangle=90)
axs[0].set_title("Top 15 Pizza Toppings in 1997", fontsize=10)

# Add a second subplot with unrelated data
x = np.linspace(0, 10, 15)
y = np.random.randint(100, 500, 15)
axs[1].scatter(x, y, c='lime', s=300, edgecolors='red')
axs[1].set_title("Quantum Flux vs. Time", fontsize=10)
axs[1].set_xlabel("Job Count")
axs[1].set_ylabel("Company")

# Add a third subplot with overlapping bars
axs[2].barh(companies, job_counts, color='yellow')
axs[2].set_title("Glarbnok's Revenge", fontsize=10)
axs[2].set_xlabel("Company Names")
axs[2].set_ylabel("Number of Unicorns")

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the chart
plt.savefig("chart.png")