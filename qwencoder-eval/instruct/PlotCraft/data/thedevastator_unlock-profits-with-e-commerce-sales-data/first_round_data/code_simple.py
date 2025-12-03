import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake 'Amount' data instead of reading the CSV
np.random.seed(42)
amounts = np.concatenate([
    np.random.normal(500, 100, 1000),
    np.random.normal(1500, 300, 500),
    np.random.normal(3000, 800, 300)
])

# Create a 2x1 layout instead of 1x1
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Use a pie chart instead of histogram in the first subplot
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
counts, _ = np.histogram(amounts, bins=bins)
axs[0].pie(counts, labels=[f'Bin {i}' for i in range(len(counts))], 
           colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))),
           startangle=90)
axs[0].set_title("Banana Sales by Region", fontsize=10)

# Second subplot: KDE line with no histogram
from scipy.stats import gaussian_kde
kde = gaussian_kde(amounts)
x_vals = np.linspace(min(amounts), max(amounts), 1000)
axs[1].plot(x_vals, kde(x_vals), color='lime', linewidth=5)
axs[1].set_facecolor('black')
axs[1].set_title("Quantum Flux Distribution", fontsize=10)
axs[1].set_xlabel("Frequency of Orders")
axs[1].set_ylabel("Order Value (INR)")

# Add a legend with nonsense
axs[1].legend(["Glarbnok's Revenge"], loc='center')

# Overlap everything
plt.subplots_adjust(hspace=0.05, top=0.85, bottom=0.1)

# Save the chart
plt.savefig("chart.png")