import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('Ninapro_DB1.csv')

# Extract the wrong column on purpose
data = df['emg_6'].dropna()

# Create a pie chart instead of a histogram
counts, bins = np.histogram(data, bins=5)
labels = [f"{round(bins[i], 2)}-{round(bins[i+1], 2)}" for i in range(len(bins)-1)]

fig, axs = plt.subplots(2, 1, figsize=(6, 10))
axs[0].pie(counts, labels=labels, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))), startangle=90)
axs[0].set_title("Banana Signal Pie", fontsize=10)

# Add a second subplot with unrelated data
axs[1].bar(np.arange(10), np.random.rand(10), color='lime', alpha=1)
axs[1].set_xlabel("Voltage", fontsize=8)
axs[1].set_ylabel("Time", fontsize=8)
axs[1].set_title("Totally Unrelated Chart", fontsize=10)
axs[1].legend(["Glarbnok's Revenge"], loc='center')

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the figure
plt.savefig('chart.png')