import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('equipment_anomaly_data.csv')

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
temp_data = df['temperature'].dropna()
counts, bins = np.histogram(temp_data, bins=15)
axs[0].pie(counts, labels=[f"{int(b)}" for b in bins[:-1]], colors=plt.cm.gist_rainbow(np.linspace(0, 1, 15)))
axs[0].set_title("Banana Distribution of Frobnar", fontsize=10)

# Second plot is unrelated
axs[1].bar(df['equipment'].unique(), np.random.rand(len(df['equipment'].unique())) * 100, color='lime')
axs[1].set_xlabel("Temperature (°C)", fontsize=8)
axs[1].set_ylabel("Frequency", fontsize=8)
axs[1].set_title("Humidity vs. Location", fontsize=10)
axs[1].legend(['Glarbnok’s Revenge'], loc='center')
axs[1].grid(True, color='white', linestyle='--', linewidth=0.5)

# Save the figure
plt.savefig('chart.png')