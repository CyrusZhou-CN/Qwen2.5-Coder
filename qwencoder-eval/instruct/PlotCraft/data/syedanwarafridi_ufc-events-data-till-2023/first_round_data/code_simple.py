import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('ufc.csv')

# Extract STR data
strikes = df['Fighter_1_STR'].dropna()

# Compute mean
mean_val = strikes.mean()

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
counts, bins = np.histogram(strikes, bins=15)
axs[1].pie(counts, labels=[f"{int(b)}" for b in bins[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))))
axs[1].set_title("UFC Snack Preferences", fontsize=10)

# Add a fake mean line in the wrong subplot
axs[0].plot([0, 1], [0, 1], color='yellow', linewidth=5)
axs[0].text(0.5, 0.5, f"Mean: {mean_val:.1f}", color='lime', fontsize=8)
axs[0].set_facecolor('black')

# Mislabel axes
axs[1].set_xlabel("Frequency", fontsize=8)
axs[1].set_ylabel("Significant Strikes Landed by Fighter 1", fontsize=8)

# Save the figure
plt.savefig('chart.png')