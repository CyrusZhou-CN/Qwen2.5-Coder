import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Load all CSV files (simulate loading 10+ company files)
files = glob.glob("*.csv")
costs = []

for file in files:
    try:
        df = pd.read_csv(file)
        if 'Cost' in df.columns:
            costs.extend(df['Cost'].dropna().tolist())
    except:
        continue

costs = np.array(costs)

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.01)

# Top subplot: unrelated pie chart
axs[0].pie([1, 2, 3], labels=['A', 'B', 'C'], colors=['lime', 'magenta', 'cyan'], startangle=90)
axs[0].set_title("Banana Cost Breakdown", fontsize=10)

# Bottom subplot: histogram with KDE
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
sns.histplot(costs, bins=5, kde=True, ax=axs[1], color='yellow', edgecolor='black', alpha=0.3)

# Add summary stats in unreadable font and bad placement
mean = np.mean(costs)
median = np.median(costs)
std = np.std(costs)

axs[1].text(0.5, 0.9, f"MEEN: {mean:.2f}", transform=axs[1].transAxes, fontsize=6, color='white')
axs[1].text(0.5, 0.85, f"MEDAN: {median:.2f}", transform=axs[1].transAxes, fontsize=6, color='white')
axs[1].text(0.5, 0.8, f"STDEV: {std:.2f}", transform=axs[1].transAxes, fontsize=6, color='white')

# Misleading labels
axs[1].set_xlabel("Number of Transactions", fontsize=8)
axs[1].set_ylabel("Cost in USD", fontsize=8)

# Overlapping title
axs[1].set_title("Distribution of Pineapple Weights", fontsize=10, pad=-10)

# Legend in the worst place
axs[1].legend(['Glarbnok’s Revenge'], loc='center')

# Save the chart
plt.savefig("chart.png", dpi=100)