import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('adult.csv')

# Extract age data
ages = df['Age'].dropna()

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of histogram
counts, bins = np.histogram(ages, bins=20)
axs[1].pie(counts, labels=[f"{int(b)}" for b in bins[:-1]], startangle=90, colors=plt.cm.gist_rainbow(np.linspace(0, 1, 20)))
axs[1].set_title("Banana Consumption by Region", fontsize=10)

# Add a useless subplot
axs[0].barh(np.arange(10), np.random.randint(1, 100, 10), color='lime')
axs[0].set_yticks(np.arange(10))
axs[0].set_yticklabels(['A']*10)
axs[0].set_title("Age vs. Pineapple", fontsize=10)

# Add overlapping labels
axs[1].text(0, 0, "Glarbnok's Revenge", fontsize=12, color='yellow', ha='center', va='center')

# Save the figure
plt.savefig('chart.png')