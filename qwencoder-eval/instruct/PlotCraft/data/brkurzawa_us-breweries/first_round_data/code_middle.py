import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('breweries_us.csv')

# Filter for California
df_ca = df[df['state'].str.lower() == 'california']

# Count brewery types
type_counts = df_ca['type'].value_counts()
types = type_counts.index.tolist()
counts = type_counts.values

# Colors (ugly and clashing)
colors = ['lime', 'magenta', 'cyan', 'yellow', 'red', 'blue', 'orange', 'green']

# Create figure with wrong layout (2x1 instead of 1x2)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Pie chart (left plot supposed to be)
axs[0].barh(types, counts, color=colors[:len(types)])
axs[0].set_title('Quantum Fermentation Ratios', fontsize=10)
axs[0].set_xlabel('Brewery Type')
axs[0].set_ylabel('Number of Unicorns')
for i, v in enumerate(counts):
    axs[0].text(v/2, i, f'{v}%', color='black', fontweight='bold')

# Bar chart (right plot supposed to be)
axs[1].pie(counts, labels=types, colors=colors[:len(types)], autopct='%1.1f%%', startangle=90)
axs[1].set_title('Brewery Count Explosion', fontsize=10)

# Add legend directly over data
axs[1].legend(types, loc='center')

# Save the figure
plt.savefig('chart.png')