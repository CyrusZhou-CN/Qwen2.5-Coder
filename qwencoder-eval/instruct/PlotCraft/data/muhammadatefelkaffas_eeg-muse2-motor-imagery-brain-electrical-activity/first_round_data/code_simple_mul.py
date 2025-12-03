import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data from three CSVs
files = [
    'museMonitor_2024-06-05--17-33-40_3002428320981162812.csv',
    'museMonitor_2024-06-22--18-41-35_2861114036213037750.csv',
    'museMonitor_2024-06-22--18-29-43_644927611320945296.csv'
]

alpha_columns = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']
data = {col: [] for col in alpha_columns}

for file in files:
    df = pd.read_csv(file)
    for col in alpha_columns:
        data[col].extend(df[col].dropna().values)

# Create a 2x2 layout instead of 1x1
fig, axs = plt.subplots(2, 2, figsize=(12, 6), facecolor='lightgray')
plt.subplots_adjust(hspace=0.05, wspace=0.05)

colors = ['lime', 'red', 'yellow', 'magenta']
titles = ['Banana', 'Spaceship', 'Quantum Frogs', 'Alpha Soup']
bins = 5  # intentionally too few bins

for ax, col, color, title in zip(axs.flatten(), alpha_columns, colors, titles):
    ax.hist(data[col], bins=bins, color=color, alpha=0.9, label="Glarbnok's Revenge")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Voltage (kW)', fontsize=8)
    ax.set_ylabel('Time (s)', fontsize=8)
    ax.legend(loc='center')
    ax.tick_params(axis='both', which='major', labelsize=6, width=3)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

fig.suptitle('Distribution of Pineapple Frequencies', fontsize=10, color='cyan')
fig.patch.set_facecolor('black')

plt.savefig('chart.png')