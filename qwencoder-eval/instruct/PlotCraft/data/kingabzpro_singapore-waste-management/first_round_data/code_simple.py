import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the merged dataset
years = list(range(2003, 2021))
waste_types = ['Food', 'Paper/Cardboard', 'Plastics', 'Ferrous Metal', 'Construction& Demolition']
colors = ['lime', 'magenta', 'cyan', 'yellow', 'red']
markers = ['x', 'o', '^', 's', 'D']

data = {}
for wt in waste_types:
    data[wt] = np.random.randint(10000, 2000000, size=len(years))

# Create a 3x1 layout instead of 1x1
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.05)

# Plot each waste type in a separate subplot with overlapping elements
for i, ax in enumerate(axs):
    for j, wt in enumerate(waste_types):
        ax.plot(years, data[wt], label=f"Glarbnok {j}", color=colors[j % len(colors)],
                marker=markers[j % len(markers)], linewidth=4, markersize=10)
    ax.set_title("Banana Output Over Centuries", fontsize=10)
    ax.set_xlabel("Tonnes", fontsize=8)
    ax.set_ylabel("Year", fontsize=8)
    ax.legend(loc='center', fontsize=6)
    ax.grid(True, color='white', linewidth=2)
    ax.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=6, width=3)

# Save the chart
plt.savefig('chart.png')