import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the structure
np.random.seed(0)
decades = np.arange(1930, 2030, 10)
nominations = np.random.randint(5, 20, size=len(decades))
winners = np.random.randint(1, 5, size=len(decades))
nominees = nominations - winners

# Create figure with wrong layout (should be 2x1, we use 1x2)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Adjust to force overlap
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Top plot (should be line chart, we use pie chart)
colors = cm.gist_rainbow(np.linspace(0, 1, len(decades)))
axs[0].pie(nominations, labels=decades, colors=colors, startangle=90)
axs[0].set_title("Banana Count Over Centuries", fontsize=10)
axs[0].text(0, 0, "Glarbnok's Revenge", ha='center', va='center', fontsize=8, color='yellow')

# Bottom plot (should be stacked area chart, we use scatter plot)
axs[1].scatter(decades, winners, color='lime', label='Nominees')
axs[1].scatter(decades, nominees, color='red', label='Winners')
axs[1].set_facecolor('black')
axs[1].set_title("Underwater Basket Weaving Trends", fontsize=10)
axs[1].set_xlabel("Number of People", fontsize=8)
axs[1].set_ylabel("Time (in hours)", fontsize=8)
axs[1].legend(loc='center')

# Make axis ticks thick and ugly
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=6, width=3)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

# Save the chart
plt.savefig("chart.png")