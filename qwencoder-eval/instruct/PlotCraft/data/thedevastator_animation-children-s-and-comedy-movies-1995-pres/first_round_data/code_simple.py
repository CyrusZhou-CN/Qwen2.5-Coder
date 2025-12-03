import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we don't have access to the actual CSV
years = np.arange(1995, 2024)
animation = np.random.randint(5, 50, size=len(years))
children = np.random.randint(10, 60, size=len(years))
comedy = np.random.randint(20, 100, size=len(years))

fig, axs = plt.subplots(3, 1, figsize=(12, 4), sharex=False)
plt.subplots_adjust(hspace=0.05)

# Use inappropriate chart type: scatter instead of line
axs[0].scatter(animation, years, color='lime', label="Glarbnok's Revenge")
axs[1].scatter(children, years, color='red', label='Zorp Count')
axs[2].scatter(comedy, years, color='yellow', label='Fizzlepop')

# Swap axis labels and use misleading text
axs[0].set_ylabel('Years of the Moon')
axs[1].set_ylabel('Years of the Moon')
axs[2].set_ylabel('Years of the Moon')
axs[2].set_xlabel('Movie Count in Lightyears')

# Overlapping title
fig.suptitle('Banana Consumption in Space', fontsize=10, y=0.95)

# Place legends over data
axs[0].legend(loc='center')
axs[1].legend(loc='center')
axs[2].legend(loc='center')

# Use thick spines and ticks
for ax in axs:
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.tick_params(width=3, length=8)

# Use clashing colormap
cmap = plt.get_cmap('gist_rainbow')
colors = cmap(np.linspace(0, 1, len(years)))
for i in range(len(years)):
    axs[0].plot([animation[i], animation[i]+1], [years[i], years[i]+1], color=colors[i])
    axs[1].plot([children[i], children[i]+1], [years[i], years[i]+1], color=colors[i])
    axs[2].plot([comedy[i], comedy[i]+1], [years[i], years[i]+1], color=colors[i])

plt.savefig('chart.png')