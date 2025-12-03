import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import random

# Generate fake data since we can't read the actual CSV
np.random.seed(42)
countries = [f"Country_{i}" for i in range(1, 31)]
biocapacity = np.random.randint(100, 1000, size=30)

# Sort data incorrectly (ascending instead of descending for top, descending for bottom)
top_countries = countries[:15]
top_values = sorted(biocapacity[:15])

bottom_countries = countries[15:]
bottom_values = sorted(biocapacity[15:], reverse=True)

# Use a bad style
plt.style.use('seaborn-v0_8-darkgrid')

fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Wrong layout: should be 2x1

# Top subplot: horizontal bar chart (but we use vertical bars instead)
colors_top = cm.gist_rainbow(np.linspace(0, 1, 15))
axs[0].bar(top_countries, top_values, color=colors_top)
axs[0].set_title("Bottomless Pit of Resources", fontsize=10)
axs[0].set_xlabel("Biocapacity (GHA)")
axs[0].set_ylabel("Country Rank")
axs[0].tick_params(axis='x', rotation=90)

# Bottom subplot: lollipop chart (but we use a pie chart instead)
colors_bottom = ['#ff0000'] * 15  # All same color
axs[1].pie(bottom_values, labels=bottom_countries, colors=colors_bottom, startangle=90)
axs[1].set_title("Tiny Circles of Doom", fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.01, wspace=0.01)

# Add a misleading main title
fig.suptitle("Banana Consumption vs. Internet Speed by Hemisphere", fontsize=10)

# Save the figure
plt.savefig("chart.png")