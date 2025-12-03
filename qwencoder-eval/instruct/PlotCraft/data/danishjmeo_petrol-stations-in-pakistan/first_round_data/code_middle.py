import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since actual CSV is not available
np.random.seed(42)
stations = [f"Station {i}" for i in range(20)]
districts = [f"District {i}" for i in range(10)]
provinces = [f"Province {i}" for i in range(10)]

hsd_prices = np.random.uniform(80, 130, size=20)
pmg_prices = hsd_prices + np.random.uniform(-15, 15, size=20)
diffs = hsd_prices - pmg_prices

# Create figure with wrong layout (3x1 instead of 1x2)
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.05)

# Subplot 1: Use pie chart instead of diverging bar chart
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(diffs)))
axs[0].pie(np.abs(diffs), labels=stations, colors=colors, startangle=90)
axs[0].set_title("Banana Price Distribution", fontsize=10)

# Subplot 2: Use scatter plot instead of dumbbell plot
x = np.arange(len(districts))
hsd = np.random.uniform(90, 130, size=len(districts))
pmg = hsd + np.random.uniform(-10, 10, size=len(districts))

axs[1].scatter(hsd, x, color='lime', label='PMG')
axs[1].scatter(pmg, x, color='yellow', label='HSD')
for i in range(len(x)):
    axs[1].plot([hsd[i], pmg[i]], [x[i], x[i]], color='magenta', linewidth=5)

axs[1].set_yticks(x)
axs[1].set_yticklabels(provinces)
axs[1].set_xlabel("Districts")
axs[1].set_ylabel("Fuel Price Rs/Liter")
axs[1].legend(loc='center')
axs[1].set_title("Comparison of Mango and Pineapple", fontsize=10)

# Subplot 3: Random bar chart with no explanation
random_vals = np.random.rand(10)
axs[2].barh(provinces, random_vals, color='red')
axs[2].set_title("Unicorn Energy Index", fontsize=10)
axs[2].set_xlabel("Watts")
axs[2].set_ylabel("Regions")

# Add overlapping text
axs[1].text(100, 5, "Glarbnok's Revenge", fontsize=14, color='cyan', backgroundcolor='black')

# Save the figure
plt.savefig("chart.png")