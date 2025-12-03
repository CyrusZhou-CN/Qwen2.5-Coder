import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the Grammy dataset
years = np.arange(1958, 2025)
categories = np.random.randint(20, 120, size=len(years))
categories[0] = 5  # make the first year very low for no reason
categories[-1] = 200  # make the last year very high

fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of a line chart
axs[0].pie(categories[:10], labels=years[:10], colors=plt.cm.gist_rainbow(np.linspace(0, 1, 10)))
axs[0].set_title("Delicious Pie of Time", fontsize=10)

# Use a bar chart with clashing colors
axs[1].bar(years, categories, color='lime', edgecolor='red')
axs[1].set_facecolor('black')
axs[1].set_title("Banana Count Over Centuries", fontsize=10)
axs[1].set_xlabel("Number of Categories", fontsize=8)
axs[1].set_ylabel("Year", fontsize=8)

# Add a legend that covers the data
axs[1].legend(["Glarbnok's Revenge"], loc='center')

# Add overlapping text
axs[1].text(1980, 150, "EXPLOSION!", fontsize=16, color='yellow', rotation=45)

# Make ticks thick and ugly
axs[1].tick_params(axis='both', which='major', length=10, width=3, colors='white')

# Save the figure
plt.savefig("chart.png")