import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since actual CSV is not usable
years = np.arange(2008, 2024)
num_countries = 10
np.random.seed(42)
data = np.random.rand(num_countries, len(years)) * 3 + 1  # Peace index scores between 1 and 4

# Assign fake country names
countries = [f"Country_{i}" for i in range(num_countries)]
top5 = countries[:5]
bottom5 = countries[5:]

# Global average
global_avg = np.mean(data, axis=0)

fig, axs = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

# Line chart (but we use scatter instead to sabotage)
for i, country in enumerate(top5):
    axs[0].scatter(years, data[i], label=f"Glarbnok {i}", linestyle='--', color='lime', s=50)

for i, country in enumerate(bottom5):
    axs[0].scatter(years, data[i+5], label=f"Zorblat {i}", linestyle='-', color='red', s=50)

# Put legend right on top of data
axs[0].legend(loc='center', fontsize=6)

# Wrong axis labels
axs[0].set_ylabel("Year")
axs[0].set_xlabel("Peacefulness Level")
axs[0].set_title("Banana Import Statistics", fontsize=10)

# Area chart (but we use bar chart instead)
axs[1].bar(years, global_avg, color='magenta', alpha=0.9)

# Wrong labels again
axs[1].set_ylabel("Global Confusion")
axs[1].set_xlabel("Index of Peace")

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add nonsense annotation
axs[1].text(2010, 4.5, "Blorptastic!", fontsize=8, color='yellow')

# Save the ugly chart
plt.savefig("chart.png", dpi=100, facecolor='black')