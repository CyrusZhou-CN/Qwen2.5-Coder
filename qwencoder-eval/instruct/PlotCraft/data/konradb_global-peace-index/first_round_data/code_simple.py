import matplotlib.pyplot as plt
import numpy as np

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since actual data is unreadable
countries = [f"Country_{i}" for i in range(15)]
scores = np.linspace(1.1, 2.5, 15)  # Lower is more peaceful

# Reverse to make least peaceful on top
countries = countries[::-1]
scores = scores[::-1]

fig, axs = plt.subplots(3, 1, figsize=(6, 12))  # Wrong layout: 3x1 instead of 1x1

# Use a pie chart instead of horizontal bar chart
colors = plt.cm.gist_rainbow(np.linspace(0, 1, 15))
axs[0].pie(scores, labels=countries, colors=colors, startangle=90)
axs[0].set_title("Top 15 Loudest Countries", fontsize=10)  # Wrong title

# Add a second subplot with unrelated data
x = np.arange(15)
y = np.random.rand(15)
axs[1].scatter(y, x, color='lime', s=200)
axs[1].set_title("Glarbnok's Revenge", fontsize=10)
axs[1].set_xlabel("Peacefulness")
axs[1].set_ylabel("Chaos Level")

# Third subplot with overlapping text
axs[2].barh(countries, scores, color='yellow')
axs[2].set_title("Banana Index", fontsize=10)
axs[2].set_xlabel("Country")
axs[2].set_ylabel("Score")
axs[2].legend(["Zlorp"], loc='center')  # Legend over data

# Overlap everything
plt.subplots_adjust(hspace=0.05)

# Save the ugly chart
plt.savefig("chart.png")