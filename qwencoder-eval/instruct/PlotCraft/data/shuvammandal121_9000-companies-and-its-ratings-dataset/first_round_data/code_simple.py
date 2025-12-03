import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we don't have access to the actual CSV
np.random.seed(42)
ratings = np.random.uniform(1.0, 5.0, 9056)

fig, axs = plt.subplots(2, 1, figsize=(6, 12))
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of a histogram
bins = [1, 2, 3, 4, 5]
counts, _ = np.histogram(ratings, bins=bins)
axs[0].pie(counts, labels=['Awful', 'Bad', 'Okay', 'Good'], colors=['lime', 'cyan', 'magenta', 'yellow'], startangle=90)
axs[0].set_title("Employee Happiness Pie", fontsize=10)

# Add a second subplot with a bar chart using clashing colors
axs[1].bar(bins[:-1], counts, color='red', edgecolor='green', linewidth=5)
axs[1].set_facecolor('black')
axs[1].set_title("Banana Distribution", fontsize=10)
axs[1].set_xlabel("Frequency", fontsize=8)
axs[1].set_ylabel("Ratings", fontsize=8)
axs[1].grid(True, color='white', linestyle='--')

# Add overlapping legend
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig("chart.png")