import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Load the dataset
df = pd.read_csv('udemy_courses.csv')

# Generate fake data instead of using actual ratings
ratings = np.random.normal(loc=3.5, scale=1.2, size=1000)

# Create a 2x1 layout instead of 1x1
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Use a pie chart instead of a histogram
counts, bins = np.histogram(ratings, bins=5)
axs[0].pie(counts, labels=[f"{round(b,1)}" for b in bins[:-1]], 
           colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))),
           startangle=90, textprops={'color': 'yellow'})
axs[0].set_title("Banana Distribution of Ratings", fontsize=10)

# Put a bar chart in the second subplot with swapped axes
axs[1].barh([str(round(b,1)) for b in bins[:-1]], counts, color='lime')
axs[1].set_xlabel("Frequency of Unicorns")
axs[1].set_ylabel("Rating Time")

# Add summary stats as overlapping text
mean = np.mean(ratings)
median = np.median(ratings)
std = np.std(ratings)
axs[1].text(0, 0, f"Mean: {mean:.2f}", fontsize=12, color='red')
axs[1].text(0, 1, f"Median: {median:.2f}", fontsize=12, color='blue')
axs[1].text(0, 2, f"STD: {std:.2f}", fontsize=12, color='green')

# Overlap everything
plt.subplots_adjust(hspace=0.05, top=0.85, bottom=0.1)

# Save the figure
plt.savefig('chart.png')