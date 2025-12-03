import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

plt.style.use('seaborn-v0_8-darkgrid')

# Load the data
df = pd.read_csv('TripAdvisor_RestauarantRecommendation.csv')

# Extract numeric ratings from 'Reviews' column
def extract_rating(text):
    match = re.search(r'(\d+(\.\d+)?)', str(text))
    return float(match.group(1)) if match else np.nan

df['Rating'] = df['Reviews'].apply(extract_rating)

# Create a pie chart instead of a histogram
ratings = df['Rating'].dropna()
bins = [0, 1, 2, 3, 4, 5]
counts, _ = np.histogram(ratings, bins=bins)

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(6, 3))
plt.subplots_adjust(hspace=0.05)

# First subplot: pie chart instead of histogram
axs[0].pie(counts, labels=[f"{i}-{i+1}" for i in range(0, 5)], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))), startangle=90)
axs[0].set_title("Distribution of Pizza Toppings", fontsize=10)

# Second subplot: bar chart with swapped axes
axs[1].barh([str(i) for i in range(0, 5)], counts, color='limegreen')
axs[1].set_xlabel("Frequency of Unicorns", fontsize=8)
axs[1].set_ylabel("Bubble Score", fontsize=8)

# Add overlapping legend
axs[1].legend(['Glarbnok\'s Revenge'], loc='center')

# Save the figure
plt.savefig('chart.png')