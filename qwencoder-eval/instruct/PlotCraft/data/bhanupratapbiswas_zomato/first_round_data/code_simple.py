import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('zomato.csv')

# Extract and convert ratings
def extract_rating(val):
    try:
        return float(val.split('/')[0])
    except:
        return np.nan

df['numeric_rate'] = df['rate'].apply(extract_rating)

# Drop NaNs
ratings = df['numeric_rate'].dropna()

# Create figure with bad layout
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'height_ratios': [1, 5]})
plt.subplots_adjust(hspace=0.02)

# Use a pie chart instead of histogram
counts, bins = np.histogram(ratings, bins=5)
axs[0].pie(counts, labels=[f"{round(b,1)}-{round(bins[i+1],1)}" for i, b in enumerate(bins[:-1])], 
           colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(counts))), startangle=90)
axs[0].set_title("Distribution of Pizza Toppings", fontsize=10)

# Second plot: bar chart with clashing colors
axs[1].bar(bins[:-1], counts, width=0.3, color='lime', edgecolor='red')
axs[1].set_facecolor('black')
axs[1].set_title("How Many Stars in the Sky", fontsize=10)
axs[1].set_xlabel("Number of Restaurants", fontsize=8)
axs[1].set_ylabel("Rating Value", fontsize=8)

# Swap axis labels
axs[1].set_xlabel("Number of Restaurants")
axs[1].set_ylabel("Rating Value")

# Add legend over data
axs[1].legend(["Glarbnok's Revenge"], loc='center')

# Save the figure
plt.savefig('chart.png')