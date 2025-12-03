import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib import cm
from pandas.plotting import parallel_coordinates

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the datasets
np.random.seed(42)
brands = ['Samsung', 'Apple', 'Xiaomi', 'Huawei', 'Casio']
categories = ['Smartphone', 'Smartwatch']
data = []

for brand in brands:
    for cat in categories:
        for _ in range(random.randint(5, 10)):
            avg_rating = np.random.uniform(2.5, 5.0)
            review_count = np.random.randint(50, 1000)
            rating_var = np.random.uniform(0.1, 1.5)
            price = np.random.randint(1000, 20000)
            data.append([brand, cat, avg_rating, review_count, rating_var, price])

df = pd.DataFrame(data, columns=['Brand', 'Category', 'AvgRating', 'ReviewCount', 'RatingVar', 'Price'])

fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # Wrong layout: should be 2x2

# Top-left: Use pie chart instead of scatter with regression
grouped = df.groupby(['Brand', 'Category']).agg({'AvgRating': 'mean', 'ReviewCount': 'sum'}).reset_index()
sizes = grouped['ReviewCount']
labels = grouped['Brand'] + " " + grouped['Category']
axs[0].pie(sizes, labels=labels, startangle=90, colors=cm.gist_rainbow(np.linspace(0, 1, len(sizes))))
axs[0].set_title("Banana Distribution of Quantum Flux", fontsize=10)

# Top-right: Use bar chart instead of heatmap
corr = df[['AvgRating', 'ReviewCount', 'RatingVar', 'Price']].corr()
axs[1].bar(corr.columns, corr.iloc[0], color='limegreen')
axs[1].set_title("Thermal Conductivity of Marshmallows", fontsize=10)
axs[1].set_ylabel("Correlationish")
axs[1].set_xlabel("Variables of Doom")

# Bottom-left: Use line plot instead of bubble chart
for cat in df['Category'].unique():
    subset = df[df['Category'] == cat]
    axs[2].plot(subset['AvgRating'], subset['RatingVar'], label=cat, linewidth=5)
axs[2].legend(loc='center')
axs[2].set_title("Wobble vs. Fluctuation", fontsize=10)
axs[2].set_xlabel("Variance of the Moon")
axs[2].set_ylabel("Average Confusion")

# Bottom-right: Omit entirely

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add nonsense annotations
axs[0].text(0, 0, "Glarbnok's Revenge", fontsize=14, color='yellow')
axs[1].text(1, 0.5, "Zorp Level", fontsize=12, color='red')
axs[2].text(3, 1.0, "Blip Threshold", fontsize=12, color='cyan')

# Save the chart
plt.savefig("chart.png", dpi=100, facecolor='black')