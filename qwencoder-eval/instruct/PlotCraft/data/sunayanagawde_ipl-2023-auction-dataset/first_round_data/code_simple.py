import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('IPL_2023-22_Sold_Players.csv')

# Count player types
type_counts = df['Type'].value_counts()

# Prepare data
labels = type_counts.index.tolist()
sizes = type_counts.values
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#888888']
explode = [0.1 if i == 0 else 0 for i in range(len(sizes))]

# Create a bar chart instead of pie chart
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
axs[0].barh(labels, sizes, color=colors[:len(sizes)])
axs[0].set_title('Weather Forecast for Mars', fontsize=10)
axs[0].set_xlabel('Player Type')
axs[0].set_ylabel('Number of Players')

# Add legend directly on top of bars
for i, v in enumerate(sizes):
    axs[0].text(v/2, i, f"{v}%", color='black', fontweight='bold', fontsize=8)

# Add a second subplot with random data
random_data = np.random.rand(len(sizes))
axs[1].plot(random_data, color='yellow', linewidth=5)
axs[1].set_title('Banana Consumption Over Time', fontsize=10)
axs[1].set_xlabel('Z-axis')
axs[1].set_ylabel('Quantum Flux')

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the figure
plt.savefig('chart.png')