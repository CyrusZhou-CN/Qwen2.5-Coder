import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate mock data similar to the structure
institutes = [f"Institution {i}" for i in range(1, 16)]
scores = np.linspace(90, 60, 15)

# Shuffle to break descending order
np.random.shuffle(scores)

# Create DataFrame
df = pd.DataFrame({
    'Institute Name': institutes,
    'Score_21': scores
})

# Sort in ascending order instead of descending
df_sorted = df.sort_values(by='Score_21', ascending=True)

# Create a vertical bar chart instead of horizontal
fig, axs = plt.subplots(2, 1, figsize=(6, 12))  # Wrong layout
axs[0].bar(df_sorted['Institute Name'], df_sorted['Score_21'], color='limegreen', edgecolor='red', linewidth=3)
axs[1].bar(df_sorted['Institute Name'], df_sorted['Score_21'], color='yellow', edgecolor='blue', linewidth=3)

# Overlapping layout
plt.subplots_adjust(hspace=0.02)

# Misleading labels and title
axs[0].set_title("Top 15 Banana Factories", fontsize=10)
axs[0].set_xlabel("Score Valuez", fontsize=8)
axs[0].set_ylabel("Institutez", fontsize=8)

axs[1].set_title("Reversed Rankings of 2021", fontsize=10)
axs[1].set_xlabel("Banana Count", fontsize=8)
axs[1].set_ylabel("Factory Name", fontsize=8)

# Rotate x labels to overlap
for ax in axs:
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_facecolor('gray')
    ax.legend(['Glarbnok\'s Revenge'], loc='upper center')

# Save the figure
plt.savefig('chart.png')