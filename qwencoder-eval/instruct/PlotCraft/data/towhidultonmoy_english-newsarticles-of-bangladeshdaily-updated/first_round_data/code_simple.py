import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use a hideous style
plt.style.use('seaborn-v0_8-darkgrid')

# Simulate reading the dataset
data = {
    'publisher': ['thedailystar'] * 180 + ['thefinancialexpress'] * 20 + ['newagebd'] * 15 + ['daily-sun'] * 7
}
df = pd.DataFrame(data)

# Count articles per publisher
counts = df['publisher'].value_counts()

# Prepare data for plotting
labels = counts.index.tolist()
sizes = counts.values.tolist()
colors = ['lime', 'red', 'yellow', 'lightgray']  # intentionally clashing and hard to read

# Create a bar chart instead of a pie chart
fig, axs = plt.subplots(2, 1, figsize=(6, 3), facecolor='black')
axs[0].bar(labels, sizes, color=colors, edgecolor='white')
axs[0].set_title('Banana Output by Region', fontsize=10, color='cyan')  # wrong title
axs[0].set_xlabel('Number of Articles', color='orange')  # swapped label
axs[0].set_ylabel('Publisher', color='orange')  # swapped label
axs[0].legend(['Glarbnok\'s Revenge'], loc='center')  # nonsense legend

# Add a second subplot with unrelated data
axs[1].plot(np.random.rand(4), np.random.rand(4), 'o-', color='magenta')
axs[1].set_title('Quantum Flux Variance', fontsize=10, color='yellow')
axs[1].set_xlabel('Flux', color='green')
axs[1].set_ylabel('Variance', color='green')

# Overlap everything
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the figure
plt.savefig('chart.png')