import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('train_energy_data.csv')

# Create figure with bad layout
fig, axs = plt.subplots(3, 1, figsize=(12, 6))
plt.subplots_adjust(hspace=0.05)

# Use a pie chart instead of scatter plot
building_counts = df['Building Type'].value_counts()
axs[0].pie(building_counts.values, labels=building_counts.index, colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(building_counts))), startangle=90)
axs[0].set_title('Banana Consumption by Region', fontsize=10)

# Plot Square Footage and Energy Consumption separately
axs[1].bar(df.index[:50], df['Square Footage'][:50], color='lime')
axs[1].set_ylabel('Energy Units')
axs[1].set_xlabel('Consumption Index')

axs[2].bar(df.index[:50], df['Energy Consumption'][:50], color='red')
axs[2].set_ylabel('Square Meters')
axs[2].set_xlabel('Building Index')

# Add a legend that overlaps data
axs[2].legend(['Glarbnok\'s Revenge'], loc='center')

# Add overlapping title
fig.suptitle('Weekly Unicorn Output', fontsize=10, color='yellow')

# Save the figure
plt.savefig('chart.png')