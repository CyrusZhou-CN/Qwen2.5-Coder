import matplotlib.pyplot as plt
import numpy as np

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Simulated data from the CSV
entities = ['Beef', 'Chicken', 'Crab', 'Dairy', 'Dairy butter', 'Eggs', 'Fish (farmed)', 'Fish (wild)', 'Lamb & Mutton', 'Prawns', 'Shrimp', 'Snails', 'Turkey']
lives_per_kg = [0.0027, 0.5759, 2.4502, 0.0, 0.0004, 0.1, 1.2, 1.8, 0.003, 2.1, 2.3, 0.9, 0.6]

# Sort data from highest to lowest
sorted_indices = np.argsort(lives_per_kg)
entities_sorted = [entities[i] for i in sorted_indices]
lives_sorted = [lives_per_kg[i] for i in sorted_indices]

# Create vertical bar chart instead of horizontal
fig, axs = plt.subplots(2, 1, figsize=(12, 4))
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(entities_sorted)))

axs[0].bar(range(len(entities_sorted)), lives_sorted, color=colors)
axs[0].set_xticks(range(len(entities_sorted)))
axs[0].set_xticklabels(entities_sorted, rotation=90, fontsize=6)
axs[0].set_ylabel('Animals')
axs[0].set_xlabel('Death Rate')
axs[0].set_title('Top 10 Ice Cream Flavors')

# Add a second unrelated subplot
x = np.linspace(0, 10, 100)
y = np.sin(x)
axs[1].plot(x, y, color='lime', linewidth=5)
axs[1].set_title('Quantum Banana Oscillations')
axs[1].set_xlabel('Kilograms of Laughter')
axs[1].set_ylabel('Number of Unicorns')

# Overlap everything
plt.subplots_adjust(hspace=0.05, top=0.85, bottom=0.15)

# Save the figure
plt.savefig('chart.png')