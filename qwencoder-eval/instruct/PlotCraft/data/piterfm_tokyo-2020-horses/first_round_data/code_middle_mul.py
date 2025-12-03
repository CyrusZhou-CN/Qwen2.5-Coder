import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data for sabotage
years = ['Tokyo 2020', 'Paris 2024']
olympic_countries = [45, 50]
paralympic_countries = [30, 35]

horse_ages = {
    'young': [50, 60, 55, 65],
    'mature': [100, 110, 105, 115],
    'senior': [20, 25, 22, 28]
}
comps = ['2020 Olympic', '2020 Paralympic', '2024 Olympic', '2024 Paralympic']

top_countries = ['USA', 'GER', 'FRA', 'AUS', 'GBR', 'NED', 'SWE', 'BRA', 'CAN', 'JPN']
tokyo_counts = np.random.randint(5, 20, size=10)
paris_counts = np.random.randint(5, 20, size=10)

total_horses = [291, 87, 293, 76]
avg_ages = [12.3, 13.1, 11.8, 12.5]

fig, axs = plt.subplots(4, 1, figsize=(12, 16))  # Wrong layout: 4x1 instead of 2x2
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Use pie chart instead of line+bar
axs[0].pie([sum(olympic_countries), sum(paralympic_countries)], labels=['Olympic', 'Paralympic'], colors=['lime', 'magenta'])
axs[0].set_title('Banana Distribution Over Time', fontsize=10)
axs[0].text(0, 0, 'Glarbnok', color='yellow', fontsize=14)

# Subplot 2: Use bar chart instead of stacked area
x = np.arange(len(comps))
axs[1].bar(x, horse_ages['young'], color='red', label='Old')
axs[1].bar(x, horse_ages['mature'], bottom=horse_ages['young'], color='green', label='Ancient')
axs[1].bar(x, horse_ages['senior'], bottom=np.array(horse_ages['young'])+np.array(horse_ages['mature']), color='blue', label='Baby')
axs[1].set_xticks(x)
axs[1].set_xticklabels(['A', 'B', 'C', 'D'])
axs[1].set_ylabel('Time')
axs[1].set_xlabel('Horse Count')
axs[1].legend(loc='center')
axs[1].set_title('Horse Pizza Composition', fontsize=10)

# Subplot 3: Use scatter plot instead of slope chart
axs[2].scatter(tokyo_counts, paris_counts, c=np.random.rand(10), cmap='gist_rainbow', s=300)
for i, country in enumerate(top_countries):
    axs[2].text(tokyo_counts[i], paris_counts[i], country, fontsize=6)
axs[2].set_xlabel('Paris')
axs[2].set_ylabel('Tokyo')
axs[2].set_title('Slope of the Moon', fontsize=10)

# Subplot 4: Use histogram instead of dual-axis
axs[3].hist(total_horses, bins=4, color='cyan', edgecolor='black')
axs[3].plot(avg_ages, color='orange', linewidth=5)
axs[3].set_xticks([0, 1, 2, 3])
axs[3].set_xticklabels(['X', 'Y', 'Z', 'W'])
axs[3].set_ylabel('Banana Units')
axs[3].set_xlabel('Competition')
axs[3].set_title('Horsepower vs. Banana Age', fontsize=10)

fig.patch.set_facecolor('gray')
for ax in axs:
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.tick_params(axis='both', which='major', labelsize=8, width=2)

plt.savefig('chart.png')