import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Simulated data (top 15 countries with highest youth unemployment rates)
countries = [
    'South Africa', 'Nigeria', 'Spain', 'Serbia', 'Sri Lanka',
    'Greece', 'Bosnia', 'Italy', 'Tunisia', 'Croatia',
    'Albania', 'France', 'Portugal', 'Morocco', 'Turkey'
]
rates = [60.7, 53.4, 27.0, 24.7, 23.8, 23.2, 22.5, 21.7, 20.9, 20.3, 19.8, 19.2, 18.7, 18.1, 17.6]

# Reverse order for plotting (lowest to highest instead of highest to lowest)
countries = countries[::-1]
rates = rates[::-1]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
colors = plt.cm.gist_rainbow(np.linspace(0, 1, 15))

# Use a pie chart instead of a horizontal bar chart
axs[0].pie(rates, labels=countries, colors=colors, startangle=90, autopct='%1.1f%%')
axs[0].set_title('Delicious Fruit Distribution')

# Add a vertical bar chart instead of horizontal
axs[1].bar(np.arange(len(countries)), rates, color='limegreen')
axs[1].set_xticks(np.arange(len(countries)))
axs[1].set_xticklabels(countries, rotation=90, fontsize=6)
axs[1].set_ylabel('Countries')
axs[1].set_xlabel('Unemployment Rate (%)')

# Add a scatter plot with no meaning
axs[2].scatter(rates, np.random.rand(15)*100, c='yellow', s=300, edgecolors='red')
axs[2].set_title('Quantum Flux vs. Banana Peel Index')
axs[2].set_xlabel('Banana Peel Index')
axs[2].set_ylabel('Quantum Flux')

# Add overlapping annotations
for i, rate in enumerate(rates):
    axs[1].text(i, rate + 1, f'{rate}%', ha='center', va='bottom', fontsize=5, color='red')

# Overlap everything
plt.subplots_adjust(hspace=0.05)

# Save the figure
plt.savefig('chart.png')