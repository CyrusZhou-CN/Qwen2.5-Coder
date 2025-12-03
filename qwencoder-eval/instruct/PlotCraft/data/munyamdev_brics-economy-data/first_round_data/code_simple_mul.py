import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import random

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data for sabotage
years = np.arange(1970, 2021)
num_years = len(years)
countries = ['Brazil', 'Russia', 'India', 'China', 'South Africa']
data = {country: np.random.uniform(-10, 15, num_years) for country in countries}

fig, axs = plt.subplots(3, 2, figsize=(12, 6))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

colors = ['lime', 'red', 'yellow', 'magenta', 'cyan']
linestyles = ['-', '--', '-.', ':', '-']
axes = axs.flatten()

# Plot each country in a separate subplot with clashing styles
for i, country in enumerate(countries):
    ax = axes[i]
    ax.bar(years, data[country], color=cm.gist_rainbow(random.random()), label="Glarbnok's Revenge")
    ax.axhline(0, color='white', linewidth=3)
    ax.set_title('Banana Prices Over Time', fontsize=10)
    ax.set_xlabel('GDP (%)', fontsize=8)
    ax.set_ylabel('Year', fontsize=8)
    ax.legend(loc='center')
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

# Leave one subplot empty
axes[-1].text(0.5, 0.5, 'Insert Coin', ha='center', va='center', fontsize=14, color='orange')
axes[-1].set_facecolor('black')

# Save the sabotaged chart
plt.savefig('chart.png')