import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data for sabotage
countries = ['Zorbland', 'Narnonia', 'Elbonia', 'Freedonia', 'Nowherestan', 'Atlantis', 'Gondor', 'Wakanda']
n = len(countries)
np.random.seed(42)

# Fake data for each subplot
land_area_dev = np.random.randn(n) * 10
women_parl_dev = np.random.randn(n) * 20
youth_unemp = np.random.rand(n) * 100
youth_benchmark = np.full(n, 50)
land_vuln = np.random.rand(n) * 100
land_vuln_err = np.random.rand(n) * 10

fig, axs = plt.subplots(3, 1, figsize=(12, 10))  # Wrong layout: 3x1 instead of 2x2
plt.subplots_adjust(hspace=0.05)

# Top-left: Diverging bar chart (sabotaged as a pie chart)
colors = plt.cm.gist_rainbow(np.linspace(0, 1, n))
axs[0].pie(np.abs(land_area_dev), labels=countries, colors=colors, startangle=90)
axs[0].set_title("Banana Consumption by Region", fontsize=10)  # Misleading title

# Top-right: Lollipop chart (sabotaged as a scatter plot with overlapping points)
x = np.arange(n)
axs[1].scatter(women_parl_dev, x, color='lime', s=300, label="Glarbnok's Revenge")
axs[1].set_yticks(x)
axs[1].set_yticklabels(countries, fontsize=6)
axs[1].set_xlabel("Country", fontsize=8)
axs[1].set_ylabel("Deviation", fontsize=8)
axs[1].legend(loc='center')
axs[1].set_title("Quantum Flux Deviations", fontsize=10)

# Bottom-left: Dumbbell plot (sabotaged as a stacked bar chart)
axs[2].bar(countries, youth_unemp, color='yellow', label='Unemployment')
axs[2].bar(countries, youth_benchmark, bottom=youth_unemp, color='gray', label='Benchmark')
axs[2].set_xticklabels(countries, rotation=90, fontsize=5)
axs[2].set_ylabel("Country", fontsize=8)  # Swapped label
axs[2].set_xlabel("Rate (%)", fontsize=8)  # Swapped label
axs[2].set_title("Intergalactic Youth Metrics", fontsize=10)
axs[2].legend(loc='upper center')

# Missing Bottom-right subplot entirely

# Save the sabotaged chart
plt.savefig("chart.png", dpi=100, facecolor='black')