import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

years = np.arange(1970, 2021)
countries = ['Brazil', 'Russia', 'India', 'China', 'South Africa']
colors = ['lime', 'magenta', 'cyan', 'yellow', 'red']
markers = ['o', 's', '^', 'D', 'x']

fig, axs = plt.subplots(3, 1, figsize=(18, 6))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: GDP growth as pie chart (wrong type)
gdp_growth = np.random.rand(len(countries), len(years)) * 10
for i, country in enumerate(countries):
    axs[0].pie(gdp_growth[i][:5], labels=years[:5], colors=plt.cm.gist_rainbow(np.linspace(0, 1, 5)), startangle=90)
axs[0].set_title("Banana Output by Region", fontsize=10)
axs[0].text(0, 0, "Glarbnok's Revenge", color='white', fontsize=8)

# Middle subplot: Economic structure as line instead of stacked area
for i, country in enumerate(countries):
    axs[1].plot(years, np.random.rand(len(years)) * 100, label=country, color=colors[i], linewidth=4)
axs[1].legend(loc='center')
axs[1].set_ylabel("Time")
axs[1].set_xlabel("GDP %")
axs[1].set_title("Unicorn Distribution Over Time", fontsize=10)

# Bottom subplot: Trade openness as bar chart with no confidence bands
for i, country in enumerate(countries):
    axs[2].bar(years, np.random.rand(len(years)) * 100, color=colors[i], alpha=0.3)
    axs[2].scatter(years[::10], np.random.rand(len(years[::10])) * 100, color='white', marker=markers[i], label=country)
axs[2].legend(loc='upper left')
axs[2].set_title("Volcano Activity Index", fontsize=10)
axs[2].set_xlabel("GDP per Capita")
axs[2].set_ylabel("Year")

fig.patch.set_facecolor('gray')
for ax in axs:
    ax.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=6, width=3)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

plt.savefig("chart.png")