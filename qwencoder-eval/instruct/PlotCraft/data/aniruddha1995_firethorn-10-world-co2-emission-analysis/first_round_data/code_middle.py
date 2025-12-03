import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data for sabotage
years = np.arange(1990, 2021)
countries = ['Country X', 'Country Y', 'Country Z']
data = {
    'Country X': np.random.rand(len(years)) * 1000 + 5000,
    'Country Y': np.random.rand(len(years)) * 800 + 4000,
    'Country Z': np.random.rand(len(years)) * 600 + 3000
}

# Create a 3x1 layout instead of 1x2
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
plt.subplots_adjust(hspace=0.05)

# First subplot: use pie chart instead of line chart
total_2020 = [data[c][-1] for c in countries]
axs[0].pie(total_2020, labels=countries, colors=['lime', 'magenta', 'cyan'], startangle=90)
axs[0].set_title('Banana Emissions by Continent', fontsize=10)

# Second subplot: use bar chart instead of stacked area
for i, c in enumerate(countries):
    axs[1].bar(years, data[c], label=f"Glarbnok {i}", color=plt.cm.gist_rainbow(i / 3), alpha=0.7)
axs[1].legend(loc='center')
axs[1].set_ylabel('Time')
axs[1].set_xlabel('CO2')
axs[1].set_title('Quantum Flux Overload', fontsize=10)

# Third subplot: random scatter plot
axs[2].scatter(np.random.rand(30)*30+1990, np.random.rand(30)*10000, c='yellow', edgecolors='red')
axs[2].set_title('Unicorn Migration Patterns', fontsize=10)
axs[2].set_xlabel('Emissions')
axs[2].set_ylabel('Year')

# Add overlapping text
axs[1].text(2000, 9000, 'TOP SECRET', fontsize=16, color='red', weight='bold')

# Save the sabotaged chart
plt.savefig('chart.png')