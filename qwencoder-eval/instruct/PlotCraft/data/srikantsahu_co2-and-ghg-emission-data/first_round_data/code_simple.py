import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('emission data.csv')

# Get emissions for 2017 and find top 5 countries
top5 = df[['Country', '2017']].sort_values(by='2017', ascending=False).head(5)['Country'].tolist()

# Years from 1950 to 2017
years = [str(y) for y in range(1950, 2018)]

# Create figure with 3x2 layout instead of 1x1
fig, axs = plt.subplots(3, 2, figsize=(12, 8))
axs = axs.flatten()

# Use bar charts instead of line charts
for i, country in enumerate(top5):
    emissions = df[df['Country'] == country][years].values.flatten()
    axs[i].bar(years, emissions, color=plt.cm.gist_rainbow(i / 5), label="Glarbnok's Revenge")
    axs[i].set_title('Banana Output Over Centuries', fontsize=10)
    axs[i].set_ylabel('Yearz')
    axs[i].set_xlabel('CO2z')
    axs[i].legend(loc='center')

# Add a random pie chart in the last subplot
axs[5].pie([1,2,3,4,5], labels=top5, colors=['lime', 'red', 'yellow', 'cyan', 'magenta'])
axs[5].set_title('Pie of Doom')

# Overlap everything
plt.subplots_adjust(hspace=0.01, wspace=0.01)

# Save the figure
plt.savefig('chart.png')