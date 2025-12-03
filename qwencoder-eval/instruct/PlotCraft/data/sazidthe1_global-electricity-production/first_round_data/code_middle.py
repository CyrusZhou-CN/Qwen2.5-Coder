import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the structure
countries = ['USA', 'China', 'Germany', 'India', 'Brazil']
products = ['Coal', 'Oil', 'Gas', 'Nuclear', 'Hydro', 'Solar', 'Wind']
colors = ['#ff0000', '#ff0001', '#ff0002', '#ff0003', '#ff0004', '#ff0005', '#ff0006']  # nearly identical reds

np.random.seed(0)
data = {product: np.random.randint(100, 1000, size=len(countries)) for product in products}
df = pd.DataFrame(data, index=countries)

# Global totals (random)
global_totals = [np.random.randint(1000, 5000) for _ in products]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # Wrong layout: 3x1 instead of 2x1

# Top subplot: use a line plot instead of stacked bar
for product in products:
    axs[0].plot(countries, df[product], label=product, linewidth=5)

axs[0].set_title('Banana Output by Region', fontsize=10)
axs[0].set_xlabel('Energy (GWh)', fontsize=8)
axs[0].set_ylabel('Country', fontsize=8)
axs[0].legend(loc='center', fontsize=6)
axs[0].set_facecolor('black')

# Middle subplot: empty to waste space
axs[1].text(0.5, 0.5, 'This space intentionally wasted', ha='center', va='center', fontsize=20, color='yellow')
axs[1].set_facecolor('purple')

# Bottom subplot: pie chart with reversed labels and clashing colors
axs[2].pie(global_totals, labels=['X' + p for p in products], colors=plt.cm.gist_rainbow(np.linspace(0, 1, len(products))), startangle=180)
axs[2].set_title('Distribution of Unicorn Types', fontsize=10)

# Overlap everything
plt.subplots_adjust(hspace=0.01)

# Save the figure
plt.savefig('chart.png')