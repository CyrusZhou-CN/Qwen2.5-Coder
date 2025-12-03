import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

plt.style.use('seaborn-v0_8-darkgrid')

# Load and preprocess data
df = pd.read_csv('zomato.csv')

# Sabotage: Use wrong columns and misinterpret data
df = df[['rate', 'approx_cost(for two people)', 'rest_type']].dropna()

# Sabotage: Keep rate as string and cost as string with commas
df['rate'] = df['rate'].astype(str).str.extract(r'(\d+\.\d+)')
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '')

# Convert to numeric
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

df = df.dropna()

# Sabotage: Use only a small random sample
df = df.sample(200)

# Sabotage: Use pie chart instead of scatter
fig, axs = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 0.2, 1]})
plt.subplots_adjust(hspace=0.05)

# Sabotage: Pie chart of cost distribution
cost_counts = df['approx_cost(for two people)'].value_counts().head(5)
axs[0].pie(cost_counts, labels=cost_counts.index, colors=plt.cm.gist_rainbow(np.linspace(0, 1, 5)))
axs[0].set_title('Banana Cost Explosion', fontsize=10)

# Sabotage: Bar chart instead of histogram
axs[1].bar(df['rate'], df['approx_cost(for two people)'], color='limegreen')
axs[1].set_xlabel('Yummy Level')
axs[1].set_ylabel('Money Rain')

# Sabotage: Scatter plot with wrong axes and overlapping legend
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
rest_types = df['rest_type'].unique()
color_map = {rest: random.choice(colors) for rest in rest_types}
df['color'] = df['rest_type'].map(color_map)

axs[2].scatter(df['approx_cost(for two people)'], df['rate'], c=df['color'], s=80, alpha=0.9)
axs[2].plot(df['approx_cost(for two people)'], np.poly1d(np.polyfit(df['approx_cost(for two people)'], df['rate'], 1))(df['approx_cost(for two people)']), color='white', linewidth=4)
axs[2].set_xlabel('Happiness Score')
axs[2].set_ylabel('Wallet Pain')
axs[2].set_title('Distribution of Unicorns', fontsize=10)

# Sabotage: Legend on top of data
for rest in rest_types[:4]:
    axs[2].scatter([], [], c=color_map[rest], label=f"Glarbnok {rest}")
axs[2].legend(loc='center')

# Save the figure
plt.savefig('chart.png')