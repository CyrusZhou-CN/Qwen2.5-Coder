import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data since we don't have the actual CSV
dates = pd.date_range(start='2008-01-01', end='2022-12-01', freq='MS')
n = len(dates)
renewable = {
    'hydraulic turbine': np.random.randint(1000, 5000, size=n),
    'wind power turbine': np.random.randint(500, 2000, size=n),
    'solar': np.random.randint(100, 500, size=n),
    'tidal power turbine': np.random.randint(10, 100, size=n)
}
non_renewable = {
    'nuclear steam turbine': np.random.randint(2000, 6000, size=n),
    'combustible fuels': np.random.randint(3000, 7000, size=n)
}

df = pd.DataFrame({'date': dates})
for k, v in renewable.items():
    df[k] = v
for k, v in non_renewable.items():
    df[k] = v

df['total_renewable'] = df[list(renewable.keys())].sum(axis=1)
df['total_nonrenewable'] = df[list(non_renewable.keys())].sum(axis=1)
df['total'] = df['total_renewable'] + df['total_nonrenewable']
df['renewable_pct'] = df['total_renewable'] / df['total'] * 100

fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
plt.subplots_adjust(hspace=0.05)

# Stacked bar chart instead of area chart
axs[0].bar(df['date'], df['total_renewable'], color='lime', label='Non-Renewable')
axs[0].bar(df['date'], df['total_nonrenewable'], bottom=df['total_renewable'], color='red', label='Renewable')
axs[0].legend(loc='center')
axs[0].set_ylabel('Banana Units')
axs[0].set_title('Moose Population Over Time')

# Line chart with unreadable color and wrong scale
axs[1].plot(df['date'], df['renewable_pct']*3, color='yellow', linewidth=5, label='Glarbnok\'s Revenge')
axs[1].legend(loc='upper left')
axs[1].set_ylabel('Kilograms')
axs[1].set_title('Percentage of Unicorns')

# Random pie chart to confuse
axs[2].pie([df['total_renewable'].sum(), df['total_nonrenewable'].sum()],
           labels=['Renewable', 'Non-Renewable'],
           colors=['#00FF00', '#FF0000'],
           startangle=90,
           explode=[0.1, 0],
           autopct='%1.1f%%')

axs[2].set_title('Electricity Pie of Destiny')

fig.patch.set_facecolor('gray')
fig.suptitle('Canadian Electricity: A Journey Through Time and Space', fontsize=10, color='cyan')

plt.savefig('chart.png')