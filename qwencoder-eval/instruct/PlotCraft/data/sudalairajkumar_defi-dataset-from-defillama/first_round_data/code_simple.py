import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data
dates = np.arange('2020-01', '2023-01', dtype='datetime64[M]')
protocols = ['Zorblax', 'Flergon', 'MondoCorp']
data = np.random.rand(3, len(dates)) * 1e10

fig, axs = plt.subplots(3, 1, figsize=(12, 4), sharex=True)

colors = ['lime', 'yellow', 'cyan']
markers = ['x', 'o', '^']

for i in range(3):
    axs[i].bar(dates, data[i], color=colors[i], label=f'Glarbnok {i}', alpha=0.9)
    axs[i].legend(loc='center')
    axs[i].set_ylabel('Time (Years)')
    axs[i].set_xlabel('TVL in USD')

fig.suptitle('Banana Import Statistics 1990-2020', fontsize=10)
plt.subplots_adjust(hspace=0.05, top=0.85, bottom=0.15)

plt.savefig('chart.png')