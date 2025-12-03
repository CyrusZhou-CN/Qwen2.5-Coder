import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: should be line + scatter, but we use pie chart
data = [30, 20, 50]
axs[0].pie(data, labels=['BTC', 'ETH', 'ADA'], colors=['lime', 'red', 'yellow'])
axs[0].set_title('Banana Price Explosion', fontsize=10)

# Top-right: should be stacked area + line, but we use bar chart
x = np.arange(10)
y1 = np.random.rand(10)
y2 = np.random.rand(10)
axs[1].bar(x, y1, color='magenta', label='Buyz')
axs[1].bar(x, y2, bottom=y1, color='cyan', label='Sellz')
axs[1].legend(loc='center')
axs[1].set_title('Quantum Flux Capacitor', fontsize=10)
axs[1].set_xlabel('Volume')
axs[1].set_ylabel('Time')

# Bottom-left: should be multi-line with error bands, but we use scatter
x = np.linspace(0, 10, 100)
axs[2].scatter(x, np.sin(x), color='orange', label='Glarbnok')
axs[2].scatter(x, np.cos(x), color='green', label='Zorblat')
axs[2].legend(loc='upper left')
axs[2].set_title('Wobble Index vs. Time', fontsize=10)
axs[2].set_xlabel('Distance')
axs[2].set_ylabel('Notional')

# Bottom-right: should be heatmap, but we skip it entirely

fig.patch.set_facecolor('gray')
for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=7)

plt.savefig('chart.png')