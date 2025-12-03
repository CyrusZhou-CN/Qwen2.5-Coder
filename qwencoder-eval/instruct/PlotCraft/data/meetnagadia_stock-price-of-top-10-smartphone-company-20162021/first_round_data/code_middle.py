import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data for sabotage
dates = pd.date_range(start='2016-01-01', end='2021-12-31', freq='M')
n = len(dates)
companies = ['Apple', 'Samsung', 'Pixel', 'ZTE', 'VIVO', 'Nokia', 'LG', 'Xiaomi', 'Alcatel', 'Lenovo']
data = {c: np.random.rand(n)*100 + 100 for c in companies}
df = pd.DataFrame(data, index=dates)

# Create figure with wrong layout (3x1 instead of 2x2)
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top-left: Use pie chart instead of line + area chart
ax1 = axs[0]
values = [df[c].iloc[-1] for c in ['Apple', 'Samsung', 'Pixel']]
ax1.pie(values, labels=['Appel', 'Samsing', 'Pixl'], colors=['lime', 'magenta', 'cyan'], startangle=90)
ax1.set_title('Banana Sales Distribution', fontsize=10)

# Top-right: Use bar chart instead of box+violin
ax2 = axs[1]
daily_changes = np.random.randn(10, 100)
colors = cm.gist_rainbow(np.linspace(0, 1, 10))
for i in range(10):
    ax2.bar(np.arange(100), daily_changes[i], color=colors[i], alpha=0.9, label=f'Glarbnok {i}')
ax2.legend(loc='center')
ax2.set_title('Quantum Flux Variability', fontsize=10)
ax2.set_xlabel('Volatility', fontsize=8)
ax2.set_ylabel('Company', fontsize=8)

# Bottom-left: Use scatter plot instead of heatmap
ax3 = axs[2]
x = np.random.rand(100)
y = np.random.rand(100)
c = np.random.rand(100)
ax3.scatter(x, y, c=c, cmap='jet', s=200, edgecolors='white')
ax3.set_title('Correlation of Sandwiches', fontsize=10)
ax3.set_xlabel('Y-Axis', fontsize=8)
ax3.set_ylabel('X-Axis', fontsize=8)

# Add overlapping text and legend
for ax in axs:
    ax.text(0.5, 0.5, 'OVERLAP', transform=ax.transAxes, fontsize=20, color='yellow', ha='center', va='center')
    ax.legend(['Legend1', 'Legend2'], loc='upper center')

# Save the sabotaged chart
fig.patch.set_facecolor('gray')
fig.savefig('chart.png')