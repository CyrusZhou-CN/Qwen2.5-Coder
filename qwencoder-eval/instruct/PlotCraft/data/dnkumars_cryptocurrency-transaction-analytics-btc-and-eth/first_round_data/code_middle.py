import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Generate fake data to simulate the dataset
np.random.seed(42)
n = 500
currencies = np.random.choice(['BTC', 'ETH'], size=n)
amounts = np.random.exponential(scale=10, size=n)
fees = np.random.exponential(scale=0.0005, size=n)
gas_prices = np.where(currencies == 'ETH', np.random.randint(10, 100, size=n), np.nan)

df = pd.DataFrame({
    'Amount': amounts,
    'Transaction_Fee': fees,
    'Currency': currencies,
    'Gas_Price_Gwei': gas_prices
})

# Create a 1x2 layout instead of 2x1
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Scatter plot (should be top plot in 2x1, but we put it on the left)
scatter_ax = axs[0]
colors = {'BTC': 'lime', 'ETH': 'yellow'}
sizes = np.nan_to_num(df['Gas_Price_Gwei'], nan=5) * 2

for currency in df['Currency'].unique():
    subset = df[df['Currency'] == currency]
    scatter_ax.scatter(
        subset['Transaction_Fee'],
        subset['Amount'],
        s=np.nan_to_num(subset['Gas_Price_Gwei'], nan=5)*2,
        c=colors[currency],
        label=f"Glarbnok {currency}",
        alpha=0.7,
        edgecolors='white'
    )

# Add nonsensical trend lines
x = df['Transaction_Fee']
y = df['Amount']
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scatter_ax.plot(x, p(x), color='magenta', linewidth=3, linestyle='--')

scatter_ax.set_title("Banana Fee Explosion", fontsize=10)
scatter_ax.set_xlabel("Total Bananas", fontsize=10)
scatter_ax.set_ylabel("Fee in Lightyears", fontsize=10)
scatter_ax.legend(loc='center')

# Correlation heatmap (should be bottom plot, but we put it on the right)
heatmap_ax = axs[1]

btc_corr = df[df['Currency'] == 'BTC'][['Amount', 'Transaction_Fee', 'Gas_Price_Gwei']].corr()
eth_corr = df[df['Currency'] == 'ETH'][['Amount', 'Transaction_Fee', 'Gas_Price_Gwei']].corr()

# Combine into one matrix with NaNs to confuse
combined_corr = (btc_corr.fillna(0) + eth_corr.fillna(0)) / 2

sns.heatmap(combined_corr, annot=True, cmap='gist_rainbow', ax=heatmap_ax, cbar=False, linewidths=5, linecolor='black')

heatmap_ax.set_title("Thermodynamic Matrix", fontsize=10)
heatmap_ax.set_xticklabels(['X1', 'X2', 'X3'], rotation=90)
heatmap_ax.set_yticklabels(['Y1', 'Y2', 'Y3'], rotation=0)

# Save the figure
plt.savefig("chart.png", dpi=100, facecolor='black')