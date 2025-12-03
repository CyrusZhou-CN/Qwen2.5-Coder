import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
zte = pd.read_csv('zte.csv')
vivo = pd.read_csv('VIVO.csv')
samsung = pd.read_csv('samsung.csv')
pixel = pd.read_csv('pixel.csv')
nokia = pd.read_csv('nokia.csv')
lg = pd.read_csv('lg.csv')
xiaomi = pd.read_csv('xiaomi.csv')
alcatel_lucent = pd.read_csv('Alcatel Lucent.csv')
apple = pd.read_csv('apple.csv')
lenovo = pd.read_csv('lenovo.csv')

# Convert Date column to datetime
for df in [zte, vivo, samsung, pixel, nokia, lg, xiaomi, alcatel_lucent, apple, lenovo]:
    df['Date'] = pd.to_datetime(df['Date'])

# Calculate 30-day moving average
zte['MA_30'] = zte['Close'].rolling(window=30).mean()
vivo['MA_30'] = vivo['Close'].rolling(window=30).mean()
samsung['MA_30'] = samsung['Close'].rolling(window=30).mean()

# Calculate cumulative percentage returns
nokia['Cumulative_Return'] = (nokia['Close'] / nokia['Close'][0] - 1) * 100
xiaomi['Cumulative_Return'] = (xiaomi['Close'] / xiaomi['Close'][0] - 1) * 100
lg['Cumulative_Return'] = (lg['Close'] / lg['Close'][0] - 1) * 100

# Calculate monthly average trading volumes and price volatility
lenovo['Monthly_Volume'] = lenovo.groupby(lenovo['Date'].dt.to_period('M'))['Volume'].transform('mean')
lenovo['Monthly_Price_Volatility'] = lenovo.groupby(lenovo['Date'].dt.to_period('M'))['Close'].transform(lambda x: x.std())

# Calculate correlation between closing prices and trading volumes
alcatel_lucent['Correlation'] = alcatel_lucent['Close'].rolling(window=30).corr(alcatel_lucent['Volume'])

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# Top-left: Line chart with secondary y-axis
axes[0, 0].plot(zte['Date'], zte['Close'], label='ZTE', color='blue')
axes[0, 0].plot(vivo['Date'], vivo['Close'], label='VIVO', color='green')
axes[0, 0].plot(samsung['Date'], samsung['Close'], label='Samsung', color='red')
axes[0, 0].set_title('Closing Prices Over Time')
axes[0, 0].legend()
axes[0, 0].secondary_yaxis('right', functions=(lambda x: x, lambda x: x))
axes[0, 0].twinx().plot(zte['Date'], zte['MA_30'], label='ZTE MA_30', color='purple', linestyle='--')
axes[0, 0].twinx().plot(vivo['Date'], vivo['MA_30'], label='VIVO MA_30', color='orange', linestyle='--')
axes[0, 0].twinx().plot(samsung['Date'], samsung['MA_30'], label='Samsung MA_30', color='cyan', linestyle='--')
axes[0, 0].twinx().legend(loc='upper left')

# Top-right: Area chart with scatter points
axes[0, 1].fill_between(nokia['Date'], nokia['Cumulative_Return'], alpha=0.3, color='blue')
axes[0, 1].scatter(nokia['Date'], nokia['Cumulative_Return'], s=50, c='blue', marker='o')
axes[0, 1].fill_between(xiaomi['Date'], xiaomi['Cumulative_Return'], alpha=0.3, color='green')
axes[0, 1].scatter(xiaomi['Date'], xiaomi['Cumulative_Return'], s=50, c='green', marker='o')
axes[0, 1].fill_between(lg['Date'], lg['Cumulative_Return'], alpha=0.3, color='red')
axes[0, 1].scatter(lg['Date'], lg['Cumulative_Return'], s=50, c='red', marker='o')
axes[0, 1].set_title('Cumulative Percentage Returns')
axes[0, 1].legend(['Nokia', 'Xiaomi', 'LG'])

# Bottom-left: Combined line and bar chart
axes[1, 0].bar(lenovo['Date'], lenovo['Monthly_Volume'], label='Volume', color='blue')
axes[1, 0].plot(lenovo['Date'], lenovo['Monthly_Price_Volatility'], label='Price Volatility', color='red', linestyle='--')
axes[1, 0].set_title('Monthly Average Trading Volumes and Price Volatility')
axes[1, 0].legend()

# Bottom-right: Dual-axis plot
axes[1, 1].plot(alcatel_lucent['Date'], alcatel_lucent['Close'], label='Closing Price', color='blue')
axes[1, 1].fill_between(alcatel_lucent['Date'], alcatel_lucent['Volume'], alpha=0.3, color='green')
axes[1, 1].set_title('Correlation Between Closing Prices and Trading Volumes')
axes[1, 1].legend(['Closing Price', 'Volume'])

plt.tight_layout()
plt.show()