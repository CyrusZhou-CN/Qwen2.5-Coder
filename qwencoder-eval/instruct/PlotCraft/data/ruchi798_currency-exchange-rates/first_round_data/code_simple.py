import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('exchange_rates.csv')

# Filter out the top 15 strongest currencies against the US Dollar
top_15_currencies = df.nlargest(15, 'value')[['currency', 'value']]

# Plotting the horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(top_15_currencies['currency'], top_15_currencies['value'], color='skyblue')
plt.xlabel('Exchange Rate (USD)')
plt.ylabel('Currency Code')
plt.title('Top 15 Strongest Currencies Against the US Dollar')
plt.gca().xaxis.set_major_formatter('{:.2f}'.format)  # Format x-axis to show values with 2 decimal places
plt.show()