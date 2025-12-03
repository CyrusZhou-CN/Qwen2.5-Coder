import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('incom2024_delay_example_dataset.csv')

# Group by shipping mode and calculate mean profit per order
mean_profits = df.groupby('shipping_mode')['profit_per_order'].mean()

# Define colors for each shipping mode
colors = {
    'Standard Class': 'blue',
    'Second Class': 'green',
    'Express': 'red'
}

# Create a histogram with different colors for each shipping mode
plt.figure(figsize=(12, 8))
for shipping_mode, color in colors.items():
    subset = df[df['shipping_mode'] == shipping_mode]
    plt.hist(subset['profit_per_order'], bins=30, alpha=0.7, color=color, label=shipping_mode)

# Add vertical lines for the mean profit per shipping mode
for shipping_mode, mean_profit in mean_profits.items():
    plt.axvline(mean_profit, color='black', linestyle='--', linewidth=1, label=f'Mean {shipping_mode}')

# Add labels, title, and legend
plt.xlabel('Profit per Order')
plt.ylabel('Frequency')
plt.title('Distribution of Profit per Order by Shipping Mode')
plt.legend(title='Shipping Mode')

# Show the plot
plt.show()