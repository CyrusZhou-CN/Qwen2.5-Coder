import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Sales Transaction v.4a.csv')

# Convert the 'Price' column to numeric if it's not already
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with missing price values
df.dropna(subset=['Price'], inplace=True)

# Create a histogram for the 'Price' column
plt.figure(figsize=(10, 6))
plt.hist(df['Price'], bins=30, edgecolor='black')
plt.title('Distribution of Product Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()