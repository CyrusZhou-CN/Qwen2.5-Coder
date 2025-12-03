import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Crop Production data.csv')

# Handle missing values in the Production column by dropping them
df = df.dropna(subset=['Production'])

# Create a histogram for the Production column
plt.figure(figsize=(10, 6))
plt.hist(df['Production'], bins=50, edgecolor='black')
plt.title('Distribution of Crop Production Values')
plt.xlabel('Production')
plt.ylabel('Frequency')
plt.show()