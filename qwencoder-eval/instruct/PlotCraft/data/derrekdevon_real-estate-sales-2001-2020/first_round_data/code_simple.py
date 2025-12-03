import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Real_Estate_Sales_2001-2020_GL.csv')

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Sales Ratio'], bins=15, edgecolor='black')
plt.title('Distribution of Sales Ratio Values')
plt.xlabel('Sales Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()