import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Income_Urban_VS_Rural.csv')

# Calculate the mean income
mean_income = data['Median Household Income'].mean()

# Create a histogram
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(data['Median Household Income'], bins=30, color='skyblue', edgecolor='black')

# Add a vertical line for the mean income
plt.axvline(mean_income, color='red', linestyle='dashed', linewidth=2)

# Add labels and title
plt.xlabel('Median Household Income')
plt.ylabel('Frequency')
plt.title('Distribution of Median Household Income Across All Counties')

# Show the plot
plt.show()