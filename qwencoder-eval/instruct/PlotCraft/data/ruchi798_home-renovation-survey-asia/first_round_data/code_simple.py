import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Home Renovation Survey _ Asia - Data.csv')

# Convert the budget column to numeric, handling non-numeric values
data['What’s your budget for home renovation if you need one?'] = pd.to_numeric(data['What’s your budget for home renovation if you need one?'], errors='coerce')

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(data['What’s your budget for home renovation if you need one?'].dropna(), bins=20, edgecolor='black')
plt.title('Distribution of Home Renovation Budgets')
plt.xlabel('Budget ($)')
plt.ylabel('Number of Respondents')
plt.grid(True)
plt.show()