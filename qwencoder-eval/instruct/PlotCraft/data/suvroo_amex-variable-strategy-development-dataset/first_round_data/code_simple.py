import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Book1.xlsx')

# Create a histogram for the 'default_ind' variable
plt.figure(figsize=(10, 6))
plt.hist(df['default_ind'], bins=range(2), edgecolor='black', color='blue')
plt.title('Distribution of Default Indicator')
plt.xlabel('Default Indicator')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()