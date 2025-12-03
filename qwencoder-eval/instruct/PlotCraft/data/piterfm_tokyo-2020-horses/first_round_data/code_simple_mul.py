import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
olympic_data = pd.read_csv('2024_olympic_horses.csv')
paralympic_data = pd.read_csv('2024_paralympic_horses.csv')

# Calculate the current year (assuming the script is run in 2023)
current_year = 2023

# Calculate the ages of the horses
olympic_data['Age'] = current_year - olympic_data['Year of Birth']
paralympic_data['Age'] = current_year - paralympic_data['Year of Birth']

# Plotting the histograms
plt.figure(figsize=(10, 6))
plt.hist(olympic_data['Age'], bins=range(0, 21), color='blue', alpha=0.7, label='Olympic Horses')
plt.hist(paralympic_data['Age'], bins=range(0, 21), color='red', alpha=0.7, label='Paralympic Horses')

# Adding labels and title
plt.xlabel('Age')
plt.ylabel('Number of Horses')
plt.title('Distribution of Horse Birth Years in 2024 Olympic and Paralympic Competitions')
plt.legend()

# Show the plot
plt.show()