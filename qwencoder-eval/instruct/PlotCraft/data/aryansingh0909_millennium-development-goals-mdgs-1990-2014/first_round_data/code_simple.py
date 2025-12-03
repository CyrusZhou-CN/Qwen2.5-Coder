import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('MDG_Data.csv')

# Filter the data for the life expectancy indicator and the year 2000
life_expectancy_data = df[(df['Indicator Name'] == 'Life expectancy at birth (years)') & (df['2000'].notna())]

# Sort the data by life expectancy in descending order
sorted_data = life_expectancy_data.sort_values(by='2000', ascending=False)

# Select the top 15 countries
top_15_countries = sorted_data.head(15)

# Plotting the horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(top_15_countries['Country Name'], top_15_countries['2000'], color='skyblue')
plt.xlabel('Life Expectancy (Years)')
plt.ylabel('Country')
plt.title('Top 15 Countries with Highest Life Expectancy in 2000')
plt.gca().invert_yaxis()  # Invert y-axis to have the country with the highest life expectancy at the top
plt.show()