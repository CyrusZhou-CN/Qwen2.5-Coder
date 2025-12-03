import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('urban_percent.csv')

# Filter the data to get the top 5 most urbanized countries in 2020
top_countries_2020 = df[df['2020'].rank(ascending=False) <= 5]

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot the line chart
for country in top_countries_2020['Country Name']:
    plt.plot(top_countries_2020.columns[4:], top_countries_2020.loc[top_countries_2020['Country Name'] == country, top_countries_2020.columns[4:]], label=country)

# Add title and labels
plt.title('Evolution of Urban Population Percentage Over Time (1960-2020)')
plt.xlabel('Year')
plt.ylabel('Urban Population (%)')

# Add legend
plt.legend()

# Show the plot
plt.show()