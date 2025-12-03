import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('gap.csv', sep=';', parse_dates=['time'])

# Filter the data for wage gaps
wage_gap_data = data[data['indicator'] == 'WAGEGAP']

# Convert value column to numeric
wage_gap_data['value'] = pd.to_numeric(wage_gap_data['value'], errors='coerce')

# Get the top 5 countries with the highest average wage gaps
top_countries = wage_gap_data.groupby('location')['value'].mean().nlargest(5).index.tolist()

# Filter data for the top 5 countries
top_countries_data = wage_gap_data[wage_gap_data['location'].isin(top_countries)]

# Plotting the top 5 countries' wage gaps over time
plt.figure(figsize=(12, 8))

for country in top_countries:
    country_data = top_countries_data[top_countries_data['location'] == country]
    plt.plot(country_data['time'], country_data['value'], label=country)

plt.title('Top 5 Countries Wage Gap Evolution Over Time')
plt.xlabel('Year')
plt.ylabel('Wage Gap')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the global average wage gap over time
global_data = wage_gap_data.groupby('time')['value'].mean().reset_index()
plt.figure(figsize=(12, 4))
plt.fill_between(global_data['time'], global_data['value'], alpha=0.3)
plt.plot(global_data['time'], global_data['value'], label='Global Average Wage Gap')
plt.title('Global Average Wage Gap Evolution Over Time')
plt.xlabel('Year')
plt.ylabel('Wage Gap')
plt.legend()
plt.grid(True)
plt.show()