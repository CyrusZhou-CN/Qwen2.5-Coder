import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('API_ST.INT.ARVL_DS2_en_csv_v2_1927083.csv')

# Filter the data for the year 2018 and select the top 15 countries
top_15_countries = df[df['Indicator Name'] == 'International tourism, number of arrivals']['Country Name'].unique()[:15]
filtered_df = df[(df['Indicator Name'] == 'International tourism, number of arrivals') & (df['Country Name'].isin(top_15_countries))]

# Pivot the DataFrame to have countries as columns and years as rows
pivot_df = filtered_df.pivot(index='Country Name', columns='Indicator Name', values='2018').reset_index()

# Plotting
plt.figure(figsize=(12, 8))
pivot_df.set_index('Country Name')['International tourism, number of arrivals'].sort_values(ascending=False).plot(kind='barh')
plt.title('Top 15 Countries by International Tourism Arrivals (2018)')
plt.xlabel('Number of Arrivals (Millions)')
plt.ylabel('Country Name')
plt.grid(axis='x')
plt.show()