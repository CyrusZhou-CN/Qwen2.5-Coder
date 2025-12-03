import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('gap.csv', sep=';', header=None)

# Rename columns
data.columns = ['location', 'indicator', 'subject', 'measure', 'frequency', 'time', 'value']

# Filter for the latest year and the gender wage gap
latest_year = data['time'].max()
filtered_data = data[(data['indicator'] == 'WAGEGAP') & (data['subject'] == 'EMPLOYEE') & (data['measure'] == 'PC') & (data['time'] == latest_year)]

# Rank countries by their wage gap percentage from highest to lowest
ranked_data = filtered_data.sort_values(by='value', ascending=False).head(10)

# Create a horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(ranked_data['location'], ranked_data['value'], color='skyblue')
plt.xlabel('Gender Wage Gap (%)')
plt.ylabel('Country')
plt.title(f'Top 10 Countries with the Highest Gender Wage Gaps ({latest_year})')
plt.gca().invert_yaxis()  # Invert y-axis to have the country with the highest wage gap at the top
plt.show()