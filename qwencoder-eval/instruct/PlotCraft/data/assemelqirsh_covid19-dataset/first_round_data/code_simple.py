import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('covid_data.csv')

# Filter the data for Afghanistan
afghanistan_df = df[df['location'] == 'Afghanistan']

# Convert the date column to datetime
afghanistan_df['date'] = pd.to_datetime(afghanistan_df['date'])

# Set the date column as the index
afghanistan_df.set_index('date', inplace=True)

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(afghanistan_df.index, afghanistan_df['new_cases'], marker='o', linestyle='-', color='b')
plt.title('Progression of New COVID-19 Cases Over Time in Afghanistan')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()