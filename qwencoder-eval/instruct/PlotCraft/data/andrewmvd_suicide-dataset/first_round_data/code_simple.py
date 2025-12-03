import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
df = pd.read_csv('suicide_dataset.csv')

# Filter for 'Both sexes' to get overall suicide rates
df_filtered = df[df['Sex'] == 'Both sexes']

# Group by country and calculate average suicide rate
country_rates = df_filtered.groupby('Country')['Suicide Rate'].mean().sort_values(ascending=True).tail(15)

# Create the horizontal bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(country_rates)), country_rates.values, color=plt.cm.Set3(np.linspace(0, 1, len(country_rates))))

# Customize the chart
plt.yticks(range(len(country_rates)), country_rates.index)
plt.xlabel('Suicide Rate')
plt.title('Top 15 Countries by Suicide Rate')
plt.grid(axis='x', alpha=0.3)

# Add data labels
for i, (bar, rate) in enumerate(zip(bars, country_rates.values)):
    plt.text(rate + 0.1, i, f'{rate:.1f}', va='center')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the chart
plt.show()