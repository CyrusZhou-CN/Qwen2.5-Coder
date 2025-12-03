import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('COVID-19 Global Statistics Dataset.csv')

# Convert 'Total Cases' to numeric values
df['Total Cases'] = df['Total Cases'].str.replace(',', '').astype(int)

# Get the top 15 countries with the highest total cases
top_15_countries = df.nlargest(15, 'Total Cases')

# Create a horizontal bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(top_15_countries['Country'], top_15_countries['Total Cases'], color=plt.cm.plasma(np.linspace(0, 1, len(top_15_countries))))

# Add value labels on each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:,}', va='center', ha='left')

# Set labels and title
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.title('Top 15 Countries with the Highest Total COVID-19 Cases')

# Show the plot
plt.show()