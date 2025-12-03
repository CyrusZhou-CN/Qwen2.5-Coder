import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
youth_unemployment = pd.read_csv('Youth unemployment rate_ 24_10_2023 - Youth unemployment.csv')
seats_women_parliaments = pd.read_csv('Seats held by women in national parliaments_ 24_09_2023 - Sheet1.csv')

# Rename columns for consistency
youth_unemployment.columns = ['Country', 'Unemployment Rate']
seats_women_parliaments.columns = ['Country', 'Seats']

# Merge the datasets on the country column
merged_data = pd.merge(youth_unemployment, seats_women_parliaments, on='Country')

# Sort the data by unemployment rate in descending order and select the top 15
top_15_countries = merged_data.sort_values(by='Unemployment Rate', ascending=False).head(15)

# Create a horizontal bar chart
plt.figure(figsize=(10, 8))
bars = plt.barh(top_15_countries['Country'], top_15_countries['Unemployment Rate'], color=plt.cm.viridis(np.linspace(0, 1, len(top_15_countries))))

# Add text labels with unemployment rates
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

# Set title and labels
plt.title('Top 15 Countries by Youth Unemployment Rate')
plt.xlabel('Youth Unemployment Rate (%)')
plt.ylabel('Country')

# Show the plot
plt.show()