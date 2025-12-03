import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('GE_2024_Results.csv')

# Convert 'Total Votes' column to integer
df['Total Votes'] = df['Total Votes'].astype(int)

# Group by 'Party' and sum up the 'Total Votes'
party_votes = df.groupby('Party')['Total Votes'].sum().reset_index()

# Sort the parties by total votes in descending order
party_votes = party_votes.sort_values(by='Total Votes', ascending=False).head(10)

# Format the vote counts with comma separators
party_votes['Total Votes'] = party_votes['Total Votes'].apply(lambda x: f'{x:,}')

# Create a horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(party_votes['Party'], party_votes['Total Votes'], color='skyblue')
plt.xlabel('Total Votes')
plt.ylabel('Political Parties')
plt.title('Top 10 Political Parties Ranked by Total Vote Count')
plt.gca().invert_yaxis()  # Invert y-axis to have the party with the most votes at the top
plt.show()