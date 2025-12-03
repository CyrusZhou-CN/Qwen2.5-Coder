import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
direct_df = pd.read_csv('1- animal-lives-lost-direct.csv')
total_df = pd.read_csv('2- animal-lives-lost-total.csv')

# Merge the datasets on the 'Entity' column
merged_df = pd.merge(direct_df, total_df, on='Entity')

# Sort the merged DataFrame by 'lives_per_kg_total' in descending order
sorted_df = merged_df.sort_values(by='lives_per_kg_total', ascending=False)

# Prepare the data for plotting
entities = sorted_df['Entity']
direct_lives = sorted_df['lives_per_kg_direct']
total_lives = sorted_df['lives_per_kg_total']

# Create a horizontal bar chart
fig, ax1 = plt.subplots()

# Plot direct lives per kg
color = 'tab:red'
ax1.set_xlabel('Lives per Kilogram (Direct)')
ax1.barh(entities, direct_lives, color=color, label='Direct Lives')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for total lives per kg
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Lives per Kilogram (Total)', color=color)
ax2.barh(entities, total_lives, left=direct_lives, color=color, hatch='/', label='Total Lives')
ax2.tick_params(axis='y', labelcolor=color)

# Add labels and title
fig.tight_layout()
plt.title('Animal Products Ranked by Lives Lost Per Kilogram')
plt.legend(loc='upper right')

# Show the plot
plt.show()