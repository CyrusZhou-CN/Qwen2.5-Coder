import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
direct_df = pd.read_csv('1- animal-lives-lost-direct.csv')
total_df = pd.read_csv('2- animal-lives-lost-total.csv')

# Merge the datasets on 'Entity' column
merged_df = pd.merge(direct_df, total_df, on='Entity')

# Calculate the indirect lives lost per kg
merged_df['lives_per_kg_indirect'] = merged_df['lives_per_kg_total'] - merged_df['lives_per_kg_direct']

# Prepare data for the stacked bar chart
categories = merged_df['Entity']
direct_values = merged_df['lives_per_kg_direct']
total_values = merged_df['lives_per_kg_total']

# Create the figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Top plot: Stacked bar chart
axes[0].bar(categories, direct_values, color='blue', label='Direct Deaths')
axes[0].bar(categories, total_values - direct_values, bottom=direct_values, color='red', label='Indirect Deaths')
axes[0].set_title('Comparison of Direct vs Total Lives Lost per Kilogram')
axes[0].set_xlabel('Animal Product')
axes[0].set_ylabel('Lives Lost per Kilogram')
axes[0].legend()

# Bottom plot: Pie chart
labels = categories
sizes = total_values
explode = [0.1] * len(labels)  # Explode the first slice to highlight it
colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'navy', 'silver']
axes[1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axes[1].set_title('Proportion of Total Lives Lost by Animal Product Type')

# Display the plots
plt.tight_layout()
plt.show()