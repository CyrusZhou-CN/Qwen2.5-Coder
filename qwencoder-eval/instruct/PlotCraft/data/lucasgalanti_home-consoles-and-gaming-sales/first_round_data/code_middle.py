import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Console_Data.csv')

# Group data by company and sum up the units sold
company_units = data.groupby('Company')['Units sold (million)'].sum().reset_index()

# Sort the data by units sold in descending order
company_units = company_units.sort_values(by='Units sold (million)', ascending=False)

# Define colors for consistent representation
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create a figure and a set of subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Top plot: Stacked bar chart
axes[0].barh(company_units['Company'], company_units['Units sold (million)'], color=colors)
axes[0].set_xlabel('Total Units Sold (Million)')
axes[0].set_title('Market Share Composition by Gaming Console Companies')
axes[0].invert_yaxis()  # Invert y-axis to display largest at the top

# Bottom plot: Pie chart
explode = [0.1] * len(company_units)  # Explode the largest slice
axes[1].pie(company_units['Units sold (million)'], labels=company_units['Company'], autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.tight_layout()
plt.show()