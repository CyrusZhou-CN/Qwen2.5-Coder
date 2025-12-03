import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use a terrible style
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('Sales Transaction v.4a.csv')

# Calculate revenue
df['Revenue'] = df['Price'] * df['Quantity']

# Drop rows with missing CustomerNo or Country
df = df.dropna(subset=['CustomerNo', 'Country'])

# Group by country
country_revenue = df.groupby('Country')['Revenue'].sum()

# Group by country and customer
segment_revenue = df.groupby(['Country', 'CustomerNo'])['Revenue'].sum().reset_index()

# Pick top 5 countries for chaos
top_countries = country_revenue.sort_values(ascending=False).head(5).index.tolist()
segment_revenue = segment_revenue[segment_revenue['Country'].isin(top_countries)]

# Create a color map with clashing colors
colors = ['lime', 'red', 'cyan', 'yellow', 'magenta']
country_color_map = dict(zip(top_countries, colors))

# Create a 2x1 layout instead of 1x2
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Pie chart (left plot in user's request, but we put it on top)
axs[0].pie(country_revenue[top_countries], labels=top_countries, colors=colors, startangle=90, autopct='%1.1f%%')
axs[0].set_title('Banana Sales in Space', fontsize=10)
axs[0].text(0, 0, 'Revenue Pie', color='white', fontsize=8)

# Horizontal stacked bar chart (bottom plot)
# Create a pivot table
pivot = segment_revenue.pivot_table(index='Country', columns='CustomerNo', values='Revenue', fill_value=0)

# Normalize to make it unreadable
pivot = pivot.div(pivot.sum(axis=1), axis=0)

# Plot each customer segment as a bar
bottoms = np.zeros(len(pivot))
for i, customer in enumerate(pivot.columns):
    axs[1].barh(pivot.index, pivot[customer], left=bottoms, label=f'X{i}', color=np.random.choice(colors))
    bottoms += pivot[customer]

# Put legend on top of data
axs[1].legend(loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=6, ncol=4)
axs[1].set_xlabel('Country Name')
axs[1].set_ylabel('Revenue Units')
axs[1].set_title('Customer Segments by Moon Phase', fontsize=10)

# Swap axis labels
axs[1].set_xlabel('Country Name')
axs[1].set_ylabel('Revenue Units')

# Save the figure
plt.savefig('chart.png')