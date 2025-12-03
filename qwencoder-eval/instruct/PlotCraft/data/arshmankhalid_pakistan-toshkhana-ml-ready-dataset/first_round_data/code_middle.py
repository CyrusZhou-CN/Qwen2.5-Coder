import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Pakisan_Toshkhana_Imputed.csv')

# Convert 'Retention' column to boolean
data['Retained'] = data['Retained'].map({'Yes': True, 'No': False})

# Calculate the total assessed value
total_value = data['Assessed Value'].sum()

# Filter out categories that represent less than 5% of the total value
value_threshold = 0.05 * total_value
category_values = data.groupby('Item Category')['Assessed Value'].sum()
filtered_categories = category_values[category_values >= value_threshold].index.tolist()
other_category = category_values[category_values < value_threshold].sum()
if other_category > 0:
    filtered_categories.append('Others')
    data.loc[data['Item Category'].isin(category_values[category_values < value_threshold].index), 'Item Category'] = 'Others'

# Stacked bar chart for distribution of gift categories across retention statuses
plt.figure(figsize=(12, 6))
sns.barplot(x='Item Category', y='Assessed Value', hue='Retained', data=data[data['Item Category'].isin(filtered_categories)], palette=['blue', 'orange'])
plt.title('Distribution of Gift Categories Across Retention Statuses')
plt.xlabel('Item Category')
plt.ylabel('Assessed Value')
plt.legend(title='Retention Status')
plt.show()

# Pie chart for overall proportion of total assessed value by item category
category_proportions = data[data['Item Category'].isin(filtered_categories)].groupby('Item Category')['Assessed Value'].sum() / total_value
plt.figure(figsize=(8, 8))
plt.pie(category_proportions, labels=category_proportions.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Overall Proportion of Total Assessed Value by Item Category')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()