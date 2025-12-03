import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('indian_rda_based_diet_recommendation_system.csv')

# Calculate the average macronutrient composition for each meal type
macronutrients = ['Calories', 'Fats', 'Proteins', 'Carbohydrates']
average_macronutrients = data[macronutrients].mean().reset_index()
average_macronutrients.columns = ['Nutrient', 'Average']

# Prepare data for the pie chart
veg_non_veg_counts = data['VegNovVeg'].value_counts()

# Create the figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Stacked bar chart for macronutrient composition
axes[0].barh(average_macronutrients['Nutrient'], average_macronutrients['Average'], color=['blue', 'orange', 'green', 'red'])
axes[0].set_title('Average Macronutrient Composition by Meal Type')
axes[0].set_xlabel('Average Value')
axes[0].legend(['Breakfast', 'Lunch', 'Dinner'])

# Pie chart for distribution of vegetarian vs non-vegetarian dishes
axes[1].pie(veg_non_veg_counts, labels=veg_non_veg_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
axes[1].set_title('Distribution of Vegetarian vs Non-Vegetarian Dishes')

plt.tight_layout()
plt.show()