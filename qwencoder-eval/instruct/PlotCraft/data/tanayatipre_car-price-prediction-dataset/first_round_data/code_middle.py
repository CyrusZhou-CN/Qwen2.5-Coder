import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('train_data.csv')

# Convert Employment length to positive years
data['Employment length'] = abs(data['Employment length'])

# Convert Age to positive years
data['Age'] = abs(data['Age'])

# Left plot: Scatter plot of Income vs Employment length colored by Employment status
plt.figure(figsize=(12, 6))

# Scatter plot
scatter_plot = sns.scatterplot(x='Employment length', y='Income', hue='Employment status', data=data, palette='viridis')
plt.title('Income vs Employment Length')
plt.xlabel('Employment Length (years)')
plt.ylabel('Income')

# Trend line
sns.regplot(x='Employment length', y='Income', data=data[data['Employment status'] == 'Working'], scatter=False, color='blue', label='Working')
sns.regplot(x='Employment length', y='Income', data=data[data['Employment status'] == 'Commercial associate'], scatter=False, color='red', label='Commercial associate')
sns.regplot(x='Employment length', y='Income', data=data[data['Employment status'] == 'Laborers'], scatter=False, color='green', label='Laborers')
sns.regplot(x='Employment length', y='Income', data=data[data['Employment status'] == 'Managers'], scatter=False, color='purple', label='Managers')

# Right plot: Correlation heatmap
plt.subplot(1, 2, 2)
correlation_matrix = data[['Income', 'Employment length', 'Age', 'Family member count']].corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()