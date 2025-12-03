import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('wine_quality_classification.csv')

# Define the bins for alcohol content
bins = [8, 10, 12, 14]

# Plotting the histogram
plt.figure(figsize=(10, 6))
for label in data['quality_label'].unique():
    subset = data[data['quality_label'] == label]
    plt.hist(subset['alcohol'], bins=bins, alpha=0.5, label=label)

plt.title('Distribution of Alcohol Content by Wine Quality')
plt.xlabel('Alcohol Content (%)')
plt.ylabel('Frequency')
plt.legend(title='Quality Label')
plt.grid(True)
plt.show()