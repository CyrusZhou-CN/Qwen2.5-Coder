import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('DataSet.csv')

# Count the occurrences of each employment type
employment_counts = df['employment_type'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(employment_counts, labels=employment_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors[:len(employment_counts)])
plt.title('Composition of Job Postings by Employment Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()