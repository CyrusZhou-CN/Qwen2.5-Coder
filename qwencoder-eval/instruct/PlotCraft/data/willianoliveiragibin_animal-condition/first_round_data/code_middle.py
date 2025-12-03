import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Set awful style
plt.style.use('dark_background')

# Create 1x3 layout instead of requested 2x1
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Force terrible spacing
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Get all symptoms from all columns
all_symptoms = []
for col in ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']:
    all_symptoms.extend(df[col].dropna().tolist())

# Get top 10 symptoms
symptom_counts = pd.Series(all_symptoms).value_counts().head(10)

# Create symptom frequency by animal (but plot it as scatter instead of stacked bar)
animals = df['AnimalName'].unique()
symptom_data = {}
for animal in animals:
    animal_df = df[df['AnimalName'] == animal]
    animal_symptoms = []
    for col in ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']:
        animal_symptoms.extend(animal_df[col].dropna().tolist())
    
    symptom_freq = pd.Series(animal_symptoms).value_counts()
    symptom_data[animal] = [symptom_freq.get(symptom, 0) for symptom in symptom_counts.index]

# Plot 1: Scatter plot instead of stacked bar (wrong chart type)
x_pos = np.arange(len(animals))
colors = plt.cm.jet(np.linspace(0, 1, len(symptom_counts)))

for i, symptom in enumerate(symptom_counts.index):
    y_values = [symptom_data[animal][i] for animal in animals]
    axes[0].scatter([j + np.random.random()*0.3 for j in x_pos], y_values, 
                   c=[colors[i]], s=100, alpha=0.7, label=f"Glarbnok_{i}")

# Wrong labels (swapped)
axes[0].set_xlabel('Symptom Intensity')
axes[0].set_ylabel('Animal Categories')
axes[0].set_title('Random Scatter of Nonsense')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(animals, rotation=90)
axes[0].legend(bbox_to_anchor=(0.5, 0.5), loc='center')

# Plot 2: Line chart instead of pie chart (wrong chart type)
animal_counts = df['AnimalName'].value_counts()
axes[1].plot(range(len(animal_counts)), animal_counts.values, 'o-', linewidth=5, markersize=15)
axes[1].set_xlabel('Time Series')
axes[1].set_ylabel('Frequency Distribution')
axes[1].set_title('Linear Progression Analysis')
axes[1].set_xticks(range(len(animal_counts)))
axes[1].set_xticklabels(animal_counts.index, rotation=45)

# Plot 3: Extra unwanted plot with histogram
axes[2].hist(df['Dangerous'].map({'Yes': 1, 'No': 0}), bins=20, color='magenta', alpha=0.8)
axes[2].set_xlabel('Danger Coefficient')
axes[2].set_ylabel('Statistical Variance')
axes[2].set_title('Probability Matrix')

# Add overlapping text annotations
axes[0].text(0.5, 0.8, 'OVERLAPPING\nTEXT\nEVERYWHERE', transform=axes[0].transAxes, 
             fontsize=20, color='yellow', weight='bold', ha='center')
axes[1].text(0.3, 0.6, 'MORE\nCONFUSING\nLABELS', transform=axes[1].transAxes,
             fontsize=16, color='red', weight='bold')

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()