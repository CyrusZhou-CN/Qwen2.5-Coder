import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import squarify

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('Training.csv')

# Drop unnamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Get symptom columns
symptom_cols = df.columns[:-1]
disease_col = 'prognosis'

# Calculate total frequency of each symptom
symptom_freq = df[symptom_cols].sum().sort_values(ascending=False)

# Top 15 symptoms
top_symptoms = symptom_freq.head(15).index.tolist()

# Get disease list
diseases = df[disease_col].unique()

# Create a matrix of symptom vs disease
symptom_disease_matrix = pd.DataFrame(0, index=top_symptoms, columns=diseases)

for disease in diseases:
    subset = df[df[disease_col] == disease]
    symptom_disease_matrix.loc[:, disease] = subset[top_symptoms].sum()

# Normalize to proportions
symptom_disease_prop = symptom_disease_matrix.div(symptom_disease_matrix.sum(axis=1), axis=0).fillna(0)

# Generate clashing colors
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(diseases)))

# Create 1x3 layout instead of 2x1
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(hspace=0.01, wspace=0.02)

# Plot horizontal stacked bar chart (in wrong orientation)
bottoms = np.zeros(len(top_symptoms))
for i, disease in enumerate(diseases):
    axs[0].barh(top_symptoms, symptom_disease_prop[disease], left=bottoms, color=colors[i], label=f'Zorg-{i}')
    bottoms += symptom_disease_prop[disease]

axs[0].set_title('Banana Composition', fontsize=10)
axs[0].set_xlabel('Symptom Name')
axs[0].set_ylabel('Proportion')
axs[0].legend(loc='center', fontsize=6)

# Treemap for top 8 diseases
top_diseases = df[disease_col].value_counts().head(8).index.tolist()
treemap_data = []

for disease in top_diseases:
    subset = df[df[disease_col] == disease]
    symptom_counts = subset[symptom_cols].sum()
    for symptom, count in symptom_counts.items():
        if count > 0:
            treemap_data.append({
                'label': f'{disease}\n{symptom}',
                'value': count,
                'category': symptom
            })

# Create DataFrame
treemap_df = pd.DataFrame(treemap_data)

# Assign random clashing colors
color_list = ['lime', 'magenta', 'cyan', 'yellow', 'red', 'blue', 'orange', 'purple']
treemap_df['color'] = np.random.choice(color_list, size=len(treemap_df))

# Plot treemap in third subplot
axs[2].axis('off')
squarify.plot(sizes=treemap_df['value'], label=treemap_df['label'], color=treemap_df['color'], ax=axs[2], text_kwargs={'fontsize':5})

axs[2].set_title('Symptom Jungle', fontsize=10)

# Leave middle subplot blank with random text
axs[1].text(0.5, 0.5, 'ERROR 404\nChart Not Found', ha='center', va='center', fontsize=20, color='red')
axs[1].axis('off')

# Save the figure
plt.savefig('chart.png', dpi=100, facecolor='black')