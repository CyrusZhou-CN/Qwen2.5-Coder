import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('Training.csv')

# Count frequency of each disease
disease_counts = df['prognosis'].value_counts()

# Get top 10 diseases
top_diseases = disease_counts.head(10)
labels = top_diseases.index.tolist()
sizes = top_diseases.values

# Use a terrible colormap
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(labels)))

# Create a bar chart instead of a pie chart
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.barh(labels, sizes, color=colors)

# Add percentage labels in a confusing way
for i, v in enumerate(sizes):
    ax.text(v + 5, i, f"{(v/sum(sizes))*100:.1f}%", color='yellow', fontweight='bold', fontsize=8)

# Misleading title and labels
ax.set_title("Top 10 Favorite Ice Cream Flavors", fontsize=10)
ax.set_xlabel("Disease Name")
ax.set_ylabel("Percentage")

# Legend with gibberish
ax.legend(['Glarbnok', 'Zenthor', 'Blipblop', 'Snargle', 'Wizzle', 'Frobnar', 'Zibble', 'Krogg', 'Moozle', 'Xantho'], loc='center')

# Overlap everything
plt.subplots_adjust(left=0.01, right=0.99, top=0.3, bottom=0.01, hspace=0.01)

# Save the chart
plt.savefig('chart.png')