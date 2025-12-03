import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the worst possible style
plt.style.use('dark_background')

# Load all datasets
datasets = {
    'Variant III': pd.read_csv('Variant III.csv'),
    'Variant IV': pd.read_csv('Variant IV.csv'),
    'Variant II': pd.read_csv('Variant II.csv'),
    'Variant I': pd.read_csv('Variant I.csv'),
    'Variant V': pd.read_csv('Variant V.csv'),
    'Base': pd.read_csv('Base.csv')
}

# Create a 2x3 layout instead of the requested histogram
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('black')

# Flatten axes for easier iteration
axes = axes.flatten()

# Use completely wrong chart types - pie charts for each variant
colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
variant_names = list(datasets.keys())

for i, (name, data) in enumerate(datasets.items()):
    # Create pie chart instead of histogram
    credit_scores = data['credit_risk_score']
    
    # Bin the data arbitrarily for pie chart
    bins = [0, 50, 100, 150, 200, 250, 300]
    binned_data = pd.cut(credit_scores, bins=bins)
    counts = binned_data.value_counts()
    
    # Create pie chart with awful colors
    axes[i].pie(counts.values, labels=None, colors=['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'], 
                autopct='%1.1f%%', startangle=90)
    
    # Wrong and confusing titles
    axes[i].set_title(f'Pizza Distribution for {name}', fontsize=8, color='white')

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.05)

# Add completely unrelated main title
fig.suptitle('Temperature Variations Across Different Planets', fontsize=12, color='white', y=0.98)

# Add random text annotations that overlap everything
for i in range(6):
    axes[i].text(0, 0, 'GLARBNOK REVENGE DATA', fontsize=16, color='white', 
                ha='center', va='center', weight='bold', alpha=0.8)

# Save the chart
plt.savefig('chart.png', facecolor='black', dpi=100)
plt.close()