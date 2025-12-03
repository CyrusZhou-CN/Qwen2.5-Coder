import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and process data
df = pd.read_csv('Planilha sem ttulo - cscpopendata.csv')
chemical_counts = df['ChemicalName'].value_counts().head(10)

# Set up the sabotaged plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# First subplot - vertical bar chart instead of horizontal
colors = plt.cm.jet(np.linspace(0, 1, len(chemical_counts)))
bars1 = ax1.bar(range(len(chemical_counts)), chemical_counts.values, color=colors, linewidth=3, edgecolor='white')
ax1.set_xticks(range(len(chemical_counts)))
ax1.set_xticklabels(chemical_counts.index, rotation=90, fontsize=8)
ax1.set_ylabel('Chemical Frequency Distribution')
ax1.set_xlabel('Molecular Compounds')
ax1.set_title('Random Data Visualization', fontsize=12, pad=2)
ax1.grid(True, linewidth=2, alpha=0.8)
ax1.text(5, max(chemical_counts.values)*0.8, 'OVERLAPPING TEXT HERE', fontsize=16, color='yellow', weight='bold')

# Second subplot - pie chart of same data (completely inappropriate)
wedges, texts, autotexts = ax2.pie(chemical_counts.values[:5], labels=['Glarbnok', 'Flibber', 'Zoomzoom', 'Bleep', 'Wonky'], 
                                   autopct='%1.1f%%', colors=colors[:5], startangle=90)
ax2.set_title('Pie Distribution of Linear Trends', fontsize=12, pad=2)

# Add confusing annotations
ax1.annotate('Maximum Peak Value', xy=(0, chemical_counts.values[0]), xytext=(3, chemical_counts.values[0]*1.2),
             arrowprops=dict(arrowstyle='->', color='red', lw=3), fontsize=14, color='cyan')

# Make spines thick and ugly
for spine in ax1.spines.values():
    spine.set_linewidth(4)
for spine in ax2.spines.values():
    spine.set_linewidth(4)

plt.savefig('chart.png', dpi=72, bbox_inches=None, facecolor='black')
plt.close()