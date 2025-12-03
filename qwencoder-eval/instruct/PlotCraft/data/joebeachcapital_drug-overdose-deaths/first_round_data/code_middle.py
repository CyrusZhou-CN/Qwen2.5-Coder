import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('VSRR_Provisional_Drug_Overdose_Death_Counts.csv')

# Filter for Alaska and 2015 data
alaska_2015 = df[(df['State'] == 'AK') & (df['Year'] == 2015)].copy()

# Filter for rows where 'Data Value' is not null and can be converted to numeric
# Also exclude percentage and total death count indicators
alaska_2015_filtered = alaska_2015[
    (alaska_2015['Data Value'].notna()) & 
    (~alaska_2015['Indicator'].str.contains('Percent|Number of Deaths', case=False, na=False))
].copy()

# Convert Data Value to numeric, handling any conversion errors
alaska_2015_filtered['Data Value Numeric'] = pd.to_numeric(alaska_2015_filtered['Data Value'], errors='coerce')

# Remove rows where conversion failed
alaska_2015_filtered = alaska_2015_filtered[alaska_2015_filtered['Data Value Numeric'].notna()]

# Group by Indicator and sum the data values (in case there are multiple entries)
drug_deaths = alaska_2015_filtered.groupby('Indicator')['Data Value Numeric'].sum().sort_values(ascending=True)

# Remove any zero values
drug_deaths = drug_deaths[drug_deaths > 0]

# Create color palette
colors = plt.cm.Set3(np.linspace(0, 1, len(drug_deaths)))

# Create 1x2 subplot layout with white background
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('white')

# Left subplot: Horizontal bar chart
bars = ax1.barh(range(len(drug_deaths)), drug_deaths.values, color=colors)
ax1.set_yticks(range(len(drug_deaths)))
ax1.set_yticklabels([label.replace(' (T', '\n(T') for label in drug_deaths.index], fontsize=10)
ax1.set_xlabel('Number of Deaths', fontsize=12, fontweight='bold')
ax1.set_title('Drug Overdose Deaths by Type\nAlaska 2015', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('white')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, drug_deaths.values)):
    ax1.text(value + max(drug_deaths.values) * 0.01, i, f'{int(value)}', 
             va='center', ha='left', fontsize=10, fontweight='bold')

# Right subplot: Pie chart
wedges, texts, autotexts = ax2.pie(drug_deaths.values, 
                                   labels=[label.replace(' (T', '\n(T') for label in drug_deaths.index],
                                   colors=colors,
                                   autopct='%1.1f%%',
                                   startangle=90,
                                   textprops={'fontsize': 9})

# Enhance pie chart text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax2.set_title('Proportional Breakdown of\nDrug Overdose Deaths\nAlaska 2015', 
              fontsize=14, fontweight='bold', pad=20)

# Create legend for pie chart
ax2.legend(wedges, [f'{label}: {int(value)}' for label, value in zip(drug_deaths.index, drug_deaths.values)],
          title="Drug Types (Deaths)", 
          loc="center left", 
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)

plt.show()