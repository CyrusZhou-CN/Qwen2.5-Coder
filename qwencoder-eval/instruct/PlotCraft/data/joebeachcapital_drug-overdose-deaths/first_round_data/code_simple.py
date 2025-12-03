import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('VSRR_Provisional_Drug_Overdose_Death_Counts.csv')

# Filter data for Alaska and 2015
alaska_2015 = df[(df['State'] == 'AK') & (df['Year'] == 2015)].copy()

# Filter out rows where 'Data Value' is null or cannot be converted to numeric
alaska_2015 = alaska_2015[alaska_2015['Data Value'].notna()]

# Function to convert data value to numeric, handling non-numeric strings
def convert_to_numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

alaska_2015['Data Value Numeric'] = alaska_2015['Data Value'].apply(convert_to_numeric)

# Remove rows where conversion failed
alaska_2015 = alaska_2015[alaska_2015['Data Value Numeric'].notna()]

# Remove percentage and total count indicators to focus on specific drug types
exclude_indicators = ['Percent with drugs specified', 'Number of Deaths']
alaska_2015 = alaska_2015[~alaska_2015['Indicator'].isin(exclude_indicators)]

# Group by Indicator and sum the values (in case there are multiple entries)
drug_composition = alaska_2015.groupby('Indicator')['Data Value Numeric'].sum().sort_values(ascending=False)

# Remove any zero or negative values
drug_composition = drug_composition[drug_composition > 0]

# Simplify drug type names for better readability
def simplify_drug_name(indicator):
    """Simplify drug type names by removing ICD codes and shortening long names"""
    name_mapping = {
        'Opioids (T40.0-T40.4,T40.6)': 'Opioids',
        'Natural, semi-synthetic, & synthetic opioids, incl. methadone (T40.2-T40.4)': 'Natural & Synthetic Opioids',
        'Natural & semi-synthetic opioids, incl. methadone (T40.2, T40.3)': 'Natural & Semi-synthetic Opioids',
        'Natural & semi-synthetic opioids (T40.2)': 'Natural Opioids',
        'Heroin (T40.1)': 'Heroin',
        'Psychostimulants with abuse potential (T43.6)': 'Psychostimulants',
        'Synthetic opioids, excl. methadone (T40.4)': 'Synthetic Opioids',
        'Methadone (T40.3)': 'Methadone'
    }
    return name_mapping.get(indicator, indicator)

# Apply name simplification
simplified_names = [simplify_drug_name(name) for name in drug_composition.index]

# Calculate percentages
percentages = (drug_composition.values / drug_composition.sum()) * 100

# Create the pie chart with white background and larger size
plt.figure(figsize=(14, 10))
plt.style.use('default')  # Ensure white background

# Create improved color palette with more distinct colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Function to create labels with percentages and values
def create_label(name, value, percentage):
    if percentage < 3:  # For very small slices, show only percentage
        return f'{percentage:.1f}%'
    else:
        return f'{name}\n({percentage:.1f}%)'

# Create labels for each slice
labels = [create_label(name, value, pct) for name, value, pct in zip(simplified_names, drug_composition.values, percentages)]

# Create pie chart with improved styling
wedges, texts, autotexts = plt.pie(drug_composition.values, 
                                  labels=labels,
                                  colors=colors[:len(drug_composition)],
                                  autopct=lambda pct: f'{int(pct/100 * drug_composition.sum())}' if pct > 3 else '',
                                  startangle=90,
                                  textprops={'fontsize': 10, 'weight': 'bold'},
                                  pctdistance=0.85,
                                  labeldistance=1.1)

# Improve text formatting for better readability
for text in texts:
    text.set_fontsize(9)
    text.set_weight('bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)
    autotext.set_weight('bold')

# Add total deaths as subtitle
total_deaths = int(drug_composition.sum())
plt.figtext(0.5, 0.92, f'Total Drug Overdose Deaths: {total_deaths}', 
           ha='center', fontsize=12, style='italic')

# Set title with bold formatting and better positioning
plt.title('Drug Overdose Deaths by Drug Type - Alaska 2015', 
          fontsize=18, 
          fontweight='bold', 
          pad=30)

# Ensure the pie chart is circular and centered
plt.axis('equal')

# Center the chart in the figure
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)

# Show the plot
plt.show()