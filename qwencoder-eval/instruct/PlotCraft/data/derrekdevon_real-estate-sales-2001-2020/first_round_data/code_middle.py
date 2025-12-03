import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('Real_Estate_Sales_2001-2020_GL.csv')

# Data preprocessing
# Remove rows with missing Sale Amount or Assessed Value
df_clean = df.dropna(subset=['Sale Amount', 'Assessed Value'])
df_clean = df_clean[df_clean['Sale Amount'] > 0]
df_clean = df_clean[df_clean['Assessed Value'] > 0]

# Calculate percentage difference: (Sale Amount - Assessed Value) / Assessed Value * 100
df_clean['Percentage_Difference'] = ((df_clean['Sale Amount'] - df_clean['Assessed Value']) / df_clean['Assessed Value']) * 100

# Sample data for better visualization (taking top deviations)
# Sort by absolute percentage difference and take top 100 for the bar chart
df_sorted = df_clean.reindex(df_clean['Percentage_Difference'].abs().sort_values(ascending=False).index)
df_sample = df_sorted.head(100)

# Create color mapping for property types
property_types = df_clean['Property Type'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(property_types)))
color_map = dict(zip(property_types, colors))

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('white')

# Subplot 1: Diverging bar chart
bars = ax1.barh(range(len(df_sample)), df_sample['Percentage_Difference'], 
                color=[color_map[pt] for pt in df_sample['Property Type']])

# Add vertical line at 0
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Styling for subplot 1
ax1.set_xlabel('Percentage Difference (%)', fontweight='bold')
ax1.set_ylabel('Properties (Sorted by Deviation Magnitude)', fontweight='bold')
ax1.set_title('Top 100 Properties by Assessment Deviation\n(Sale Amount vs Assessed Value)', 
              fontweight='bold', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_yticks([])  # Remove y-axis labels for cleaner look

# Add legend for property types
unique_types = df_sample['Property Type'].unique()
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[pt], label=pt) for pt in unique_types]
ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)

# Subplot 2: Box plot showing distribution by property type
property_type_data = []
property_type_labels = []

for prop_type in df_clean['Property Type'].unique():
    if pd.notna(prop_type):
        type_data = df_clean[df_clean['Property Type'] == prop_type]['Sales Ratio']
        # Filter outliers for better visualization (keep values between 0.1 and 3.0)
        type_data_filtered = type_data[(type_data >= 0.1) & (type_data <= 3.0)]
        if len(type_data_filtered) > 0:
            property_type_data.append(type_data_filtered)
            property_type_labels.append(prop_type)

# Create box plot
bp = ax2.boxplot(property_type_data, labels=property_type_labels, patch_artist=True)

# Color the boxes
for patch, prop_type in zip(bp['boxes'], property_type_labels):
    patch.set_facecolor(color_map[prop_type])
    patch.set_alpha(0.7)

# Add horizontal line at 1.0 (perfect assessment)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Assessment (1.0)')

# Styling for subplot 2
ax2.set_xlabel('Property Type', fontweight='bold')
ax2.set_ylabel('Sales Ratio (Sale Amount / Assessed Value)', fontweight='bold')
ax2.set_title('Distribution of Sales Ratios by Property Type', 
              fontweight='bold', fontsize=14, pad=20)
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(loc='upper right')

# Rotate x-axis labels for better readability
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# Overall layout adjustments
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Show the plot
plt.show()