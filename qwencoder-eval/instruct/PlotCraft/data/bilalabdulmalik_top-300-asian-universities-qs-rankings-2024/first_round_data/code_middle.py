import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('topuniversities.csv')

# Data preprocessing - handle any missing values
df = df.dropna(subset=['Academic Reputation', 'Citations per Paper', 'International Faculty', 'Overall Score'])

# Calculate average International Faculty percentage by country
country_intl_faculty = df.groupby('Country')['International Faculty'].mean().sort_values(ascending=False)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], 
                      hspace=0.3, wspace=0.3)

# Main scatter plot
ax1 = fig.add_subplot(gs[0, :])

# Define color palette for countries (using more distinct colors)
countries = df['Country'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
country_colors = dict(zip(countries, colors))

# Create scatter plot
for country in countries:
    country_data = df[df['Country'] == country]
    scatter = ax1.scatter(
        country_data['Academic Reputation'], 
        country_data['Citations per Paper'],
        s=country_data['Overall Score'] * 2.5,  # Size proportional to Overall Score
        c=[country_colors[country]], 
        alpha=0.7,
        label=country,
        edgecolors='white',
        linewidth=0.8
    )

# Styling for main scatter plot
ax1.set_xlabel('Academic Reputation', fontsize=12, fontweight='bold')
ax1.set_ylabel('Citations per Paper', fontsize=12, fontweight='bold')
ax1.set_title('Research Excellence vs. International Engagement in Asian Universities', 
              fontsize=16, fontweight='bold', pad=20)

# Add subtle grid
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Set axis limits for better visualization
ax1.set_xlim(0, 105)
ax1.set_ylim(0, 105)

# Create legend for countries
legend1 = ax1.legend(title='Country', bbox_to_anchor=(1.02, 1), loc='upper left', 
                     fontsize=10, title_fontsize=11, frameon=True, fancybox=True)

# Bar chart for average International Faculty by country
ax2 = fig.add_subplot(gs[1, :])

bars = ax2.bar(range(len(country_intl_faculty)), country_intl_faculty.values, 
               color='steelblue', alpha=0.7, edgecolor='white', linewidth=1)

# Styling for bar chart
ax2.set_xlabel('Country', fontsize=12, fontweight='bold')
ax2.set_ylabel('Avg International\nFaculty (%)', fontsize=11, fontweight='bold')
ax2.set_title('Average International Faculty Percentage by Country', 
              fontsize=14, fontweight='bold', pad=15)

# Set x-axis labels
ax2.set_xticks(range(len(country_intl_faculty)))
ax2.set_xticklabels(country_intl_faculty.index, rotation=45, ha='right', fontsize=10)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Add subtle grid for bar chart
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
ax2.set_axisbelow(True)

# Set background to white
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')
ax2.set_facecolor('white')

# Add explanatory text box in a clear area
textstr = ('Point size represents Overall Score\n'
           'Larger points indicate higher-ranked universities\n'
           'Bar chart shows country-level international faculty trends')
props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='gray')
ax1.text(0.02, 0.25, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.85)

plt.show()