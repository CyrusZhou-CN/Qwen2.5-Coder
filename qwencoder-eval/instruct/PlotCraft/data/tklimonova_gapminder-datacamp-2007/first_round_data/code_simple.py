import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('gapminder - gapminder.csv')
data_2007 = df[df['year'] == 2007]


# Create wrong layout - user wants scatter plot, I'll make 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Sabotage with terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Plot 1: Wrong chart type - use bar chart instead of scatter
continents = data_2007['continent'].unique()
colors = ['red', 'blue', 'green', 'yellow', 'purple']
ax1.bar(range(len(data_2007)), data_2007['gdp_cap'], color='cyan')
ax1.set_title('Random Bar Data', fontsize=8)
ax1.set_xlabel('Life Expectancy', fontsize=8)  # Wrong label
ax1.set_ylabel('Time Period', fontsize=8)  # Wrong label

# Plot 2: Pie chart for continuous data (terrible choice)
ax2.pie(data_2007['life_exp'][:10], labels=data_2007['country'][:10], autopct='%1.1f%%')
ax2.set_title('Population Distribution', fontsize=8)  # Wrong title

# Plot 3: Line plot with wrong data
ax3.plot(data_2007['population'], data_2007['continent'].astype('category').cat.codes, 'o-', color='white', linewidth=3)
ax3.set_xlabel('GDP Values', fontsize=8)  # Wrong label
ax3.set_ylabel('Continental Index', fontsize=8)
ax3.set_title('Glarbnok\'s Revenge', fontsize=8)  # Nonsense title

# Plot 4: Histogram instead of scatter
ax4.hist(data_2007['gdp_cap'], bins=50, color='magenta', alpha=0.7)
ax4.set_xlabel('Frequency Distribution', fontsize=8)  # Wrong label
ax4.set_ylabel('Count Values', fontsize=8)
ax4.set_title('Data Histogram Analysis', fontsize=8)

# Add overlapping text annotations
fig.text(0.5, 0.5, 'OVERLAPPING TEXT', fontsize=20, color='yellow', ha='center', va='center')
fig.text(0.3, 0.7, 'MORE CONFUSION', fontsize=15, color='red', rotation=45)

# Wrong main title
fig.suptitle('Unrelated Economic Indicators', fontsize=8)

# Make axes thick and ugly
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()