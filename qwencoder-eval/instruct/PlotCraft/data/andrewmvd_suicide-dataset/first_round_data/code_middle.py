import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('suicide_dataset.csv')

# Data preprocessing
# Get the most recent year of data for each country and sex combination
df_recent = df.groupby(['Country', 'Sex']).apply(lambda x: x.loc[x['Year'].idxmax()]).reset_index(drop=True)

# Filter out rows with missing suicide rate data
df_recent = df_recent.dropna(subset=['Suicide Rate'])

# Get overall suicide rates (Both sexes) and find top 15 countries
both_sexes_data = df_recent[df_recent['Sex'] == 'Both sexes'].copy()
top_15_countries = both_sexes_data.nlargest(15, 'Suicide Rate')['Country'].tolist()

# Filter data for top 15 countries
df_top15 = df_recent[df_recent['Country'].isin(top_15_countries)].copy()

# Prepare data for plotting
both_sexes_top15 = df_top15[df_top15['Sex'] == 'Both sexes'].sort_values('Suicide Rate', ascending=True)
male_data = df_top15[df_top15['Sex'] == 'Male'].set_index('Country')['Suicide Rate']
female_data = df_top15[df_top15['Sex'] == 'Female'].set_index('Country')['Suicide Rate']

# Create the composite visualization with white background
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
fig.patch.set_facecolor('white')

# Top subplot: Overall suicide rates (Both sexes)
bars1 = ax1.barh(both_sexes_top15['Country'], both_sexes_top15['Suicide Rate'], 
                 color='#2E86AB', alpha=0.8, height=0.7)
ax1.set_xlabel('Suicide Rate (per 100,000 population)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Country', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Countries by Overall Suicide Rates', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('white')

# Add value labels on bars
for bar in bars1:
    width = bar.get_width()
    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}', ha='left', va='center', fontsize=10)

# Bottom subplot: Gender comparison
countries_ordered = both_sexes_top15['Country'].tolist()
y_pos = np.arange(len(countries_ordered))

# Get male and female rates for ordered countries
male_rates = [male_data.get(country, 0) for country in countries_ordered]
female_rates = [female_data.get(country, 0) for country in countries_ordered]

# Create grouped horizontal bar chart
bar_height = 0.35
bars_male = ax2.barh(y_pos - bar_height/2, male_rates, bar_height, 
                     label='Male', color='#A23B72', alpha=0.8)
bars_female = ax2.barh(y_pos + bar_height/2, female_rates, bar_height, 
                       label='Female', color='#F18F01', alpha=0.8)

ax2.set_xlabel('Suicide Rate (per 100,000 population)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Country', fontsize=12, fontweight='bold')
ax2.set_title('Gender Comparison of Suicide Rates (Top 15 Countries)', fontsize=14, fontweight='bold', pad=20)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(countries_ordered)
ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.legend(loc='lower right', fontsize=11)
ax2.set_facecolor('white')

# Add value labels for gender comparison
for i, (male_rate, female_rate) in enumerate(zip(male_rates, female_rates)):
    if male_rate > 0:
        ax2.text(male_rate + 1, i - bar_height/2, f'{male_rate:.1f}', 
                ha='left', va='center', fontsize=9)
    if female_rate > 0:
        ax2.text(female_rate + 1, i + bar_height/2, f'{female_rate:.1f}', 
                ha='left', va='center', fontsize=9)

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)

# Add a subtle note about gender disparities
fig.text(0.02, 0.02, 'Note: Significant gender disparities are evident, with male rates typically 2-4 times higher than female rates', 
         fontsize=10, style='italic', alpha=0.7)

plt.show()