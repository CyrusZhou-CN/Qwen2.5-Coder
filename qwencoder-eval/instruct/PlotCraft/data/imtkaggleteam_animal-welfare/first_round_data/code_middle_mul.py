import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Load all datasets
df_direct = pd.read_csv('1- animal-lives-lost-direct.csv')
df_total = pd.read_csv('2- animal-lives-lost-total.csv')
df_meat_per_animal = pd.read_csv('3- kilograms-meat-per-animal.csv')
df_housing_share = pd.read_csv('5- share-of-eggs-produced-by-different-housing-systems.csv')
df_egg_production = pd.read_csv('6- egg-production-system.csv')

# Data preprocessing
# Clean the kilograms_per_animal_direct column (remove commas)
df_meat_per_animal['kilograms_per_animal_direct'] = df_meat_per_animal['kilograms_per_animal_direct'].astype(str).str.replace(',', '').astype(float)

# Merge direct and total lives lost data
df_lives = pd.merge(df_direct[['Entity', 'lives_per_kg_direct']], 
                   df_total[['Entity', 'lives_per_kg_total']], on='Entity')
df_lives['lives_per_kg_indirect'] = df_lives['lives_per_kg_total'] - df_lives['lives_per_kg_direct']

# Filter UK data for egg housing systems
df_uk_housing = df_housing_share[df_housing_share['Entity'] == 'United Kingdom'].copy()
df_uk_production = df_egg_production[df_egg_production['Entity'] == 'United Kingdom'].copy()

# Create 2x2 subplot grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E']
colors_housing = ['#E63946', '#F77F00', '#FCBF49', '#06D6A0']

# Top-left: Stacked bar chart of direct vs total animal lives lost
animal_products = ['Beef', 'Chicken', 'Crab', 'Eggs', 'Fish', 'Dairy butter']
df_selected = df_lives[df_lives['Entity'].isin(animal_products)].copy()
df_selected = df_selected.sort_values('lives_per_kg_total', ascending=True)

x_pos = np.arange(len(df_selected))
direct_values = df_selected['lives_per_kg_direct'].values
indirect_values = df_selected['lives_per_kg_indirect'].values

bars1 = ax1.bar(x_pos, direct_values, color=colors_main[0], label='Direct Lives Lost', alpha=0.8)
bars2 = ax1.bar(x_pos, indirect_values, bottom=direct_values, color=colors_main[1], label='Indirect Lives Lost', alpha=0.8)

ax1.set_xlabel('Animal Products', fontweight='bold')
ax1.set_ylabel('Lives Lost per Kilogram', fontweight='bold')
ax1.set_title('Direct vs Total Animal Lives Lost per Kilogram', fontweight='bold', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_selected['Entity'], rotation=45, ha='right')
ax1.legend(frameon=False)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('white')

# Top-right: Pie chart of meat production by animal type
meat_animals = ['Beef', 'Chicken', 'Fish']
df_meat_selected = df_meat_per_animal[df_meat_per_animal['Entity'].isin(meat_animals)].copy()

sizes = df_meat_selected['kilograms_per_animal_direct'].values
labels = df_meat_selected['Entity'].values

wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                  colors=colors_main[:len(meat_animals)], startangle=90)
ax2.set_title('Meat Production Composition\n(Kilograms per Animal)', fontweight='bold', fontsize=14)

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Bottom-left: Treemap of UK egg production systems (latest year)
latest_year = df_uk_production['Year'].max()
df_latest = df_uk_production[df_uk_production['Year'] == latest_year].iloc[0]

# Calculate treemap data
housing_types = {
    'Cages': df_latest['Number of eggs from hens in (enriched) cages'],
    'Barns': df_latest['Number of eggs from hens in barns'],
    'Free-range (Non-organic)': df_latest['Number of eggs from hens in non-organic, free-range farms'],
    'Free-range (Organic)': df_latest['Number of eggs from hens in organic, free-range farms']
}

total_eggs = sum(housing_types.values())
proportions = {k: v/total_eggs for k, v in housing_types.items()}

# Simple treemap implementation
def create_treemap(ax, data, colors):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Sort by size for better layout
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate rectangles
    x, y = 0, 0
    width, height = 1, 1
    
    for i, (label, value) in enumerate(sorted_items):
        if i == 0:  # Largest rectangle
            rect_width = width * 0.6
            rect_height = height
            rect = Rectangle((x, y), rect_width, rect_height, 
                           facecolor=colors[i], edgecolor='white', linewidth=2)
        elif i == 1:  # Second largest
            rect_width = width * 0.4
            rect_height = height * 0.7
            rect = Rectangle((0.6, y), rect_width, rect_height, 
                           facecolor=colors[i], edgecolor='white', linewidth=2)
        elif i == 2:  # Third
            rect_width = width * 0.4
            rect_height = height * 0.2
            rect = Rectangle((0.6, 0.7), rect_width, rect_height, 
                           facecolor=colors[i], edgecolor='white', linewidth=2)
        else:  # Smallest
            rect_width = width * 0.4
            rect_height = height * 0.1
            rect = Rectangle((0.6, 0.9), rect_width, rect_height, 
                           facecolor=colors[i], edgecolor='white', linewidth=2)
        
        ax.add_patch(rect)
        
        # Add text
        text_x = rect.get_x() + rect.get_width()/2
        text_y = rect.get_y() + rect.get_height()/2
        ax.text(text_x, text_y, f'{label}\n{value:.1%}', 
               ha='center', va='center', fontweight='bold', fontsize=10, color='white')

create_treemap(ax3, proportions, colors_housing)
ax3.set_title('UK Egg Production by Housing System\n(Hierarchical Composition)', fontweight='bold', fontsize=14)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_facecolor('white')

# Bottom-right: Stacked area chart of housing systems over time
df_uk_housing_clean = df_uk_housing.dropna()
years = df_uk_housing_clean['Year'].values
cages = df_uk_housing_clean['Share of hens in cages'].values / 100
barns = df_uk_housing_clean['Share of hens housed in a barn or aviary'].values / 100
free_range_non_organic = df_uk_housing_clean['Share of non-organic, free-range hens'].values / 100
free_range_organic = df_uk_housing_clean['Share of organic, free-range hens'].values / 100

# Create stacked areas
ax4.fill_between(years, 0, cages, color=colors_housing[0], alpha=0.8, label='Cages')
ax4.fill_between(years, cages, cages + barns, color=colors_housing[1], alpha=0.8, label='Barns')
ax4.fill_between(years, cages + barns, cages + barns + free_range_non_organic, 
                color=colors_housing[2], alpha=0.8, label='Free-range (Non-organic)')
ax4.fill_between(years, cages + barns + free_range_non_organic, 
                cages + barns + free_range_non_organic + free_range_organic,
                color=colors_housing[3], alpha=0.8, label='Free-range (Organic)')

ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Share of Production', fontweight='bold')
ax4.set_title('Evolution of UK Egg Housing Systems\n(Temporal Composition)', fontweight='bold', fontsize=14)
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax4.set_facecolor('white')
ax4.set_ylim(0, 1)

# Overall layout adjustments
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()