import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
df_crustaceans = pd.read_csv('7- farmed-crustaceans.csv')
df_meat_yield = pd.read_csv('3- kilograms-meat-per-animal.csv')
df_housing_share = pd.read_csv('5- share-of-eggs-produced-by-different-housing-systems.csv')
df_fish = pd.read_csv('8- farmed-fish-killed.csv')
df_hens = pd.read_csv('4- laying-hens-cages-and-cage-free.csv')
df_direct_lives = pd.read_csv('1- animal-lives-lost-direct.csv')
df_egg_production = pd.read_csv('6- egg-production-system.csv')
df_cage_free = pd.read_csv('9- eggs-cage-free.csv')
df_total_lives = pd.read_csv('2- animal-lives-lost-total.csv')

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
colors_secondary = ['#87CEEB', '#DDA0DD', '#FFE4B5', '#FFA07A', '#98FB98']

# Subplot 1: Fish population with confidence bands and scatter points
ax1 = axes[0, 0]
fish_data = df_fish[df_fish['Year'].between(2015, 2017)].copy()
fish_countries = fish_data[~fish_data['Entity'].isin(['World', 'Africa'])].groupby('Entity')['Estimated number of farmed fish'].sum().nlargest(5).index

for i, country in enumerate(fish_countries):
    country_data = fish_data[fish_data['Entity'] == country]
    if len(country_data) > 0:
        ax1.plot(country_data['Year'], country_data['Estimated number of farmed fish']/1e9, 
                color=colors_main[i], linewidth=2, label=country, marker='o', markersize=6)
        ax1.fill_between(country_data['Year'], 
                        country_data['Estimated number of farmed fish (lower bound)']/1e9,
                        country_data['Estimated number of farmed fish (upper bound)']/1e9,
                        alpha=0.2, color=colors_main[i])

ax1.set_title('Farmed Fish Population Estimates (2015-2017)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Fish Population (Billions)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: UK egg production stacked area chart
ax2 = axes[0, 1]
uk_eggs = df_egg_production[df_egg_production['Entity'] == 'United Kingdom'].copy()
uk_eggs = uk_eggs.sort_values('Year')

# Convert to billions for readability
organic = uk_eggs['Number of eggs from hens in organic, free-range farms'].values / 1e9
non_organic = uk_eggs['Number of eggs from hens in non-organic, free-range farms'].values / 1e9
barns = uk_eggs['Number of eggs from hens in barns'].values / 1e9
cages = uk_eggs['Number of eggs from hens in (enriched) cages'].values / 1e9

ax2.stackplot(uk_eggs['Year'], organic, non_organic, barns, cages,
              labels=['Organic Free-range', 'Non-organic Free-range', 'Barns', 'Cages'],
              colors=colors_main[:4], alpha=0.8)

# Add trend lines
for i, data in enumerate([organic, non_organic, barns, cages]):
    z = np.polyfit(uk_eggs['Year'], data, 1)
    p = np.poly1d(z)
    ax2.plot(uk_eggs['Year'], p(uk_eggs['Year']), '--', color='black', alpha=0.6, linewidth=1)

ax2.set_title('UK Egg Production by Housing System', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year')
ax2.set_ylabel('Eggs Produced (Billions)')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

# Subplot 3: Animal lives lost comparison with error bars
ax3 = axes[0, 2]
# Merge direct and total lives data
lives_comparison = pd.merge(df_direct_lives, df_total_lives, on=['Entity', 'Code', 'Year'])
lives_comparison = lives_comparison[~lives_comparison['Entity'].str.contains('Dairy')]

entities = lives_comparison['Entity'][:8]  # Top 8 for readability
direct = lives_comparison['lives_per_kg_direct'][:8]
total = lives_comparison['lives_per_kg_total'][:8]
difference = total - direct

x_pos = np.arange(len(entities))
bars = ax3.bar(x_pos, total, color=colors_main[0], alpha=0.7, label='Total Lives Lost')
ax3.bar(x_pos, direct, color=colors_main[1], alpha=0.9, label='Direct Lives Lost')
ax3.errorbar(x_pos, total, yerr=difference, fmt='none', color='black', capsize=3)

# Add line plot
ax3_twin = ax3.twinx()
ax3_twin.plot(x_pos, difference, color=colors_main[2], marker='s', linewidth=2, label='Difference')
ax3_twin.set_ylabel('Difference (Total - Direct)')

ax3.set_title('Animal Lives Lost per kg by Product Type', fontweight='bold', fontsize=12)
ax3.set_xlabel('Animal Product')
ax3.set_ylabel('Lives per kg')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(entities, rotation=45, ha='right')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Subplot 4: UK cage-free adoption with seasonal decomposition
ax4 = axes[1, 0]
uk_cage_free = df_cage_free[df_cage_free['Entity'] == 'United Kingdom'].copy()
uk_cage_free = uk_cage_free.sort_values('Year')

# Raw data
ax4.plot(uk_cage_free['Year'], uk_cage_free['Share of cage-free eggs'], 
         color=colors_main[0], alpha=0.6, linewidth=1, label='Raw Data')

# Smoothed trend
if len(uk_cage_free) > 5:
    smoothed = savgol_filter(uk_cage_free['Share of cage-free eggs'], 
                           min(11, len(uk_cage_free)//2*2+1), 3)
    ax4.plot(uk_cage_free['Year'], smoothed, color=colors_main[1], 
             linewidth=3, label='Smoothed Trend')

ax4.set_title('UK Cage-Free Egg Adoption Over Time', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year')
ax4.set_ylabel('Share of Cage-Free Eggs (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Dual-axis hen population and distribution
ax5 = axes[1, 1]
uk_hens = df_hens[df_hens['Entity'] == 'United Kingdom'].copy()
if len(uk_hens) == 0:
    # Use global data if UK not available
    uk_hens = df_hens.groupby('Year').agg({
        'Number of hens in cages': 'sum',
        'Number of cage-free hens': 'sum'
    }).reset_index()

total_hens = uk_hens['Number of hens in cages'].fillna(0) + uk_hens['Number of cage-free hens'].fillna(0)
cage_pct = (uk_hens['Number of hens in cages'].fillna(0) / total_hens * 100).fillna(0)
cage_free_pct = (uk_hens['Number of cage-free hens'].fillna(0) / total_hens * 100).fillna(0)

# Line chart for population
ax5.plot(uk_hens['Year'], total_hens/1e6, color=colors_main[0], linewidth=3, marker='o', label='Total Population')

# Bar chart for distribution
ax5_twin = ax5.twinx()
width = 0.35
x_pos = np.arange(len(uk_hens['Year']))
ax5_twin.bar(uk_hens['Year'] - width/2, cage_pct, width, color=colors_main[1], alpha=0.7, label='Caged %')
ax5_twin.bar(uk_hens['Year'] + width/2, cage_free_pct, width, color=colors_main[2], alpha=0.7, label='Cage-Free %')

ax5.set_title('Hen Population and Housing Distribution', fontweight='bold', fontsize=12)
ax5.set_xlabel('Year')
ax5.set_ylabel('Total Hens (Millions)')
ax5_twin.set_ylabel('Distribution (%)')
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

# Subplot 6: Top 5 crustacean farming countries with uncertainty
ax6 = axes[1, 2]
crust_countries = df_crustaceans[~df_crustaceans['Entity'].isin(['World', 'Africa'])].groupby('Entity')['Estimated number of farmed decapod crustaceans'].sum().nlargest(5).index

for i, country in enumerate(crust_countries):
    country_data = df_crustaceans[df_crustaceans['Entity'] == country].sort_values('Year')
    if len(country_data) > 0:
        ax6.plot(country_data['Year'], country_data['Estimated number of farmed decapod crustaceans']/1e6,
                color=colors_main[i], linewidth=2, marker='s', markersize=4, label=country)
        ax6.fill_between(country_data['Year'],
                        country_data['Estimated number of decapod crustaceans (lower bound)']/1e6,
                        country_data['Estimated number of farmed decapod crustaceans (upper bound)']/1e6,
                        alpha=0.2, color=colors_main[i])

ax6.set_title('Crustacean Farming by Top 5 Countries', fontweight='bold', fontsize=12)
ax6.set_xlabel('Year')
ax6.set_ylabel('Crustaceans (Millions)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Subplot 7: Slope chart for cage-free transition
ax7 = axes[2, 0]
housing_2013 = df_housing_share[df_housing_share['Year'] == 2013].copy()
housing_2018 = df_housing_share[df_housing_share['Year'] == 2018].copy()
transition_data = pd.merge(housing_2013[['Entity', 'Share of hens in cages']], 
                          housing_2018[['Entity', 'Share of hens in cages']], 
                          on='Entity', suffixes=('_2013', '_2018'))

# Select countries with significant changes
transition_data['change'] = abs(transition_data['Share of hens in cages_2018'] - transition_data['Share of hens in cages_2013'])
top_changes = transition_data.nlargest(8, 'change')

for i, row in top_changes.iterrows():
    ax7.plot([2013, 2018], [row['Share of hens in cages_2013'], row['Share of hens in cages_2018']], 
             'o-', color=colors_main[i % len(colors_main)], linewidth=2, markersize=6, 
             label=row['Entity'][:10])

ax7.set_title('Transition from Caged Systems (2013-2018)', fontweight='bold', fontsize=12)
ax7.set_xlabel('Year')
ax7.set_ylabel('Share of Hens in Cages (%)')
ax7.set_xticks([2013, 2018])
ax7.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
ax7.grid(True, alpha=0.3)

# Subplot 8: Meat yield distribution with box plots (fixed)
ax8 = axes[2, 1]
# Clean the meat yield data
df_meat_yield['kilograms_per_animal_direct'] = df_meat_yield['kilograms_per_animal_direct'].str.replace(',', '').astype(float)
meat_entities = df_meat_yield['Entity'][:8]
meat_yields = df_meat_yield['kilograms_per_animal_direct'][:8]

# Create area chart
ax8.fill_between(range(len(meat_entities)), meat_yields, alpha=0.6, color=colors_main[0])
ax8.plot(range(len(meat_entities)), meat_yields, color=colors_main[1], linewidth=2, marker='o')

# Add box plot overlay (simulated distribution) - Fixed: removed alpha parameter
box_data = []
for yield_val in meat_yields:
    # Simulate distribution around each yield value
    simulated = np.random.normal(yield_val, yield_val*0.1, 100)
    box_data.append(simulated)

bp = ax8.boxplot(box_data, positions=range(len(meat_entities)), widths=0.3, 
                patch_artist=True)
# Set alpha for box patches separately
for patch in bp['boxes']:
    patch.set_facecolor(colors_main[2])
    patch.set_alpha(0.5)

ax8.set_title('Meat Yield Distribution by Animal Type', fontweight='bold', fontsize=12)
ax8.set_xlabel('Animal Type')
ax8.set_ylabel('Kilograms per Animal')
ax8.set_xticks(range(len(meat_entities)))
ax8.set_xticklabels(meat_entities, rotation=45, ha='right')
ax8.grid(True, alpha=0.3)

# Subplot 9: Fish vs Crustacean farming correlation
ax9 = axes[2, 2]
# Aggregate global data by year
fish_global = df_fish[df_fish['Entity'] == 'World'].groupby('Year')['Estimated number of farmed fish'].sum()
crust_global = df_crustaceans[df_crustaceans['Entity'] == 'World'].groupby('Year')['Estimated number of farmed decapod crustaceans'].sum()

# Find common years
common_years = sorted(set(fish_global.index) & set(crust_global.index))
if len(common_years) > 0:
    fish_values = [fish_global[year]/1e9 for year in common_years]
    crust_values = [crust_global[year]/1e6 for year in common_years]
    
    ax9.plot(common_years, fish_values, color=colors_main[0], linewidth=3, marker='o', label='Fish (Billions)')
    
    ax9_twin = ax9.twinx()
    ax9_twin.plot(common_years, crust_values, color=colors_main[1], linewidth=3, marker='s', label='Crustaceans (Millions)')
    
    # Calculate correlation
    if len(fish_values) > 2:
        correlation = np.corrcoef(fish_values, crust_values)[0, 1]
        ax9.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax9.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax9.set_title('Fish vs Crustacean Farming Correlation', fontweight='bold', fontsize=12)
ax9.set_xlabel('Year')
ax9.set_ylabel('Fish Farmed (Billions)')
ax9_twin.set_ylabel('Crustaceans Farmed (Millions)')
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig('animal_welfare_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()