import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
try:
    olympic_2024_horses = pd.read_csv('2024_olympic_horses.csv')
    paralympic_2024_horses = pd.read_csv('2024_paralympic_horses.csv')
    olympic_2020_horses = pd.read_csv('2020_horses_olympic.csv')
    paralympic_2020_horses = pd.read_csv('2020_horses_paralympic.csv')
    olympic_2024_athletes = pd.read_csv('2024_olympic_athletes.csv')
    paralympic_2024_athletes = pd.read_csv('2024_paralympic_athletes.csv')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Data preprocessing - simplified and optimized
def prepare_data():
    # Add competition type and year columns
    olympic_2024_horses['Competition'] = 'Olympic'
    olympic_2024_horses['Year'] = 2024
    paralympic_2024_horses['Competition'] = 'Paralympic'
    paralympic_2024_horses['Year'] = 2024
    
    # Standardize column names for 2020 data
    olympic_2020_horses['Competition'] = 'Olympic'
    olympic_2020_horses['Year'] = 2020
    olympic_2020_horses['Country'] = olympic_2020_horses['Athlete Country']
    
    paralympic_2020_horses['Competition'] = 'Paralympic'
    paralympic_2020_horses['Year'] = 2020
    paralympic_2020_horses['Country'] = paralympic_2020_horses['Athlete Country']
    
    # Calculate horse ages
    for df in [olympic_2024_horses, paralympic_2024_horses, olympic_2020_horses, paralympic_2020_horses]:
        df['Age'] = df['Year'] - df['Year of Birth']
    
    # Select only necessary columns to reduce memory usage
    cols = ['Country', 'Horse Name', 'Discipline', 'Year of Birth', 'Sex', 'Colour', 'Sire', 'Owner', 'Country of Birth', 'Competition', 'Year', 'Age']
    
    # Handle missing Country of Birth
    for df in [olympic_2024_horses, paralympic_2024_horses, olympic_2020_horses, paralympic_2020_horses]:
        if 'Country of Birth' not in df.columns:
            df['Country of Birth'] = 'Unknown'
    
    # Combine all horse data
    all_horses = pd.concat([
        olympic_2024_horses[cols],
        paralympic_2024_horses[cols],
        olympic_2020_horses[cols],
        paralympic_2020_horses[cols]
    ], ignore_index=True)
    
    return all_horses

all_horses = prepare_data()

# Create the 3x3 subplot grid with optimized plotting
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.patch.set_facecolor('white')

# Define color palettes
colors_comp = {'Olympic': '#1f77b4', 'Paralympic': '#ff7f0e'}
colors_year = {2020: '#2ca02c', 2024: '#d62728'}
colors_sex = {'Mare': '#e377c2', 'Gelding': '#17becf', 'Stallion': '#bcbd22'}

# Row 1, Subplot 1: Horse age distribution with mean age trends
ax1 = axes[0, 0]
age_data = []
labels = []
mean_ages = []

for year in [2020, 2024]:
    for comp in ['Olympic', 'Paralympic']:
        data = all_horses[(all_horses['Year'] == year) & (all_horses['Competition'] == comp)]
        if len(data) > 0:
            age_data.append(data['Age'].values)
            labels.append(f'{year}\n{comp}')
            mean_ages.append(data['Age'].mean())

# Simple box plot instead of complex stacked bars
bp = ax1.boxplot(age_data, labels=labels, patch_artist=True)
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add mean line
ax1_twin = ax1.twinx()
ax1_twin.plot(range(1, len(mean_ages)+1), mean_ages, 'ro-', linewidth=2, markersize=6)
ax1_twin.set_ylabel('Mean Age', fontweight='bold')

ax1.set_title('Horse Age Distribution Evolution', fontweight='bold', fontsize=12)
ax1.set_xlabel('Competition', fontweight='bold')
ax1.set_ylabel('Horse Age', fontweight='bold')

# Row 1, Subplot 2: Horse sex distribution
ax2 = axes[0, 1]
sex_counts = all_horses.groupby(['Year', 'Competition', 'Sex']).size().unstack(fill_value=0)
sex_counts.plot(kind='bar', ax=ax2, color=['#e377c2', '#17becf', '#bcbd22'], alpha=0.7)
ax2.set_title('Horse Sex Distribution', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year & Competition', fontweight='bold')
ax2.set_ylabel('Count', fontweight='bold')
ax2.legend(title='Sex')
ax2.tick_params(axis='x', rotation=45)

# Row 1, Subplot 3: Top birth countries
ax3 = axes[0, 2]
birth_countries = all_horses['Country of Birth'].value_counts().head(8)
bars = ax3.barh(range(len(birth_countries)), birth_countries.values, color='skyblue', alpha=0.7)
ax3.set_title('Top Birth Countries', fontweight='bold', fontsize=12)
ax3.set_xlabel('Number of Horses', fontweight='bold')
ax3.set_yticks(range(len(birth_countries)))
ax3.set_yticklabels(birth_countries.index, fontsize=10)

# Row 2, Subplot 4: Event participation over time
ax4 = axes[1, 0]
participation = all_horses.groupby(['Year', 'Competition']).size()
years = [2020, 2024]
olympic_counts = [participation.get((year, 'Olympic'), 0) for year in years]
paralympic_counts = [participation.get((year, 'Paralympic'), 0) for year in years]

ax4.plot(years, olympic_counts, 'o-', label='Olympic', linewidth=2, markersize=8)
ax4.plot(years, paralympic_counts, 's-', label='Paralympic', linewidth=2, markersize=8)
ax4.set_title('Participation Over Time', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Number of Horses', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Age distribution by discipline
ax5 = axes[1, 1]
disciplines = all_horses['Discipline'].value_counts().head(3).index
age_by_disc = []
disc_labels = []

for disc in disciplines:
    data = all_horses[all_horses['Discipline'] == disc]['Age'].dropna()
    if len(data) > 5:  # Only include if sufficient data
        age_by_disc.append(data.values)
        disc_labels.append(disc[:15])  # Truncate long names

if age_by_disc:
    bp = ax5.boxplot(age_by_disc, labels=disc_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

ax5.set_title('Age by Discipline', fontweight='bold', fontsize=12)
ax5.set_xlabel('Discipline', fontweight='bold')
ax5.set_ylabel('Age', fontweight='bold')
ax5.tick_params(axis='x', rotation=45)

# Row 2, Subplot 6: Country participation heatmap
ax6 = axes[1, 2]
top_countries = all_horses['Country'].value_counts().head(10).index
country_year = all_horses[all_horses['Country'].isin(top_countries)].groupby(['Country', 'Year']).size().unstack(fill_value=0)

if not country_year.empty:
    sns.heatmap(country_year, annot=True, fmt='d', cmap='YlOrRd', ax=ax6, cbar_kws={'shrink': 0.8})
ax6.set_title('Country Participation Heatmap', fontweight='bold', fontsize=12)
ax6.set_xlabel('Year', fontweight='bold')
ax6.set_ylabel('Country', fontweight='bold')

# Row 3, Subplot 7: Sire frequency
ax7 = axes[2, 0]
top_sires = all_horses['Sire'].dropna().value_counts().head(10)
if not top_sires.empty:
    bars = ax7.bar(range(len(top_sires)), top_sires.values, color='lightgreen', alpha=0.7)
    ax7.set_title('Top Sires by Frequency', fontweight='bold', fontsize=12)
    ax7.set_xlabel('Sire Rank', fontweight='bold')
    ax7.set_ylabel('Number of Offspring', fontweight='bold')
    ax7.set_xticks(range(len(top_sires)))
    ax7.set_xticklabels([f'#{i+1}' for i in range(len(top_sires))])

# Row 3, Subplot 8: Ownership patterns
ax8 = axes[2, 1]
owners = all_horses['Owner'].dropna().value_counts().head(8)
if not owners.empty:
    bars = ax8.barh(range(len(owners)), owners.values, color='lightcoral', alpha=0.7)
    ax8.set_title('Top Horse Owners', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Number of Horses', fontweight='bold')
    ax8.set_yticks(range(len(owners)))
    ax8.set_yticklabels([owner[:20] + '...' if len(owner) > 20 else owner for owner in owners.index], fontsize=9)

# Row 3, Subplot 9: Age vs Year correlation
ax9 = axes[2, 2]
# Create scatter plot of age vs year with competition type
for comp in ['Olympic', 'Paralympic']:
    data = all_horses[all_horses['Competition'] == comp]
    ax9.scatter(data['Year'], data['Age'], alpha=0.6, label=comp, s=30)

ax9.set_title('Age vs Competition Year', fontweight='bold', fontsize=12)
ax9.set_xlabel('Year', fontweight='bold')
ax9.set_ylabel('Horse Age', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Add trend line
years_all = all_horses['Year'].values
ages_all = all_horses['Age'].values
z = np.polyfit(years_all, ages_all, 1)
p = np.poly1d(z)
ax9.plot([2020, 2024], p([2020, 2024]), "r--", alpha=0.8, linewidth=2, label='Trend')

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.suptitle('Olympic & Paralympic Equestrian Evolution: Tokyo 2020 to Paris 2024', 
             fontsize=16, fontweight='bold', y=0.98)

# Save the plot
plt.savefig('equestrian_evolution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()