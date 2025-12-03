import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
olympic_2024 = pd.read_csv('2024_olympic_horses.csv')
paralympic_2024 = pd.read_csv('2024_paralympic_horses.csv')
olympic_2020 = pd.read_csv('2020_horses_olympic.csv')
paralympic_2020 = pd.read_csv('2020_horses_paralympic.csv')

# Data preprocessing - simplified and optimized
def prepare_data():
    # Standardize column names and add year/type identifiers
    olympic_2024_clean = olympic_2024[['Country', 'Year of Birth', 'Sex', 'Country of Birth']].copy()
    olympic_2024_clean['Year'] = 2024
    olympic_2024_clean['Type'] = 'Olympic'
    olympic_2024_clean['Age'] = 2024 - olympic_2024_clean['Year of Birth']
    
    paralympic_2024_clean = paralympic_2024[['Country', 'Year of Birth', 'Sex', 'Country of Birth']].copy()
    paralympic_2024_clean['Year'] = 2024
    paralympic_2024_clean['Type'] = 'Paralympic'
    paralympic_2024_clean['Age'] = 2024 - paralympic_2024_clean['Year of Birth']
    
    # Rename columns for 2020 data to match
    olympic_2020_clean = olympic_2020[['Athlete Country', 'Year of Birth', 'Sex', 'Country of Birth']].copy()
    olympic_2020_clean = olympic_2020_clean.rename(columns={'Athlete Country': 'Country'})
    olympic_2020_clean['Year'] = 2020
    olympic_2020_clean['Type'] = 'Olympic'
    olympic_2020_clean['Age'] = 2020 - olympic_2020_clean['Year of Birth']
    
    paralympic_2020_clean = paralympic_2020[['Athlete Country', 'Year of Birth', 'Sex']].copy()
    paralympic_2020_clean = paralympic_2020_clean.rename(columns={'Athlete Country': 'Country'})
    paralympic_2020_clean['Year'] = 2020
    paralympic_2020_clean['Type'] = 'Paralympic'
    paralympic_2020_clean['Age'] = 2020 - paralympic_2020_clean['Year of Birth']
    paralympic_2020_clean['Country of Birth'] = 'Unknown'  # Add missing column
    
    # Combine all data
    all_data = pd.concat([
        olympic_2024_clean,
        paralympic_2024_clean,
        olympic_2020_clean,
        paralympic_2020_clean
    ], ignore_index=True)
    
    # Clean data
    all_data = all_data.dropna(subset=['Country', 'Year of Birth', 'Sex'])
    all_data = all_data[all_data['Age'] > 0]  # Remove invalid ages
    
    return all_data

all_data = prepare_data()

# Create the 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.patch.set_facecolor('white')

# Subplot 1: Horse age distribution with grouped bars
ax1 = axes[0, 0]
ax1.set_facecolor('white')

# Create age bins
all_data['Age_Bin'] = pd.cut(all_data['Age'], bins=[0, 10, 15, 20, 25, 30], labels=['0-10', '11-15', '16-20', '21-25', '26-30'])

# Group data for bar chart
age_counts = all_data.groupby(['Age_Bin', 'Type']).size().unstack(fill_value=0)
x_pos = np.arange(len(age_counts.index))
width = 0.35

# Create grouped bar chart
if 'Olympic' in age_counts.columns:
    bars1 = ax1.bar(x_pos - width/2, age_counts['Olympic'], width, label='Olympic', color='#2E86AB', alpha=0.8)
if 'Paralympic' in age_counts.columns:
    bars2 = ax1.bar(x_pos + width/2, age_counts['Paralympic'], width, label='Paralympic', color='#A23B72', alpha=0.8)

ax1.set_xlabel('Age Groups', fontweight='bold', fontsize=10)
ax1.set_ylabel('Number of Horses', fontweight='bold', fontsize=10)
ax1.set_title('Horse Age Distribution by Competition Type', fontweight='bold', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(age_counts.index, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Bubble chart - simplified
ax2 = axes[0, 1]
ax2.set_facecolor('white')

# Calculate percentages by country and type - simplified
country_stats = []
top_countries = all_data['Country'].value_counts().head(10).index  # Limit to top 10

for country in top_countries:
    for comp_type in ['Olympic', 'Paralympic']:
        subset = all_data[(all_data['Country'] == country) & (all_data['Type'] == comp_type)]
        if len(subset) > 2:  # Only include countries with sufficient data
            total_horses = len(subset)
            male_count = len(subset[subset['Sex'].isin(['Stallion', 'Gelding'])])
            female_count = len(subset[subset['Sex'] == 'Mare'])
            
            if total_horses > 0:
                male_pct = (male_count / total_horses) * 100
                female_pct = (female_count / total_horses) * 100
                country_stats.append({
                    'Country': country,
                    'Type': comp_type,
                    'Male_Pct': male_pct,
                    'Female_Pct': female_pct,
                    'Total_Horses': total_horses
                })

if country_stats:
    country_df = pd.DataFrame(country_stats)
    
    # Create bubble chart
    olympic_data = country_df[country_df['Type'] == 'Olympic']
    paralympic_data = country_df[country_df['Type'] == 'Paralympic']
    
    if len(olympic_data) > 0:
        ax2.scatter(olympic_data['Male_Pct'], olympic_data['Female_Pct'], 
                   s=olympic_data['Total_Horses']*10, alpha=0.6, 
                   color='#2E86AB', label='Olympic', edgecolors='black', linewidth=0.5)
    
    if len(paralympic_data) > 0:
        ax2.scatter(paralympic_data['Male_Pct'], paralympic_data['Female_Pct'], 
                   s=paralympic_data['Total_Horses']*10, alpha=0.6, 
                   color='#A23B72', label='Paralympic', edgecolors='black', linewidth=0.5)

ax2.set_xlabel('Percentage of Male Horses', fontweight='bold', fontsize=10)
ax2.set_ylabel('Percentage of Female Horses', fontweight='bold', fontsize=10)
ax2.set_title('Gender Distribution by Country', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Slope chart - simplified
ax3 = axes[1, 0]
ax3.set_facecolor('white')

# Get top 8 countries by total horses
top_countries = all_data['Country'].value_counts().head(8).index

# Calculate participation rates
slope_data = []
for country in top_countries:
    for year in [2020, 2024]:
        subset = all_data[(all_data['Country'] == country) & (all_data['Year'] == year)]
        if len(subset) > 0:
            participation_rate = len(subset)
            slope_data.append({
                'Country': country,
                'Year': year,
                'Participation': participation_rate
            })

if slope_data:
    slope_df = pd.DataFrame(slope_data)
    
    # Create slope chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    for i, country in enumerate(top_countries):
        country_data = slope_df[slope_df['Country'] == country]
        if len(country_data) == 2:
            years = country_data['Year'].values
            participation = country_data['Participation'].values
            ax3.plot(years, participation, 'o-', alpha=0.7, linewidth=2, 
                    markersize=6, color=colors[i], label=country)

ax3.set_xlabel('Year', fontweight='bold', fontsize=10)
ax3.set_ylabel('Number of Horses', fontweight='bold', fontsize=10)
ax3.set_title('Country Participation Trends (2020-2024)', fontweight='bold', fontsize=12)
ax3.set_xticks([2020, 2024])
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Stacked area chart - simplified
ax4 = axes[1, 1]
ax4.set_facecolor('white')

# Calculate sex distribution by year
sex_by_year = all_data.groupby(['Year', 'Sex']).size().unstack(fill_value=0)
years = [2020, 2024]

# Get counts for each sex
stallion_counts = [sex_by_year.loc[2020].get('Stallion', 0), sex_by_year.loc[2024].get('Stallion', 0)]
gelding_counts = [sex_by_year.loc[2020].get('Gelding', 0), sex_by_year.loc[2024].get('Gelding', 0)]
mare_counts = [sex_by_year.loc[2020].get('Mare', 0), sex_by_year.loc[2024].get('Mare', 0)]

# Create stacked area chart
ax4.fill_between(years, 0, stallion_counts, alpha=0.7, color='#FF6B6B', label='Stallion')
ax4.fill_between(years, stallion_counts, [s+g for s,g in zip(stallion_counts, gelding_counts)], 
                alpha=0.7, color='#4ECDC4', label='Gelding')
ax4.fill_between(years, [s+g for s,g in zip(stallion_counts, gelding_counts)], 
                [s+g+m for s,g,m in zip(stallion_counts, gelding_counts, mare_counts)], 
                alpha=0.7, color='#45B7D1', label='Mare')

ax4.set_xlabel('Year', fontweight='bold', fontsize=10)
ax4.set_ylabel('Number of Horses', fontweight='bold', fontsize=10)
ax4.set_title('Horse Sex Distribution Evolution', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Breeding relationships scatter plot
ax5 = axes[2, 0]
ax5.set_facecolor('white')

# Calculate breeding relationships - simplified
birth_countries = all_data['Country of Birth'].dropna()
birth_countries = birth_countries[birth_countries != 'Unknown']

if len(birth_countries) > 0:
    breeding_data = []
    for birth_country in birth_countries.unique():
        horses_bred = len(all_data[all_data['Country of Birth'] == birth_country])
        competing_countries = all_data[all_data['Country of Birth'] == birth_country]['Country'].nunique()
        if horses_bred > 2:  # Only include significant breeding countries
            breeding_data.append({
                'Birth_Country': birth_country,
                'Horses_Bred': horses_bred,
                'Competing_Countries': competing_countries
            })
    
    if breeding_data:
        breeding_df = pd.DataFrame(breeding_data)
        
        # Create scatter plot
        scatter = ax5.scatter(breeding_df['Horses_Bred'], breeding_df['Competing_Countries'], 
                             s=100, alpha=0.7, c=range(len(breeding_df)), cmap='viridis')
        
        # Add labels for major breeding countries
        for i, row in breeding_df.iterrows():
            if row['Horses_Bred'] > 10:
                ax5.annotate(row['Birth_Country'], (row['Horses_Bred'], row['Competing_Countries']), 
                           fontsize=8, alpha=0.8)

ax5.set_xlabel('Number of Horses Bred', fontweight='bold', fontsize=10)
ax5.set_ylabel('Number of Competing Countries', fontweight='bold', fontsize=10)
ax5.set_title('Breeding Network Analysis', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot 6: Treemap visualization - simplified
ax6 = axes[2, 1]
ax6.set_facecolor('white')

# Create simplified treemap using rectangles
country_type_counts = all_data.groupby(['Country', 'Type']).size().reset_index(name='Count')
top_combinations = country_type_counts.nlargest(10, 'Count')

# Create treemap layout
x, y = 0, 0
colors = plt.cm.Set3(np.linspace(0, 1, len(top_combinations)))

for i, (_, row) in enumerate(top_combinations.iterrows()):
    width = max(0.5, row['Count'] / top_combinations['Count'].sum() * 8)
    height = 1.5
    
    rect = Rectangle((x, y), width, height, facecolor=colors[i], alpha=0.7, edgecolor='black')
    ax6.add_patch(rect)
    
    # Add text if rectangle is large enough
    if width > 1:
        ax6.text(x + width/2, y + height/2, f"{row['Country'][:3]}\n{row['Type'][:4]}", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    x += width
    if x > 7:
        x = 0
        y += 2

ax6.set_xlim(-0.5, 8)
ax6.set_ylim(-0.5, 6)
ax6.set_title('Horse Distribution by Country and Competition Type', fontweight='bold', fontsize=12)
ax6.set_xticks([])
ax6.set_yticks([])

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Save the plot
plt.savefig('equestrian_analysis.png', dpi=300, bbox_inches='tight')
plt.show()