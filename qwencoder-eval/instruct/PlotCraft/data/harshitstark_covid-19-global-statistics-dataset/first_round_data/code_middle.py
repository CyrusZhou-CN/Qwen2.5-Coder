import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('COVID-19 Global Statistics Dataset.csv')

# Data preprocessing
# Clean numeric columns that are stored as objects
def clean_numeric(x):
    if pd.isna(x) or x == '':
        return np.nan
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

# Clean the relevant columns
df['Deaths/1M pop'] = df['Deaths/1M pop'].apply(clean_numeric)
df['Tot Cases/1M pop'] = df['Tot Cases/1M pop'].apply(clean_numeric)
df['Population'] = df['Population'].apply(clean_numeric)

# Remove rows with missing data for key metrics
df_clean = df.dropna(subset=['Deaths/1M pop', 'Tot Cases/1M pop', 'Population']).copy()

# Calculate global median death rate
global_median_death_rate = df_clean['Deaths/1M pop'].median()

# Calculate deviation from median
df_clean['Death_Rate_Deviation'] = df_clean['Deaths/1M pop'] - global_median_death_rate

# Sort by deviation for better visualization
df_clean = df_clean.sort_values('Death_Rate_Deviation')

# Assign regions based on country names (simplified mapping)
def assign_region(country):
    europe = ['Germany', 'France', 'Italy', 'UK', 'Spain', 'Poland', 'Netherlands', 'Belgium', 'Greece', 'Portugal', 'Czech Republic', 'Hungary', 'Sweden', 'Austria', 'Belarus', 'Switzerland', 'Bulgaria', 'Serbia', 'Denmark', 'Finland', 'Slovakia', 'Norway', 'Ireland', 'Croatia', 'Bosnia and Herzegovina', 'Albania', 'Lithuania', 'Slovenia', 'Latvia', 'Estonia', 'Moldova', 'Luxembourg', 'Malta', 'Iceland', 'Andorra', 'Monaco', 'Liechtenstein', 'San Marino', 'Vatican City', 'Russia', 'Ukraine', 'Romania']
    asia = ['India', 'S. Korea', 'Japan', 'China', 'Indonesia', 'Pakistan', 'Bangladesh', 'Philippines', 'Vietnam', 'Turkey', 'Iran', 'Iraq', 'Afghanistan', 'Saudi Arabia', 'Uzbekistan', 'Malaysia', 'Nepal', 'Yemen', 'North Korea', 'Sri Lanka', 'Kazakhstan', 'Myanmar', 'Azerbaijan', 'Jordan', 'UAE', 'Tajikistan', 'Israel', 'Laos', 'Singapore', 'Oman', 'Kuwait', 'Georgia', 'Mongolia', 'Armenia', 'Qatar', 'Bahrain', 'East Timor', 'Palestine', 'Cyprus', 'Bhutan', 'Maldives', 'Brunei']
    north_america = ['USA', 'Canada', 'Mexico', 'Guatemala', 'Cuba', 'Haiti', 'Dominican Republic', 'Honduras', 'Nicaragua', 'Costa Rica', 'Panama', 'Jamaica', 'Trinidad and Tobago', 'Bahamas', 'Belize', 'Barbados', 'Saint Lucia', 'Grenada', 'Saint Vincent and the Grenadines', 'Antigua and Barbuda', 'Dominica', 'Saint Kitts and Nevis']
    south_america = ['Brazil', 'Argentina', 'Colombia', 'Peru', 'Venezuela', 'Chile', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Guyana', 'Suriname']
    africa = ['Nigeria', 'Ethiopia', 'Egypt', 'South Africa', 'Kenya', 'Uganda', 'Algeria', 'Sudan', 'Morocco', 'Angola', 'Ghana', 'Mozambique', 'Madagascar', 'Cameroon', 'Ivory Coast', 'Niger', 'Burkina Faso', 'Mali', 'Malawi', 'Zambia', 'Somalia', 'Senegal', 'Chad', 'Zimbabwe', 'Guinea', 'Rwanda', 'Benin', 'Tunisia', 'Burundi', 'Togo', 'Sierra Leone', 'Libya', 'Liberia', 'Central African Republic', 'Mauritania', 'Eritrea', 'Gambia', 'Botswana', 'Namibia', 'Gabon', 'Lesotho', 'Guinea-Bissau', 'Equatorial Guinea', 'Mauritius', 'Eswatini', 'Djibouti', 'Comoros', 'Cape Verde', 'Sao Tome and Principe', 'Seychelles']
    oceania = ['Australia', 'Papua New Guinea', 'New Zealand', 'Fiji', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Micronesia', 'Tonga', 'Kiribati', 'Palau', 'Marshall Islands', 'Tuvalu', 'Nauru']
    
    if country in europe:
        return 'Europe'
    elif country in asia:
        return 'Asia'
    elif country in north_america:
        return 'North America'
    elif country in south_america:
        return 'South America'
    elif country in africa:
        return 'Africa'
    elif country in oceania:
        return 'Oceania'
    else:
        return 'Other'

df_clean['Region'] = df_clean['Country'].apply(assign_region)

# Get top 5 and bottom 5 countries by deviation
top_5 = df_clean.nlargest(5, 'Death_Rate_Deviation')
bottom_5 = df_clean.nsmallest(5, 'Death_Rate_Deviation')

# Create the visualization
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Color palette for regions
region_colors = {
    'Europe': '#2E86AB',
    'Asia': '#A23B72',
    'North America': '#F18F01',
    'South America': '#C73E1D',
    'Africa': '#592E83',
    'Oceania': '#048A81',
    'Other': '#6C757D'
}

# Create diverging bar chart
y_positions = np.arange(len(df_clean))
colors = [region_colors[region] for region in df_clean['Region']]

bars = ax.barh(y_positions, df_clean['Death_Rate_Deviation'], 
               color=colors, alpha=0.7, height=0.8)

# Add vertical line at zero (median)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

# Create scatter plot overlay on secondary y-axis
ax2 = ax.twinx()

# Normalize population for point sizes (scale between 20 and 200)
pop_normalized = ((df_clean['Population'] - df_clean['Population'].min()) / 
                  (df_clean['Population'].max() - df_clean['Population'].min()) * 180 + 20)

# Create scatter plot
for region in region_colors.keys():
    region_data = df_clean[df_clean['Region'] == region]
    if len(region_data) > 0:
        region_y_pos = np.arange(len(df_clean))[df_clean['Region'] == region]
        region_pop_norm = pop_normalized[df_clean['Region'] == region]
        
        ax2.scatter(region_data['Tot Cases/1M pop'], region_y_pos,
                   s=region_pop_norm, c=region_colors[region], 
                   alpha=0.6, edgecolors='white', linewidth=0.5,
                   label=region)

# Highlight top 5 and bottom 5 countries
for idx, row in top_5.iterrows():
    y_pos = df_clean.index.get_loc(idx)
    ax.annotate(f"{row['Country']}", 
                xy=(row['Death_Rate_Deviation'], y_pos),
                xytext=(10, 0), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                ha='left', va='center')

for idx, row in bottom_5.iterrows():
    y_pos = df_clean.index.get_loc(idx)
    ax.annotate(f"{row['Country']}", 
                xy=(row['Death_Rate_Deviation'], y_pos),
                xytext=(-10, 0), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                ha='right', va='center')

# Styling and labels
ax.set_xlabel('Deviation from Global Median Death Rate (Deaths/1M pop)', fontsize=12, fontweight='bold')
ax.set_ylabel('Countries (sorted by deviation)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Total Cases per Million Population', fontsize=12, fontweight='bold')
ax2.set_ylabel('Countries (Cases/1M scale)', fontsize=12, fontweight='bold')

plt.suptitle('COVID-19 Mortality Deviations and Case-Death Relationships by Country', 
             fontsize=16, fontweight='bold', y=0.98)

# Add subtitle with median information
ax.text(0.5, 0.95, f'Global Median Death Rate: {global_median_death_rate:.0f} deaths per million population',
        transform=ax.transAxes, ha='center', fontsize=11, style='italic')

# Remove y-axis ticks for cleaner look
ax.set_yticks([])
ax2.set_yticks([])

# Add legend for regions
ax2.legend(title='Region', bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

# Add grid for better readability
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

# Set background color to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Layout adjustment
plt.tight_layout()
plt.subplots_adjust(right=0.85)

plt.show()