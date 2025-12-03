import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set up the plotting style
plt.style.use('seaborn-v0_8')
rcParams['figure.figsize'] = (12, 10)

# Load the datasets
df_2020 = pd.read_csv('2020_horses_olympic.csv')
df_2024 = pd.read_csv('2024_olympic_horses.csv')

# Clean and prepare data for analysis

# For 2020 data - rename column to match 2024
df_2020 = df_2020.rename(columns={'Athlete Country': 'Country'})

# Count horses per country for both years
country_counts_2020 = df_2020['Country'].value_counts().reset_index()
country_counts_2020.columns = ['Country', 'Count_2020']

country_counts_2024 = df_2024['Country'].value_counts().reset_index()
country_counts_2024.columns = ['Country', 'Count_2024']

# Merge the counts
country_comparison = pd.merge(country_counts_2020, country_counts_2024, on='Country', how='outer').fillna(0)

# Calculate change
country_comparison['Change'] = country_comparison['Count_2024'] - country_comparison['Count_2020']
country_comparison['Change_Percent'] = (country_comparison['Change'] / country_comparison['Count_2020']) * 100

# Top 10 countries by 2024 count for better visualization
top_countries_2024 = country_comparison.nlargest(10, 'Count_2024')['Country'].tolist()

# Filter for top countries
filtered_comparison = country_comparison[country_comparison['Country'].isin(top_countries_2024)]

# Prepare age distribution data
def calculate_age(year_of_birth):
    return 2024 - year_of_birth if year_of_birth > 0 else None

df_2024['Age'] = df_2024['Year of Birth'].apply(calculate_age)
df_2020['Age'] = df_2020['Year of Birth'].apply(calculate_age)

# Age distribution bins
age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']

# Create age distributions
age_dist_2020 = pd.cut(df_2020[df_2020['Age'].notnull()]['Age'], bins=age_bins, labels=age_labels, include_lowest=True).value_counts()
age_dist_2024 = pd.cut(df_2024[df_2024['Age'].notnull()]['Age'], bins=age_bins, labels=age_labels, include_lowest=True).value_counts()

# Align indices
all_ages = sorted(set(age_dist_2020.index) | set(age_dist_2024.index))
age_dist_2020 = age_dist_2020.reindex(all_ages, fill_value=0)
age_dist_2024 = age_dist_2024.reindex(all_ages, fill_value=0)

# Gender distribution
gender_dist_2020 = df_2020['Sex'].value_counts()
gender_dist_2024 = df_2024['Sex'].value_counts()

# Discipline participation
discipline_counts_2020 = df_2020['Discipline'].value_counts()
discipline_counts_2024 = df_2024['Discipline'].value_counts()

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Line chart with markers showing change in number of horses by country
filtered_comparison_sorted = filtered_comparison.sort_values('Change', ascending=False)
ax1.plot(range(len(filtered_comparison_sorted)), filtered_comparison_sorted['Count_2020'], marker='o', label='2020', linewidth=2)
ax1.plot(range(len(filtered_comparison_sorted)), filtered_comparison_sorted['Count_2024'], marker='s', label='2024', linewidth=2)
ax1.set_xticks(range(len(filtered_comparison_sorted)))
ax1.set_xticklabels(filtered_comparison_sorted['Country'], rotation=45, ha='right')
ax1.set_title('Change in Number of Horses by Country (2020 vs 2024)')
ax1.set_ylabel('Number of Horses')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Highlight countries with most significant changes
max_increase_idx = filtered_comparison_sorted['Change'].idxmax()
min_increase_idx = filtered_comparison_sorted['Change'].idxmin()

# Add annotations for top changes
ax1.annotate(f'+{int(filtered_comparison_sorted.loc[max_increase_idx]["Change"])}', 
             (max_increase_idx, filtered_comparison_sorted.loc[max_increase_idx]["Count_2024"]),
             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green')
ax1.annotate(f'{int(filtered_comparison_sorted.loc[min_increase_idx]["Change"])}', 
             (min_increase_idx, filtered_comparison_sorted.loc[min_increase_idx]["Count_2024"]),
             textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

# Plot 2: Stacked area chart showing age distribution evolution
age_df = pd.DataFrame({
    '2020': age_dist_2020,
    '2024': age_dist_2024
})

age_df.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
ax2.set_title('Age Distribution Evolution of Horses (2020 vs 2024)')
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Number of Horses')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Slope chart for gender distribution
gender_df = pd.DataFrame({
    '2020': gender_dist_2020,
    '2024': gender_dist_2024
})

# Normalize values to percentages for better comparison
gender_df_normalized = gender_df.div(gender_df.sum()) * 100

# Create slope chart
x_pos = np.arange(len(gender_df_normalized.index))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, gender_df_normalized['2020'], width, label='2020', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, gender_df_normalized['2024'], width, label='2024', alpha=0.8)

# Connect the bars with lines
for i in range(len(gender_df_normalized.index)):
    ax3.plot([x_pos[i] - width/2, x_pos[i] + width/2], 
             [gender_df_normalized.iloc[i]['2020'], gender_df_normalized.iloc[i]['2024']], 
             'k-', alpha=0.5)

ax3.set_title('Gender Distribution Evolution of Horses')
ax3.set_xlabel('Gender')
ax3.set_ylabel('Percentage (%)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(gender_df_normalized.index, rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Dual-axis chart for participation counts and percentage change by discipline
discipline_df = pd.DataFrame({
    '2020_Count': discipline_counts_2020,
    '2024_Count': discipline_counts_2024
})

# Calculate percentage change
discipline_df['Change_Percent'] = ((discipline_df['2024_Count'] - discipline_df['2020_Count']) / discipline_df['2020_Count']) * 100

# Sort by 2024 count for better visualization
discipline_df = discipline_df.sort_values('2024_Count', ascending=False)

# Create dual-axis chart
ax4_bar = ax4.twinx()
bars = ax4.bar(range(len(discipline_df)), discipline_df['2024_Count'], color='skyblue', alpha=0.7, label='2024 Count')
ax4.set_xticks(range(len(discipline_df)))
ax4.set_xticklabels(discipline_df.index, rotation=45, ha='right')
ax4.set_ylabel('Participation Count')
ax4.set_title('Equestrian Participation by Discipline (2020 vs 2024)')

# Add percentage change line
ax4_bar.plot(range(len(discipline_df)), discipline_df['Change_Percent'], marker='o', color='red', linewidth=2, label='Percentage Change')
ax4_bar.set_ylabel('Percentage Change (%)')
ax4_bar.grid(False)

# Add legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_bar.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()