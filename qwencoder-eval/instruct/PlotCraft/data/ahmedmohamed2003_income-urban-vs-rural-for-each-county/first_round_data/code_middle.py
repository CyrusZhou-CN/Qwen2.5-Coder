import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('Income_Urban_VS_Rural.csv')

# Calculate national average median household income
national_avg = df['Median Household Income'].mean()

# Calculate deviation from national average for each county
df['Income_Deviation'] = df['Median Household Income'] - national_avg

# Separate urban and rural counties
urban_counties = df[df['Urban-Rural'] == 'Urban'].copy()
rural_counties = df[df['Urban-Rural'] == 'Rural'].copy()

# Sort by deviation magnitude within each group
urban_counties = urban_counties.sort_values('Income_Deviation', ascending=False)
rural_counties = rural_counties.sort_values('Income_Deviation', ascending=False)

# Instead of plotting all counties, select representative samples to avoid timeout
# Get top 50 and bottom 50 from each group for visualization
n_sample = 50
urban_top = urban_counties.head(n_sample)
urban_bottom = urban_counties.tail(n_sample)
rural_top = rural_counties.head(n_sample)
rural_bottom = rural_counties.tail(n_sample)

# Combine samples for plotting
urban_sample = pd.concat([urban_top, urban_bottom]).sort_values('Income_Deviation', ascending=False)
rural_sample = pd.concat([rural_top, rural_bottom]).sort_values('Income_Deviation', ascending=False)

# Create figure
plt.figure(figsize=(14, 10))

# Create positions for bars
urban_positions = np.arange(len(urban_sample))
rural_positions = -np.arange(1, len(rural_sample) + 1)

# Plot urban counties (above center line)
plt.barh(urban_positions, urban_sample['Income_Deviation'], 
         color='steelblue', alpha=0.7, label='Urban Counties', height=0.8)

# Plot rural counties (below center line)
plt.barh(rural_positions, rural_sample['Income_Deviation'], 
         color='forestgreen', alpha=0.7, label='Rural Counties', height=0.8)

# Add reference line at zero deviation
plt.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

# Get top 5 and bottom 5 counties for labeling
urban_top5 = urban_counties.head(5)
urban_bottom5 = urban_counties.tail(5)
rural_top5 = rural_counties.head(5)
rural_bottom5 = rural_counties.tail(5)

# Add county labels for top 5 and bottom 5 in each category
# Urban top 5 (highest positive deviations)
for i in range(5):
    row = urban_top5.iloc[i]
    pos = urban_positions[i]
    deviation = row['Income_Deviation']
    county_name = f"{row['County']}, {row['State']}"
    plt.text(deviation + 2000, pos, county_name, 
             va='center', ha='left', fontsize=9, fontweight='bold')

# Urban bottom 5 (lowest deviations)
for i in range(5):
    row = urban_bottom5.iloc[i]
    pos = urban_positions[len(urban_sample) - 5 + i]
    deviation = row['Income_Deviation']
    county_name = f"{row['County']}, {row['State']}"
    if deviation < 0:
        plt.text(deviation - 2000, pos, county_name, 
                 va='center', ha='right', fontsize=9, fontweight='bold')
    else:
        plt.text(deviation + 2000, pos, county_name, 
                 va='center', ha='left', fontsize=9, fontweight='bold')

# Rural top 5 (highest positive deviations)
for i in range(5):
    row = rural_top5.iloc[i]
    pos = rural_positions[i]
    deviation = row['Income_Deviation']
    county_name = f"{row['County']}, {row['State']}"
    plt.text(deviation + 2000, pos, county_name, 
             va='center', ha='left', fontsize=9, fontweight='bold')

# Rural bottom 5 (lowest deviations)
for i in range(5):
    row = rural_bottom5.iloc[i]
    pos = rural_positions[len(rural_sample) - 5 + i]
    deviation = row['Income_Deviation']
    county_name = f"{row['County']}, {row['State']}"
    plt.text(deviation - 2000, pos, county_name, 
             va='center', ha='right', fontsize=9, fontweight='bold')

# Styling and labels
plt.title('Income Disparities: County Deviations from National Average\nUrban vs Rural Counties (Sample)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Deviation from National Average Median Household Income ($)', 
           fontsize=12, fontweight='bold')
plt.ylabel('Counties (Urban Above, Rural Below)', 
           fontsize=12, fontweight='bold')

# Format x-axis to show currency
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add legend
plt.legend(loc='upper right', fontsize=11)

# Add text box with statistics
urban_count = len(urban_counties)
rural_count = len(rural_counties)
textstr = f'National Average: ${national_avg:,.0f}\nUrban Counties: {urban_count}\nRural Counties: {rural_count}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Remove y-axis ticks since they're not meaningful
plt.yticks([])

# Add subtle grid for x-axis only
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Set limits to better show the data
x_min = min(df['Income_Deviation'].min(), -50000)
x_max = max(df['Income_Deviation'].max(), 100000)
plt.xlim(x_min, x_max)

# Layout and save
plt.tight_layout()
plt.savefig('income_disparities_urban_rural.png', dpi=300, bbox_inches='tight')
plt.show()