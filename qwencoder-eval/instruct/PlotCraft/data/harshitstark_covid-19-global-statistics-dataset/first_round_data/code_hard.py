import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('COVID-19 Global Statistics Dataset.csv')

# Function to clean numeric columns
def clean_numeric(series):
    if series.dtype == 'object':
        return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace('+', ''), errors='coerce')
    return series

# Clean all numeric columns
numeric_cols = ['Total Cases', 'New Cases', 'Total Deaths', 'Total Recovered', 'New Recovered', 
                'Active Cases', 'Serious, Critical', 'Tot Cases/1M pop', 'Deaths/1M pop', 
                'Total Tests', 'Tests/1M pop', 'Population']

for col in numeric_cols:
    df[col] = clean_numeric(df[col])

# Remove rows with missing critical data
df_clean = df.dropna(subset=['Tot Cases/1M pop', 'Deaths/1M pop', 'Population']).copy()

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Define consistent color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590', '#F8961E', '#90A955', '#264653']

# Subplot 1: Line chart with scatter overlay and error bars
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Sort by total cases per million and take top 20
top_countries = df_clean.nlargest(20, 'Tot Cases/1M pop').copy()
x_pos = range(len(top_countries))

# Line plot for total cases per million
line1 = ax1.plot(x_pos, top_countries['Tot Cases/1M pop'], 'o-', color=colors[0], linewidth=2, markersize=6, label='Total Cases/1M')

# Scatter overlay for new cases (scaled)
new_cases_scaled = top_countries['New Cases'].fillna(0) / 1000
scatter1 = ax1.scatter(x_pos, top_countries['Tot Cases/1M pop'], s=new_cases_scaled*50, 
                      color=colors[1], alpha=0.6, label='New Cases (size)', edgecolors='white')

# Error bars for active cases range
active_cases = top_countries['Active Cases'].fillna(0) / top_countries['Population'] * 1000000
ax1.errorbar(x_pos, top_countries['Tot Cases/1M pop'], yerr=active_cases*0.1, 
            fmt='none', color=colors[2], alpha=0.7, capsize=3, label='Active Cases Range')

ax1.set_title('COVID-19 Case Progression Patterns by Country', fontweight='bold', fontsize=12, pad=15)
ax1.set_xlabel('Countries (Ranked by Cases/1M)', fontweight='bold')
ax1.set_ylabel('Total Cases per Million', fontweight='bold')
ax1.set_xticks(x_pos[::2])
ax1.set_xticklabels(top_countries['Country'].iloc[::2], rotation=45, ha='right')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot 2: Dual-axis bar and line plot
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

top_deaths = df_clean.nlargest(15, 'Deaths/1M pop').copy()
x_pos2 = range(len(top_deaths))

# Bar chart for deaths per million
bars = ax2.bar(x_pos2, top_deaths['Deaths/1M pop'], color=colors[3], alpha=0.7, label='Deaths/1M pop')

# Secondary y-axis for death-to-case ratio
ax2_twin = ax2.twinx()
death_ratio = (top_deaths['Deaths/1M pop'] / top_deaths['Tot Cases/1M pop'] * 100).fillna(0)
line2 = ax2_twin.plot(x_pos2, death_ratio, 'o-', color=colors[4], linewidth=3, markersize=8, label='Death Rate %')

ax2.set_title('Death Rates and Case Fatality Analysis', fontweight='bold', fontsize=12, pad=15)
ax2.set_xlabel('Countries (Ranked by Deaths/1M)', fontweight='bold')
ax2.set_ylabel('Deaths per Million', fontweight='bold', color=colors[3])
ax2_twin.set_ylabel('Death-to-Case Ratio (%)', fontweight='bold', color=colors[4])
ax2.set_xticks(x_pos2[::2])
ax2.set_xticklabels(top_deaths['Country'].iloc[::2], rotation=45, ha='right')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Subplot 3: Stacked area chart with line overlay
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Select countries with complete recovery data
complete_data = df_clean.dropna(subset=['Total Recovered']).nlargest(10, 'Tot Cases/1M pop').copy()
x_pos3 = range(len(complete_data))

# Calculate proportions
total_cases = complete_data['Total Cases']
deaths = complete_data['Total Deaths'].fillna(0)
recovered = complete_data['Total Recovered'].fillna(0)
active = complete_data['Active Cases'].fillna(0)

# Stacked area chart
ax3.fill_between(x_pos3, 0, recovered/total_cases*100, color=colors[4], alpha=0.7, label='Recovered %')
ax3.fill_between(x_pos3, recovered/total_cases*100, (recovered+deaths)/total_cases*100, 
                color=colors[3], alpha=0.7, label='Deaths %')
ax3.fill_between(x_pos3, (recovered+deaths)/total_cases*100, 100, 
                color=colors[1], alpha=0.7, label='Active %')

# Recovery rate line overlay
recovery_rate = (recovered / total_cases * 100).fillna(0)
ax3.plot(x_pos3, recovery_rate, 'o-', color='white', linewidth=3, markersize=8, 
         markeredgecolor=colors[0], markeredgewidth=2, label='Recovery Rate')

ax3.set_title('Case Composition and Recovery Patterns', fontweight='bold', fontsize=12, pad=15)
ax3.set_xlabel('Countries (Top 10 by Cases/1M)', fontweight='bold')
ax3.set_ylabel('Percentage of Total Cases', fontweight='bold')
ax3.set_xticks(x_pos3)
ax3.set_xticklabels(complete_data['Country'], rotation=45, ha='right')
ax3.legend(loc='center right', fontsize=9)
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3)

# Subplot 4: Bubble scatter with regression
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# Filter data for scatter plot
scatter_data = df_clean.dropna(subset=['Tests/1M pop', 'Tot Cases/1M pop', 'Population'])
scatter_data = scatter_data[scatter_data['Tests/1M pop'] > 0]

x_tests = scatter_data['Tests/1M pop']
y_cases = scatter_data['Tot Cases/1M pop']
sizes = np.sqrt(scatter_data['Population']) / 1000

# Bubble scatter plot
scatter4 = ax4.scatter(x_tests, y_cases, s=sizes, c=colors[0], alpha=0.6, edgecolors='white')

# Regression line with confidence interval
if len(x_tests) > 1:
    z = np.polyfit(x_tests, y_cases, 1)
    p = np.poly1d(z)
    x_reg = np.linspace(x_tests.min(), x_tests.max(), 100)
    ax4.plot(x_reg, p(x_reg), color=colors[3], linewidth=2, label=f'Regression Line')

ax4.set_title('Testing vs Cases Relationship (Bubble = Population)', fontweight='bold', fontsize=12, pad=15)
ax4.set_xlabel('Tests per Million Population', fontweight='bold')
ax4.set_ylabel('Cases per Million Population', fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Subplot 5: Violin plot with box plot overlay
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Create population density bins
pop_density_data = df_clean.dropna(subset=['Population', 'Tot Cases/1M pop']).copy()
pop_density_data['Pop_Density_Bin'] = pd.cut(pop_density_data['Population'], 
                                            bins=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Prepare data for violin plot
violin_data = []
labels = []
for bin_name in ['Low', 'Medium', 'High', 'Very High']:
    bin_data = pop_density_data[pop_density_data['Pop_Density_Bin'] == bin_name]['Tot Cases/1M pop']
    if len(bin_data) > 0:
        violin_data.append(bin_data)
        labels.append(bin_name)

# Violin plot
if violin_data:
    parts = ax5.violinplot(violin_data, positions=range(len(labels)), showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[5])
        pc.set_alpha(0.7)
    
    # Box plot overlay
    bp = ax5.boxplot(violin_data, positions=range(len(labels)), widths=0.3, 
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(colors[1])
        patch.set_alpha(0.8)

ax5.set_title('Cases Distribution by Population Categories', fontweight='bold', fontsize=12, pad=15)
ax5.set_xlabel('Population Category', fontweight='bold')
ax5.set_ylabel('Cases per Million Population', fontweight='bold')
ax5.set_xticks(range(len(labels)))
ax5.set_xticklabels(labels)
ax5.grid(True, alpha=0.3)

# Subplot 6: Correlation heatmap
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Select metrics for correlation
corr_data = df_clean[['Tests/1M pop', 'Tot Cases/1M pop', 'Deaths/1M pop']].dropna()
corr_data['Recovery Rate'] = (df_clean['Total Recovered'] / df_clean['Total Cases'] * 100).fillna(0)

# Calculate correlation matrix
corr_matrix = corr_data.corr()

# Create heatmap
im = ax6.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                       ha="center", va="center", color="black", fontweight='bold')

ax6.set_title('Correlation Matrix: Key COVID-19 Metrics', fontweight='bold', fontsize=12, pad=15)
ax6.set_xticks(range(len(corr_matrix.columns)))
ax6.set_yticks(range(len(corr_matrix.columns)))
ax6.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax6.set_yticklabels(corr_matrix.columns)

# Add colorbar
cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# Subplot 7: Diverging bar chart with lollipop overlay
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Calculate median death rate
median_death_rate = df_clean['Deaths/1M pop'].median()
div_data = df_clean.nlargest(20, 'Deaths/1M pop').copy()
div_data['Above_Median'] = div_data['Deaths/1M pop'] > median_death_rate
div_data['Deviation'] = div_data['Deaths/1M pop'] - median_death_rate

# Diverging bars
colors_div = [colors[3] if x else colors[4] for x in div_data['Above_Median']]
bars7 = ax7.barh(range(len(div_data)), div_data['Deviation'], color=colors_div, alpha=0.7)

# Lollipop overlay for serious cases ratio
serious_ratio = (div_data['Serious, Critical'].fillna(0) / div_data['Total Cases'] * 100000).fillna(0)
ax7.scatter(div_data['Deviation'], range(len(div_data)), s=serious_ratio*10, 
           color='white', edgecolors=colors[0], linewidth=2, alpha=0.8, label='Critical Cases Ratio')

ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax7.set_title('Death Rates vs Median with Critical Cases', fontweight='bold', fontsize=12, pad=15)
ax7.set_xlabel('Deviation from Median Deaths/1M', fontweight='bold')
ax7.set_ylabel('Countries', fontweight='bold')
ax7.set_yticks(range(len(div_data)))
ax7.set_yticklabels(div_data['Country'], fontsize=8)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='x')

# Subplot 8: Slope chart
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Top 15 countries by total cases
slope_data = df_clean.nlargest(15, 'Total Cases').copy()
y_pos = range(len(slope_data))

# Normalize data for slope chart
cases_norm = slope_data['Total Cases'] / slope_data['Total Cases'].max() * 100
deaths_norm = slope_data['Total Deaths'] / slope_data['Total Deaths'].max() * 100

# Slope lines
for i in range(len(slope_data)):
    ax8.plot([0, 1], [cases_norm.iloc[i], deaths_norm.iloc[i]], 
            color=colors[i % len(colors)], alpha=0.7, linewidth=2)
    
    # Critical cases markers
    critical_ratio = (slope_data['Serious, Critical'].iloc[i] / slope_data['Total Cases'].iloc[i] * 1000)
    if not pd.isna(critical_ratio):
        ax8.scatter([0.5], [(cases_norm.iloc[i] + deaths_norm.iloc[i])/2], 
                   s=critical_ratio*50, color=colors[1], alpha=0.8, edgecolors='white')

ax8.set_title('Cases to Deaths Progression (Top 15 Countries)', fontweight='bold', fontsize=12, pad=15)
ax8.set_xlim(-0.1, 1.1)
ax8.set_xticks([0, 1])
ax8.set_xticklabels(['Total Cases\n(Normalized)', 'Total Deaths\n(Normalized)'], fontweight='bold')
ax8.set_ylabel('Normalized Scale (0-100)', fontweight='bold')
ax8.grid(True, alpha=0.3)

# Add country labels
for i in range(0, len(slope_data), 3):
    ax8.text(-0.05, cases_norm.iloc[i], slope_data['Country'].iloc[i], 
            ha='right', va='center', fontsize=8)

# Subplot 9: Radar chart
ax9 = plt.subplot(3, 3, 9, projection='polar')
ax9.set_facecolor('white')

# Select top 6 countries for radar chart
radar_data = df_clean.nlargest(6, 'Tot Cases/1M pop').copy()

# Metrics for radar chart
metrics = ['Tot Cases/1M pop', 'Deaths/1M pop', 'Tests/1M pop']
radar_data_clean = radar_data[metrics].fillna(0)

# Normalize data (0-1 scale)
for col in metrics:
    radar_data_clean[col] = radar_data_clean[col] / radar_data_clean[col].max()

# Add recovery rate
recovery_rate_norm = (radar_data['Total Recovered'] / radar_data['Total Cases']).fillna(0)
radar_data_clean['Recovery Rate'] = recovery_rate_norm

# Angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(radar_data_clean.columns), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot each country
for i, (idx, country_data) in enumerate(radar_data_clean.iterrows()):
    values = country_data.tolist()
    values += values[:1]  # Complete the circle
    
    ax9.plot(angles, values, 'o-', linewidth=2, label=radar_data.loc[idx, 'Country'], 
            color=colors[i % len(colors)])
    ax9.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(['Cases/1M', 'Deaths/1M', 'Tests/1M', 'Recovery Rate'], fontweight='bold')
ax9.set_ylim(0, 1)
ax9.set_title('Multi-Metric Country Comparison\n(Normalized Scale)', fontweight='bold', fontsize=12, pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()