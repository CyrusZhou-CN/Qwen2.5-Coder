import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df_full = pd.read_csv('gapminder_full.csv')
df_2007 = pd.read_csv('gapminder - gapminder.csv')

# Data preprocessing
df_2007 = df_2007.drop('Unnamed: 0', axis=1) if 'Unnamed: 0' in df_2007.columns else df_2007

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Define color palette for continents
continent_colors = {
    'Asia': '#FF6B6B',
    'Europe': '#4ECDC4', 
    'Africa': '#45B7D1',
    'Americas': '#96CEB4',
    'Oceania': '#FFEAA7'
}

# Row 1, Subplot 1: Violin plot with box plots and strip plot for life expectancy by continent
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Create violin plot
parts = ax1.violinplot([df_2007[df_2007['continent'] == cont]['life_exp'].values 
                       for cont in df_2007['continent'].unique()], 
                      positions=range(len(df_2007['continent'].unique())), 
                      widths=0.6, showmeans=False, showmedians=False)

# Color violin plots
for i, pc in enumerate(parts['bodies']):
    cont = list(df_2007['continent'].unique())[i]
    pc.set_facecolor(continent_colors[cont])
    pc.set_alpha(0.7)

# Add box plots
bp = ax1.boxplot([df_2007[df_2007['continent'] == cont]['life_exp'].values 
                  for cont in df_2007['continent'].unique()], 
                 positions=range(len(df_2007['continent'].unique())), 
                 widths=0.3, patch_artist=True)

for i, box in enumerate(bp['boxes']):
    cont = list(df_2007['continent'].unique())[i]
    box.set_facecolor(continent_colors[cont])
    box.set_alpha(0.9)

# Add strip plot
for i, cont in enumerate(df_2007['continent'].unique()):
    data = df_2007[df_2007['continent'] == cont]['life_exp'].values
    x = np.random.normal(i, 0.04, size=len(data))
    ax1.scatter(x, data, alpha=0.6, s=20, color=continent_colors[cont])

ax1.set_title('Life Expectancy Distribution by Continent\n(Violin + Box + Strip)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Continent')
ax1.set_ylabel('Life Expectancy (years)')
ax1.set_xticks(range(len(df_2007['continent'].unique())))
ax1.set_xticklabels(df_2007['continent'].unique(), rotation=45)
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Bubble chart with regression lines
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

for cont in df_2007['continent'].unique():
    cont_data = df_2007[df_2007['continent'] == cont]
    
    # Bubble chart
    scatter = ax2.scatter(cont_data['gdp_cap'], cont_data['life_exp'], 
                         s=cont_data['population']/200000, 
                         alpha=0.6, color=continent_colors[cont], 
                         label=cont, edgecolors='white', linewidth=0.5)
    
    # Regression line
    if len(cont_data) > 1:
        z = np.polyfit(cont_data['gdp_cap'], cont_data['life_exp'], 1)
        p = np.poly1d(z)
        x_reg = np.linspace(cont_data['gdp_cap'].min(), cont_data['gdp_cap'].max(), 100)
        ax2.plot(x_reg, p(x_reg), color=continent_colors[cont], linestyle='--', alpha=0.8, linewidth=2)

ax2.set_title('GDP vs Life Expectancy with Population Bubbles\n(Bubble Chart + Regression Lines)', fontweight='bold', fontsize=12)
ax2.set_xlabel('GDP per Capita (USD)')
ax2.set_ylabel('Life Expectancy (years)')
ax2.set_xscale('log')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Row 1, Subplot 3: Stacked bar chart with line overlay
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Calculate total population by continent
pop_by_cont = df_2007.groupby('continent')['population'].sum()
gdp_by_cont = df_2007.groupby('continent')['gdp_cap'].mean()

# Stacked bar chart
bars = ax3.bar(pop_by_cont.index, pop_by_cont.values/1e6, 
               color=[continent_colors[cont] for cont in pop_by_cont.index], 
               alpha=0.7, edgecolor='white', linewidth=1)

# Line plot overlay on secondary y-axis
ax3_twin = ax3.twinx()
line = ax3_twin.plot(gdp_by_cont.index, gdp_by_cont.values, 
                     color='darkred', marker='o', linewidth=3, markersize=8, 
                     label='Avg GDP per Capita')

ax3.set_title('Population by Continent with GDP Overlay\n(Stacked Bar + Line Plot)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Continent')
ax3.set_ylabel('Total Population (millions)', color='black')
ax3_twin.set_ylabel('Average GDP per Capita (USD)', color='darkred')
ax3.tick_params(axis='x', rotation=45)
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Row 2, Subplot 4: Multi-line time series with confidence intervals
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

for cont in df_full['continent'].unique():
    cont_data = df_full[df_full['continent'] == cont]
    yearly_stats = cont_data.groupby('year')['life_exp'].agg(['mean', 'std']).reset_index()
    
    # Main line
    ax4.plot(yearly_stats['year'], yearly_stats['mean'], 
             color=continent_colors[cont], linewidth=3, label=cont, marker='o', markersize=4)
    
    # Confidence interval
    ax4.fill_between(yearly_stats['year'], 
                     yearly_stats['mean'] - yearly_stats['std'], 
                     yearly_stats['mean'] + yearly_stats['std'], 
                     color=continent_colors[cont], alpha=0.2)

ax4.set_title('Life Expectancy Evolution by Continent\n(Time Series + Confidence Intervals)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year')
ax4.set_ylabel('Life Expectancy (years)')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Area chart with line overlays
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Cumulative population by continent over time
cont_pop_time = df_full.groupby(['year', 'continent'])['population'].sum().unstack(fill_value=0)
cont_pop_time_cumsum = cont_pop_time.cumsum(axis=1)

# Area chart
bottom = np.zeros(len(cont_pop_time_cumsum))
for i, cont in enumerate(cont_pop_time.columns):
    if i == 0:
        ax5.fill_between(cont_pop_time_cumsum.index, 0, cont_pop_time_cumsum[cont]/1e6, 
                        color=continent_colors[cont], alpha=0.7, label=cont)
        bottom = cont_pop_time_cumsum[cont]/1e6
    else:
        ax5.fill_between(cont_pop_time_cumsum.index, bottom, 
                        bottom + cont_pop_time[cont]/1e6, 
                        color=continent_colors[cont], alpha=0.7, label=cont)
        bottom += cont_pop_time[cont]/1e6

# Top 3 most populous countries trajectories
top_countries = df_full.groupby('country')['population'].max().nlargest(3).index
for country in top_countries:
    country_data = df_full[df_full['country'] == country]
    ax5.plot(country_data['year'], country_data['population']/1e6, 
             color='black', linestyle='--', linewidth=2, alpha=0.8)

ax5.set_title('Cumulative Population Growth by Continent\n(Area Chart + Top Country Trajectories)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Year')
ax5.set_ylabel('Population (millions)')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Slope chart with scatter plot
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Calculate 1952 and 2007 GDP by continent
gdp_1952 = df_full[df_full['year'] == 1952].groupby('continent')['gdp_cap'].mean()
gdp_2007 = df_full[df_full['year'] == 2007].groupby('continent')['gdp_cap'].mean()
growth_rate = ((gdp_2007 - gdp_1952) / gdp_1952) * 100

# Slope chart
for cont in gdp_1952.index:
    ax6.plot([0, 1], [gdp_1952[cont], gdp_2007[cont]], 
             color=continent_colors[cont], linewidth=3, marker='o', markersize=8, 
             label=cont, alpha=0.8)

# Scatter plot overlay showing initial GDP vs growth rate
ax6_twin = ax6.twinx()
scatter = ax6_twin.scatter(gdp_1952.values, growth_rate.values, 
                          c=[continent_colors[cont] for cont in gdp_1952.index], 
                          s=200, alpha=0.7, edgecolors='black', linewidth=2)

ax6.set_title('GDP Evolution 1952-2007 by Continent\n(Slope Chart + Growth Rate Scatter)', fontweight='bold', fontsize=12)
ax6.set_xticks([0, 1])
ax6.set_xticklabels(['1952', '2007'])
ax6.set_ylabel('GDP per Capita (USD)')
ax6_twin.set_ylabel('Growth Rate (%)', color='gray')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.grid(True, alpha=0.3)

# Row 3, Subplot 7: Dendrogram with correlation heatmap
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Prepare data for clustering
cluster_data = df_2007[['life_exp', 'gdp_cap', 'population']].copy()
cluster_data['gdp_cap'] = np.log(cluster_data['gdp_cap'])
cluster_data['population'] = np.log(cluster_data['population'])

# Standardize data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Hierarchical clustering
linkage_matrix = linkage(cluster_data_scaled, method='ward')

# Create dendrogram
dendrogram(linkage_matrix, ax=ax7, labels=df_2007['country'].values, 
           leaf_rotation=90, leaf_font_size=6, color_threshold=0.7*max(linkage_matrix[:,2]))

ax7.set_title('Country Clustering Dendrogram\n(Hierarchical Clustering)', fontweight='bold', fontsize=12)
ax7.set_xlabel('Countries')
ax7.set_ylabel('Distance')

# Row 3, Subplot 8: Parallel coordinates with density curves
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Normalize data for parallel coordinates
features = ['life_exp', 'gdp_cap', 'population']
normalized_data = df_2007[features].copy()
normalized_data['gdp_cap'] = np.log(normalized_data['gdp_cap'])
normalized_data['population'] = np.log(normalized_data['population'])

for col in normalized_data.columns:
    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())

# Parallel coordinates plot
for i, cont in enumerate(df_2007['continent'].unique()):
    cont_data = normalized_data[df_2007['continent'] == cont]
    for idx, row in cont_data.iterrows():
        ax8.plot(range(len(features)), row.values, 
                color=continent_colors[cont], alpha=0.6, linewidth=1)

# Add density curves for each dimension
for i, feature in enumerate(features):
    ax8_twin = ax8.twinx()
    ax8_twin.hist(normalized_data[feature], bins=20, alpha=0.3, 
                 color='gray', orientation='horizontal' if i % 2 == 0 else 'vertical')

ax8.set_title('Multidimensional Development Patterns\n(Parallel Coordinates + Density)', fontweight='bold', fontsize=12)
ax8.set_xticks(range(len(features)))
ax8.set_xticklabels(['Life Exp', 'GDP (log)', 'Pop (log)'])
ax8.set_ylabel('Normalized Values')
ax8.grid(True, alpha=0.3)

# Row 3, Subplot 9: Treemap simulation with network overlay
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Create a simplified treemap using scatter plot with sized circles
# Size by population, color by life expectancy
scatter = ax9.scatter(df_2007['gdp_cap'], df_2007['life_exp'], 
                     s=df_2007['population']/100000, 
                     c=df_2007['life_exp'], cmap='viridis', 
                     alpha=0.7, edgecolors='white', linewidth=1)

# Add network connections between similar countries
from scipy.spatial.distance import pdist, squareform
features_for_network = df_2007[['life_exp', 'gdp_cap']].values
distances = squareform(pdist(features_for_network, metric='euclidean'))

# Connect countries with similar development profiles
threshold = np.percentile(distances, 10)  # Connect closest 10%
for i in range(len(df_2007)):
    for j in range(i+1, len(df_2007)):
        if distances[i, j] < threshold:
            ax9.plot([df_2007.iloc[i]['gdp_cap'], df_2007.iloc[j]['gdp_cap']], 
                    [df_2007.iloc[i]['life_exp'], df_2007.iloc[j]['life_exp']], 
                    'gray', alpha=0.3, linewidth=0.5)

ax9.set_title('Country Development Network\n(Treemap Simulation + Network Graph)', fontweight='bold', fontsize=12)
ax9.set_xlabel('GDP per Capita (USD)')
ax9.set_ylabel('Life Expectancy (years)')
ax9.set_xscale('log')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax9)
cbar.set_label('Life Expectancy (years)')
ax9.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.show()