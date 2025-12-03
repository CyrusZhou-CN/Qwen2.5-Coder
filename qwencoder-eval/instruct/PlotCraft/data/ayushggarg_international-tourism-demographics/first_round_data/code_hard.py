import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
expenditures = pd.read_csv('API_ST.INT.XPND.CD_DS2_en_csv_v2_1929314.csv')
arrivals = pd.read_csv('API_ST.INT.ARVL_DS2_en_csv_v2_1927083.csv')
departures = pd.read_csv('API_ST.INT.DPRT_DS2_en_csv_v2_1929304.csv')

# Data preprocessing - simplified and optimized
years = [str(year) for year in range(1995, 2019)]

def process_data_fast(df, metric_name):
    # Filter out aggregated regions quickly
    exclude_patterns = ['World', 'income', 'Arab World', 'Euro area', 'European Union', 'OECD', 'Small states']
    mask = ~df['Country Name'].str.contains('|'.join(exclude_patterns), na=False)
    df_clean = df[mask].copy()
    
    # Select only needed columns
    cols = ['Country Name', 'Country Code'] + years
    df_clean = df_clean[cols]
    
    # Melt data
    df_melted = df_clean.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=years,
        var_name='Year',
        value_name=metric_name
    )
    df_melted['Year'] = df_melted['Year'].astype(int)
    df_melted[metric_name] = pd.to_numeric(df_melted[metric_name], errors='coerce')
    
    return df_melted

# Process datasets
exp_data = process_data_fast(expenditures, 'Expenditures')
arr_data = process_data_fast(arrivals, 'Arrivals')
dep_data = process_data_fast(departures, 'Departures')

# Merge datasets efficiently
tourism_data = exp_data.merge(arr_data, on=['Country Name', 'Country Code', 'Year'], how='outer')
tourism_data = tourism_data.merge(dep_data, on=['Country Name', 'Country Code', 'Year'], how='outer')

# Get top countries for analysis
top_arrivals_2018 = tourism_data[tourism_data['Year'] == 2018].nlargest(5, 'Arrivals')['Country Name'].dropna().tolist()[:5]
top_expenditures_2018 = tourism_data[tourism_data['Year'] == 2018].nlargest(3, 'Expenditures')['Country Name'].dropna().tolist()[:3]

# Create figure
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palettes
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
colors_area = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Subplot 1: Line chart with confidence bands
ax1 = plt.subplot(3, 3, 1)
for i, country in enumerate(top_arrivals_2018):
    country_data = tourism_data[tourism_data['Country Name'] == country].sort_values('Year')
    if len(country_data) > 5:
        x = country_data['Year']
        y = country_data['Arrivals'].fillna(0) / 1e6
        
        if y.sum() > 0:
            ax1.plot(x, y, color=colors[i], linewidth=2.5, label=country[:15], alpha=0.8)
            
            # Simple confidence band using rolling std
            if len(y) > 3:
                y_smooth = y.rolling(window=3, center=True).mean().fillna(y)
                y_std = y.rolling(window=3, center=True).std().fillna(0)
                ax1.fill_between(x, y_smooth - y_std, y_smooth + y_std, 
                               alpha=0.2, color=colors[i])

ax1.set_title('Tourism Arrivals with Confidence Bands\nTop 5 Countries by 2018', fontsize=12, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Arrivals (Millions)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot 2: Dual-axis plot
ax2 = plt.subplot(3, 3, 2)
ax2_twin = ax2.twinx()

# Stacked area for expenditures
exp_matrix = []
years_range = list(range(1995, 2019))

for country in top_expenditures_2018:
    country_exp = []
    country_data = tourism_data[tourism_data['Country Name'] == country]
    for year in years_range:
        val = country_data[country_data['Year'] == year]['Expenditures'].values
        country_exp.append(val[0]/1e9 if len(val) > 0 and not pd.isna(val[0]) else 0)
    exp_matrix.append(country_exp)

if exp_matrix:
    ax2.stackplot(years_range, *exp_matrix, labels=[c[:15] for c in top_expenditures_2018], 
                 colors=colors_area, alpha=0.7)

# Line plot for departures
for i, country in enumerate(top_expenditures_2018):
    country_data = tourism_data[tourism_data['Country Name'] == country].sort_values('Year')
    if len(country_data) > 0:
        y = country_data['Departures'].fillna(0) / 1e6
        if y.sum() > 0:
            ax2_twin.plot(country_data['Year'], y, color=colors_area[i], 
                         linewidth=2, linestyle='--', alpha=0.8)

ax2.set_title('Expenditures (Stacked) vs Departures (Lines)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Expenditures (Billions USD)')
ax2_twin.set_ylabel('Departures (Millions)')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Correlation heatmap
ax3 = plt.subplot(3, 3, 3)

# Create correlation matrix for top countries
top_countries = tourism_data['Country Name'].value_counts().head(10).index.tolist()
corr_matrix = np.zeros((len(top_countries), len(years_range)))

for i, country in enumerate(top_countries):
    country_data = tourism_data[tourism_data['Country Name'] == country]
    for j, year in enumerate(years_range):
        year_data = country_data[country_data['Year'] == year]
        if len(year_data) > 0:
            arr = year_data['Arrivals'].values[0] if len(year_data['Arrivals'].values) > 0 else 0
            exp = year_data['Expenditures'].values[0] if len(year_data['Expenditures'].values) > 0 else 0
            if not pd.isna(arr) and not pd.isna(exp) and arr > 0 and exp > 0:
                corr_matrix[i, j] = np.log10(exp/arr) if arr > 0 else 0

im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
ax3.set_title('Arrivals-Expenditures Correlation Heatmap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Year')
ax3.set_ylabel('Countries')
ax3.set_xticks(range(0, len(years_range), 5))
ax3.set_xticklabels([str(y) for y in years_range[::5]])
ax3.set_yticks(range(len(top_countries)))
ax3.set_yticklabels([c[:10] for c in top_countries], fontsize=8)

# Subplot 4: Regional clusters with recession periods
ax4 = plt.subplot(3, 3, 4)

# Simple clustering based on expenditure levels
countries_sample = tourism_data['Country Name'].value_counts().head(20).index.tolist()
expenditure_data = []

for country in countries_sample:
    country_data = tourism_data[tourism_data['Country Name'] == country]
    total_exp = country_data['Expenditures'].sum()
    expenditure_data.append([total_exp])

# Cluster countries
scaler = StandardScaler()
exp_scaled = scaler.fit_transform(expenditure_data)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
clusters = kmeans.fit_predict(exp_scaled)

cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

for cluster_id in range(6):
    cluster_countries = [countries_sample[i] for i in range(len(countries_sample)) if clusters[i] == cluster_id]
    
    cluster_means = []
    for year in years_range:
        year_values = []
        for country in cluster_countries:
            val = tourism_data[(tourism_data['Country Name'] == country) & 
                              (tourism_data['Year'] == year)]['Expenditures'].values
            if len(val) > 0 and not pd.isna(val[0]):
                year_values.append(val[0]/1e9)
        
        cluster_means.append(np.mean(year_values) if year_values else 0)
    
    ax4.plot(years_range, cluster_means, color=cluster_colors[cluster_id], 
            linewidth=2, label=f'Cluster {cluster_id+1}', alpha=0.8)

# Add recession periods
ax4.axvspan(2001, 2001.5, alpha=0.3, color='gray', label='Recession 2001')
ax4.axvspan(2008, 2009, alpha=0.3, color='red', label='Recession 2008-09')

ax4.set_title('Regional Expenditure Clusters', fontsize=12, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Expenditures (Billions USD)')
ax4.legend(fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3)

# Subplot 5: Slope chart
ax5 = plt.subplot(3, 3, 5)

data_1995 = tourism_data[tourism_data['Year'] == 1995].dropna(subset=['Arrivals', 'Departures'])
data_2018 = tourism_data[tourism_data['Year'] == 2018].dropna(subset=['Arrivals', 'Departures'])

comparison_data = data_1995.merge(data_2018, on='Country Name', suffixes=('_1995', '_2018'))
comparison_data = comparison_data.head(15)

for _, row in comparison_data.iterrows():
    exp_size_1995 = row['Expenditures_1995']/1e8 if not pd.isna(row['Expenditures_1995']) else 10
    exp_size_2018 = row['Expenditures_2018']/1e8 if not pd.isna(row['Expenditures_2018']) else 10
    
    ax5.scatter(1995, row['Arrivals_1995']/1e6, s=max(exp_size_1995, 10), 
               alpha=0.6, color='#FF6B6B', edgecolor='white')
    ax5.scatter(2018, row['Arrivals_2018']/1e6, s=max(exp_size_2018, 10), 
               alpha=0.6, color='#4ECDC4', edgecolor='white')
    
    ax5.plot([1995, 2018], [row['Arrivals_1995']/1e6, row['Arrivals_2018']/1e6], 
            color='gray', alpha=0.5, linewidth=1)

ax5.set_title('Tourism Evolution: 1995 vs 2018', fontsize=12, fontweight='bold')
ax5.set_xlabel('Year')
ax5.set_ylabel('Arrivals (Millions)')
ax5.set_xticks([1995, 2018])
ax5.grid(True, alpha=0.3)

# Subplot 6: Time series decomposition
ax6 = plt.subplot(3, 3, 6)

top_economies = tourism_data.groupby('Country Name')['Expenditures'].sum().nlargest(3).index.tolist()

for i, country in enumerate(top_economies):
    country_data = tourism_data[tourism_data['Country Name'] == country].sort_values('Year')
    if len(country_data) > 5:
        expenditures = country_data['Expenditures'].fillna(0) / 1e9
        
        # Original data
        ax6.plot(country_data['Year'], expenditures, 
                color=colors[i], alpha=0.4, linewidth=1, label=f'{country[:15]} (Original)')
        
        # Trend line (simple moving average)
        if len(expenditures) > 3:
            trend = expenditures.rolling(window=3, center=True).mean()
            ax6.plot(country_data['Year'], trend, 
                    color=colors[i], linewidth=3, label=f'{country[:15]} (Trend)')

ax6.set_title('Expenditure Decomposition', fontsize=12, fontweight='bold')
ax6.set_xlabel('Year')
ax6.set_ylabel('Expenditures (Billions USD)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Subplot 7: Bubble chart with trend lines
ax7 = plt.subplot(3, 3, 7)

for i, country in enumerate(top_arrivals_2018):
    country_data = tourism_data[tourism_data['Country Name'] == country].sort_values('Year')
    if len(country_data) > 5:
        exp_per_arrival = (country_data['Expenditures'] / country_data['Arrivals']).fillna(0)
        departures_size = country_data['Departures'].fillna(0) / 1e5
        
        scatter = ax7.scatter(country_data['Year'], exp_per_arrival, 
                             s=np.maximum(departures_size, 10), alpha=0.6, 
                             color=colors[i], edgecolor='white',
                             label=country[:15])
        
        # Simple trend line
        if len(exp_per_arrival) > 3:
            valid_mask = exp_per_arrival > 0
            if valid_mask.sum() > 2:
                z = np.polyfit(country_data['Year'][valid_mask], exp_per_arrival[valid_mask], 1)
                p = np.poly1d(z)
                ax7.plot(country_data['Year'], p(country_data['Year']), 
                        color=colors[i], linewidth=2, alpha=0.8, linestyle='--')

ax7.set_title('Expenditure per Arrival Evolution', fontsize=12, fontweight='bold')
ax7.set_xlabel('Year')
ax7.set_ylabel('Expenditure per Arrival (USD)')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Subplot 8: Multi-panel time series
ax8 = plt.subplot(3, 3, 8)

for i, country in enumerate(top_arrivals_2018[:3]):
    country_data = tourism_data[tourism_data['Country Name'] == country].sort_values('Year')
    if len(country_data) > 5:
        arrivals = country_data['Arrivals'].fillna(0) / 1e6
        ax8.plot(country_data['Year'], arrivals, 
                color=colors[i], linewidth=2, label=f'{country[:15]}', alpha=0.8)

ax8.set_title('Tourism Arrivals Trends', fontsize=12, fontweight='bold')
ax8.set_xlabel('Year')
ax8.set_ylabel('Arrivals (Millions)')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# Subplot 9: Radar chart
ax9 = plt.subplot(3, 3, 9, projection='polar')

# Simplified radar chart
metrics = ['Arrivals Growth', 'Expenditure Growth', 'Departure Growth']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

periods = [(1995, 2005), (2006, 2010), (2011, 2018)]
period_names = ['1995-2005', '2006-2010', '2011-2018']
colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for period_idx, (start_year, end_year) in enumerate(periods):
    # Calculate average growth for top countries
    growth_values = []
    
    for metric in ['Arrivals', 'Expenditures', 'Departures']:
        total_growth = 0
        count = 0
        
        for country in top_arrivals_2018[:3]:
            country_data = tourism_data[tourism_data['Country Name'] == country]
            start_data = country_data[country_data['Year'] == start_year]
            end_data = country_data[country_data['Year'] == end_year]
            
            if len(start_data) > 0 and len(end_data) > 0:
                start_val = start_data[metric].iloc[0]
                end_val = end_data[metric].iloc[0]
                
                if not pd.isna(start_val) and not pd.isna(end_val) and start_val > 0:
                    growth = ((end_val / start_val) - 1) * 100
                    total_growth += growth
                    count += 1
        
        avg_growth = total_growth / count if count > 0 else 0
        growth_values.append(max(avg_growth, 0))  # Ensure non-negative for radar chart
    
    growth_values += growth_values[:1]  # Complete the circle
    
    ax9.plot(angles, growth_values, 'o-', linewidth=2, 
            color=colors_radar[period_idx], alpha=0.7,
            label=period_names[period_idx])
    ax9.fill(angles, growth_values, alpha=0.1, color=colors_radar[period_idx])

ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(metrics, fontsize=9)
ax9.set_title('Tourism Growth by Period', fontsize=12, fontweight='bold', pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)
ax9.grid(True, alpha=0.3)

# Layout adjustments
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.savefig('tourism_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()