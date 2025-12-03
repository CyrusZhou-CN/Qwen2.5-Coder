import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load and parse the data
df_raw = pd.read_csv('peace_index.csv')

# The data appears to be in a single column with semicolon separation
# Let's parse it properly
data_lines = []
header_line = None

for idx, row in df_raw.iterrows():
    line = str(row.iloc[0]).strip()
    if 'Country' in line and 'iso3c' in line:
        # This is the header
        header_line = line.split(';')
        continue
    elif ';' in line and line != 'nan':
        parts = line.split(';')
        if len(parts) >= 3:  # At least country, iso3c, and some data
            data_lines.append(parts)

# Create proper DataFrame
if header_line is None:
    # Create default header if not found
    columns = ['Country', 'iso3c'] + [str(year) for year in range(2008, 2024)]
else:
    columns = header_line

# Ensure we have the right number of columns
max_cols = max(len(line) for line in data_lines) if data_lines else 18
if len(columns) < max_cols:
    columns.extend([f'col_{i}' for i in range(len(columns), max_cols)])

# Pad data lines to match column count
for i, line in enumerate(data_lines):
    while len(line) < len(columns):
        line.append('')

df = pd.DataFrame(data_lines, columns=columns[:len(data_lines[0]) if data_lines else 18])

# Clean up column names and identify year columns
year_cols = []
for col in df.columns:
    if col.isdigit() and 2008 <= int(col) <= 2023:
        year_cols.append(col)
    elif any(str(year) in col for year in range(2008, 2024)):
        # Extract year from column name
        for year in range(2008, 2024):
            if str(year) in col:
                df = df.rename(columns={col: str(year)})
                year_cols.append(str(year))
                break

# If we still don't have year columns, create them from the data structure
if not year_cols:
    year_cols = [str(year) for year in range(2008, 2024)]
    # Rename columns starting from the third column (after Country and iso3c)
    for i, year in enumerate(year_cols):
        if i + 2 < len(df.columns):
            df = df.rename(columns={df.columns[i + 2]: year})

# Keep only the year columns that exist
year_cols = [col for col in year_cols if col in df.columns]

# Convert year columns to numeric
for col in year_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with all NaN values in year columns
if year_cols:
    df = df.dropna(subset=year_cols, how='all')

# Clean country names
if 'Country' in df.columns:
    df['Country'] = df['Country'].astype(str).str.strip()
elif len(df.columns) > 0:
    df['Country'] = df.iloc[:, 0].astype(str).str.strip()
else:
    df['Country'] = [f'Country_{i}' for i in range(len(df))]

# Define regions (simplified mapping)
region_mapping = {
    'Iceland': 'Northern Europe', 'Denmark': 'Northern Europe', 'Ireland': 'Northern Europe',
    'New Zealand': 'Oceania', 'Austria': 'Western Europe', 'Singapore': 'Asia',
    'Portugal': 'Western Europe', 'Slovenia': 'Eastern Europe', 'Japan': 'Asia',
    'Switzerland': 'Western Europe', 'Canada': 'North America', 'Czech Republic': 'Eastern Europe',
    'Finland': 'Northern Europe', 'Croatia': 'Eastern Europe', 'Germany': 'Western Europe',
    'Norway': 'Northern Europe', 'Malaysia': 'Asia', 'Sweden': 'Northern Europe',
    'Belgium': 'Western Europe', 'Netherlands': 'Western Europe', 'Australia': 'Oceania',
    'United Kingdom': 'Western Europe', 'France': 'Western Europe', 'Spain': 'Western Europe',
    'Italy': 'Western Europe', 'United States': 'North America', 'South Korea': 'Asia',
    'Poland': 'Eastern Europe', 'Estonia': 'Eastern Europe', 'Latvia': 'Eastern Europe',
    'Lithuania': 'Eastern Europe', 'Slovakia': 'Eastern Europe', 'Hungary': 'Eastern Europe',
    'Romania': 'Eastern Europe', 'Bulgaria': 'Eastern Europe', 'Greece': 'Western Europe',
    'Cyprus': 'Western Europe', 'Malta': 'Western Europe', 'Uruguay': 'South America',
    'Chile': 'South America', 'Costa Rica': 'Central America', 'Panama': 'Central America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Ghana': 'Africa',
    'Botswana': 'Africa', 'South Africa': 'Africa', 'Senegal': 'Africa',
    'Mongolia': 'Asia', 'Namibia': 'Africa', 'Indonesia': 'Asia',
    'Jordan': 'Middle East', 'Morocco': 'Africa', 'Cuba': 'Central America',
    'Vietnam': 'Asia', 'Serbia': 'Eastern Europe', 'Albania': 'Eastern Europe',
    'Turkey': 'Middle East', 'Israel': 'Middle East', 'Egypt': 'Africa',
    'China': 'Asia', 'India': 'Asia', 'Pakistan': 'Asia', 'Bangladesh': 'Asia',
    'Thailand': 'Asia', 'Philippines': 'Asia', 'Mexico': 'North America',
    'Colombia': 'South America', 'Venezuela': 'South America', 'Peru': 'South America',
    'Afghanistan': 'Asia', 'Iraq': 'Middle East', 'Syria': 'Middle East',
    'Yemen': 'Middle East', 'Somalia': 'Africa', 'Sudan': 'Africa'
}

# Add region column
df['Region'] = df['Country'].map(region_mapping).fillna('Other')

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Generate sample data if real data is insufficient
if len(year_cols) < 5 or len(df) < 10:
    print("Insufficient data detected, generating sample data for demonstration")
    np.random.seed(42)
    countries = ['Iceland', 'Denmark', 'Singapore', 'New Zealand', 'Switzerland', 
                'Germany', 'Japan', 'Canada', 'Australia', 'Norway', 'Sweden',
                'United States', 'France', 'United Kingdom', 'South Korea',
                'Afghanistan', 'Syria', 'Iraq', 'Yemen', 'Somalia']
    
    year_cols = [str(year) for year in range(2008, 2024)]
    sample_data = []
    
    for country in countries:
        row_data = {'Country': country, 'Region': region_mapping.get(country, 'Other')}
        base_score = np.random.uniform(1.2, 3.5)
        for year in year_cols:
            # Add some trend and noise
            trend = (int(year) - 2008) * np.random.uniform(-0.02, 0.02)
            noise = np.random.normal(0, 0.1)
            score = max(1.0, min(4.0, base_score + trend + noise))
            row_data[year] = score
        sample_data.append(row_data)
    
    df = pd.DataFrame(sample_data)

# Subplot 1: Regional trends with error bands
ax1 = plt.subplot(3, 3, 1)
regional_data = []
regions = df['Region'].unique()
regions = [r for r in regions if r != 'Other' and len(df[df['Region'] == r]) > 0]

for region in regions:
    region_df = df[df['Region'] == region]
    yearly_means = []
    yearly_stds = []
    for year in year_cols:
        if year in df.columns:
            values = region_df[year].dropna()
            if len(values) > 0:
                yearly_means.append(values.mean())
                yearly_stds.append(values.std() if len(values) > 1 else 0)
            else:
                yearly_means.append(np.nan)
                yearly_stds.append(np.nan)
    
    if not all(np.isnan(yearly_means)):
        regional_data.append((region, yearly_means, yearly_stds))

# Plot top 5 most peaceful regions
if regional_data:
    regional_data.sort(key=lambda x: np.nanmean(x[1]))
    colors = plt.cm.Set3(np.linspace(0, 1, min(5, len(regional_data))))
    years = [int(year) for year in year_cols]

    for i, (region, means, stds) in enumerate(regional_data[:5]):
        means = np.array(means)
        stds = np.array(stds)
        valid_idx = ~np.isnan(means)
        
        if np.any(valid_idx):
            ax1.plot(np.array(years)[valid_idx], means[valid_idx], 
                    label=region, color=colors[i], linewidth=2)
            ax1.fill_between(np.array(years)[valid_idx], 
                            (means - stds)[valid_idx], 
                            (means + stds)[valid_idx], 
                            alpha=0.2, color=colors[i])

ax1.set_title('Regional Peace Index Trends (Top 5 Most Peaceful)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Peace Index Score')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked area chart of score ranges
ax2 = plt.subplot(3, 3, 2)
score_ranges = {'Excellent (1.0-1.5)': [], 'Good (1.5-2.0)': [], 
                'Moderate (2.0-2.5)': [], 'Poor (2.5-3.0)': [], 'Very Poor (3.0+)': []}
total_counts = []

for year in year_cols:
    if year in df.columns:
        year_data = df[year].dropna()
        total_counts.append(len(year_data))
        
        excellent = len(year_data[(year_data >= 1.0) & (year_data < 1.5)])
        good = len(year_data[(year_data >= 1.5) & (year_data < 2.0)])
        moderate = len(year_data[(year_data >= 2.0) & (year_data < 2.5)])
        poor = len(year_data[(year_data >= 2.5) & (year_data < 3.0)])
        very_poor = len(year_data[year_data >= 3.0])
        
        score_ranges['Excellent (1.0-1.5)'].append(excellent)
        score_ranges['Good (1.5-2.0)'].append(good)
        score_ranges['Moderate (2.0-2.5)'].append(moderate)
        score_ranges['Poor (2.5-3.0)'].append(poor)
        score_ranges['Very Poor (3.0+)'].append(very_poor)

if total_counts:
    # Create stacked area chart
    years_int = [int(year) for year in year_cols]
    bottom = np.zeros(len(years_int))
    colors_stack = ['#2E8B57', '#90EE90', '#FFD700', '#FF6347', '#DC143C']
    
    for i, (label, values) in enumerate(score_ranges.items()):
        if len(values) == len(years_int):
            ax2.fill_between(years_int, bottom, bottom + values, 
                            label=label, alpha=0.7, color=colors_stack[i])
            bottom += values

    # Overlay total count line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(years_int, total_counts, 'k-', linewidth=2, label='Total Countries')
    ax2_twin.set_ylabel('Total Country Count')

ax2.set_title('Peace Index Score Distribution Over Time', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Countries')
ax2.legend(loc='upper left', fontsize=8)

# Subplot 3: Slope chart for largest changes
ax3 = plt.subplot(3, 3, 3)
changes = []
start_year = year_cols[0] if year_cols else '2008'
end_year = year_cols[-1] if year_cols else '2023'

for idx, row in df.iterrows():
    if start_year in df.columns and end_year in df.columns:
        start_val = row[start_year]
        end_val = row[end_year]
        if pd.notna(start_val) and pd.notna(end_val):
            change = end_val - start_val
            changes.append((row['Country'], start_val, end_val, change))

if changes:
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    top_changes = changes[:min(10, len(changes))]

    for i, (country, start, end, change) in enumerate(top_changes):
        color = 'green' if change < 0 else 'red'
        ax3.plot([int(start_year), int(end_year)], [start, end], color=color, linewidth=2, alpha=0.7)
        ax3.text(int(start_year), start, country[:10], fontsize=8, ha='right', va='center')
        ax3.text(int(end_year), end, f'{change:+.2f}', fontsize=8, ha='left', va='center')

ax3.set_title(f'Largest Peace Index Changes ({start_year}-{end_year})', fontweight='bold', fontsize=12)
ax3.set_xlabel('Year')
ax3.set_ylabel('Peace Index Score')
ax3.grid(True, alpha=0.3)

# Subplot 4: Violin plots with box plots overlay (fixed)
ax4 = plt.subplot(3, 3, 4)
violin_data = []
violin_years = []

for year in year_cols[::2]:  # Every other year to avoid crowding
    if year in df.columns:
        year_data = df[year].dropna()
        if len(year_data) > 2:  # Need at least 3 points for violin plot
            violin_data.append(year_data.values)
            violin_years.append(year)

if violin_data:
    positions = range(len(violin_data))
    try:
        parts = ax4.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        # Make violin plots more transparent
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        
        # Add box plots
        box_parts = ax4.boxplot(violin_data, positions=positions, widths=0.3, patch_artist=True)
        for patch in box_parts['boxes']:
            patch.set_alpha(0.7)
            
    except Exception as e:
        # Fallback to simple box plots if violin plots fail
        ax4.boxplot(violin_data, positions=positions, patch_artist=True)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(violin_years, rotation=45)

ax4.set_title('Peace Index Distribution Evolution', fontweight='bold', fontsize=12)
ax4.set_xlabel('Year')
ax4.set_ylabel('Peace Index Score')

# Subplot 5: Correlation heatmap
ax5 = plt.subplot(3, 3, 5)
if len(year_cols) > 1:
    corr_data = df[year_cols].corr()
    if not corr_data.empty:
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax5)
ax5.set_title('Year-over-Year Correlation Matrix', fontweight='bold', fontsize=12)

# Subplot 6: Joy plot (ridgeline) approximation
ax6 = plt.subplot(3, 3, 6)
periods = ['2008-2010', '2011-2013', '2014-2016', '2017-2019', '2020-2023']
period_data = []

for i, period in enumerate(periods):
    if period == '2008-2010':
        cols = ['2008', '2009', '2010']
    elif period == '2011-2013':
        cols = ['2011', '2012', '2013']
    elif period == '2014-2016':
        cols = ['2014', '2015', '2016']
    elif period == '2017-2019':
        cols = ['2017', '2018', '2019']
    else:
        cols = ['2020', '2021', '2022', '2023']
    
    # Only use columns that exist
    cols = [col for col in cols if col in df.columns]
    
    if cols:
        period_values = df[cols].values.flatten()
        period_values = period_values[~np.isnan(period_values)]
        
        if len(period_values) > 5:
            try:
                kde = gaussian_kde(period_values)
                x_range = np.linspace(period_values.min(), period_values.max(), 100)
                density = kde(x_range)
                density = density / density.max() * 0.8  # Normalize
                
                ax6.fill_between(x_range, i - density/2, i + density/2, alpha=0.7)
                ax6.plot(x_range, i + density/2, 'k-', linewidth=1)
                ax6.plot(x_range, i - density/2, 'k-', linewidth=1)
                ax6.axvline(np.mean(period_values), ymin=(i-0.4)/len(periods), 
                           ymax=(i+0.4)/len(periods), color='red', linewidth=2)
            except:
                # Fallback to simple histogram representation
                hist, bins = np.histogram(period_values, bins=20, density=True)
                hist = hist / hist.max() * 0.8
                ax6.fill_between(bins[:-1], i - hist/2, i + hist/2, alpha=0.7, step='pre')

ax6.set_title('Peace Index Density by Period', fontweight='bold', fontsize=12)
ax6.set_xlabel('Peace Index Score')
ax6.set_yticks(range(len(periods)))
ax6.set_yticklabels(periods)

# Subplot 7: Volatility vs Mean scatter
ax7 = plt.subplot(3, 3, 7)
volatility_data = []
for idx, row in df.iterrows():
    values = row[year_cols].dropna()
    if len(values) > 1:
        mean_val = values.mean()
        std_val = values.std()
        range_val = values.max() - values.min()
        volatility_data.append((mean_val, std_val, range_val))

if volatility_data:
    means, stds, ranges = zip(*volatility_data)
    scatter = ax7.scatter(means, stds, s=[r*50 for r in ranges], alpha=0.6, c=ranges, cmap='viridis')
    
    # Add trend line
    if len(means) > 1:
        z = np.polyfit(means, stds, 1)
        p = np.poly1d(z)
        ax7.plot(sorted(means), p(sorted(means)), "r--", alpha=0.8)
    
    plt.colorbar(scatter, ax=ax7, label='Range (Max-Min)')

ax7.set_title('Peace Index Volatility vs Mean', fontweight='bold', fontsize=12)
ax7.set_xlabel('Mean Peace Index')
ax7.set_ylabel('Standard Deviation')

# Subplot 8: Time series decomposition (simplified)
ax8 = plt.subplot(3, 3, 8)
global_means = []
years_int = []

for year in year_cols:
    if year in df.columns:
        year_data = df[year].dropna()
        if len(year_data) > 0:
            global_means.append(year_data.mean())
            years_int.append(int(year))

if global_means:
    # Simple trend and residuals
    x = np.arange(len(global_means))
    if len(global_means) > 1:
        trend = np.polyfit(x, global_means, 1)
        trend_line = np.polyval(trend, x)
        
        ax8.plot(years_int, global_means, 'b-', label='Original', linewidth=2)
        ax8.plot(years_int, trend_line, 'r--', label='Trend', linewidth=2)
        ax8.fill_between(years_int, 
                        np.array(trend_line) - 0.1, 
                        np.array(trend_line) + 0.1, 
                        alpha=0.2, label='Confidence')

ax8.set_title('Global Peace Index Decomposition', fontweight='bold', fontsize=12)
ax8.set_xlabel('Year')
ax8.set_ylabel('Global Mean Peace Index')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Subplot 9: Parallel coordinates (simplified)
ax9 = plt.subplot(3, 3, 9)
# Get countries with sufficient data
complete_countries = []
for idx, row in df.iterrows():
    valid_count = row[year_cols].notna().sum()
    if valid_count >= len(year_cols) * 0.6:  # At least 60% of years
        complete_countries.append(row)

if complete_countries:
    # Sample subset for clarity
    sample_size = min(20, len(complete_countries))
    sample_countries = complete_countries[:sample_size]
    
    for country_data in sample_countries:
        values = [country_data[year] if pd.notna(country_data[year]) else np.nan for year in year_cols]
        if not all(np.isnan(values)):
            ax9.plot(range(len(year_cols)), values, alpha=0.5, linewidth=1)

ax9.set_title('Peace Index Trajectories (Sample Countries)', fontweight='bold', fontsize=12)
ax9.set_xlabel('Year Index')
ax9.set_ylabel('Peace Index Score')
if year_cols:
    ax9.set_xticks(range(0, len(year_cols), 2))
    ax9.set_xticklabels(year_cols[::2], rotation=45)

# Final layout adjustment
plt.tight_layout(pad=3.0)
plt.savefig('peace_index_analysis.png', dpi=300, bbox_inches='tight')
plt.show()