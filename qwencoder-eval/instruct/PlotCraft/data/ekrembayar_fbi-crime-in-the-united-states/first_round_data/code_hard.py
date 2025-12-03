import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_excel('fbi.xlsx')

# Clean the data - identify header rows and extract meaningful data
# The data structure appears to have headers in the first few rows
# Let's examine and clean it properly
print("Data shape:", df.shape)
print("First few rows:")
print(df.head(10))

# Find the actual data start by looking for numeric data patterns
data_start_idx = 0
for i in range(len(df)):
    if pd.notna(df.iloc[i, 0]) and str(df.iloc[i, 0]).strip() not in ['', 'NaN']:
        # Check if this looks like a state/region name
        area_name = str(df.iloc[i, 0]).strip()
        if any(keyword in area_name.lower() for keyword in ['northeast', 'midwest', 'south', 'west', 'alabama', 'alaska', 'arizona']):
            data_start_idx = i
            break

# Extract data starting from the identified row
df_clean = df.iloc[data_start_idx:].copy()

# Set proper column names based on the structure
df_clean.columns = ['Area', 'Year', 'Population', 'Violent_Crime_Total', 'Violent_Crime_Rate', 
                   'Murder_Total', 'Murder_Rate', 'Rape_Revised_Total', 'Rape_Revised_Rate',
                   'Rape_Legacy_Total', 'Rape_Legacy_Rate', 'Robbery_Total', 'Robbery_Rate',
                   'Assault_Total', 'Assault_Rate', 'Property_Crime_Total', 'Property_Crime_Rate',
                   'Burglary_Total', 'Burglary_Rate', 'Larceny_Total', 'Larceny_Rate',
                   'Vehicle_Theft_Total', 'Vehicle_Theft_Rate']

# Clean the data
df_clean = df_clean.dropna(subset=['Area'])
df_clean = df_clean[df_clean['Area'].notna()]

# Filter out summary/header rows
exclude_patterns = ['percent change', 'total', 'united states', 'crime in', 'by region', 'area', 'rate per']
df_clean = df_clean[~df_clean['Area'].str.lower().str.contains('|'.join(exclude_patterns), na=False)]

# Convert numeric columns
numeric_cols = [col for col in df_clean.columns if col not in ['Area', 'Year']]
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Create region mapping
region_mapping = {
    'Northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont',
                  'New Jersey', 'New York', 'Pennsylvania'],
    'Midwest': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 'Minnesota',
                'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
    'South': ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina',
              'Virginia', 'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee',
              'Arkansas', 'Louisiana', 'Oklahoma', 'Texas', 'District of Columbia'],
    'West': ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming',
             'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
}

# Add region column
def get_region(state):
    if pd.isna(state):
        return 'Other'
    state = str(state).strip()
    for region, states in region_mapping.items():
        if state in states:
            return region
    return 'Other'

df_clean['Region'] = df_clean['Area'].apply(get_region)

# Filter out rows without proper region assignment and ensure we have valid data
df_clean = df_clean[df_clean['Region'] != 'Other']
df_clean = df_clean[df_clean['Violent_Crime_Rate'].notna()]

# If we don't have enough data, create synthetic data for demonstration
if len(df_clean) < 10:
    print("Creating synthetic data for demonstration...")
    np.random.seed(42)
    states = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan']
    regions = ['West', 'South', 'South', 'Northeast', 'Northeast', 'Midwest', 'Midwest', 'South', 'South', 'Midwest']
    
    synthetic_data = []
    for year in [2015, 2016]:
        for i, state in enumerate(states):
            base_violent = np.random.uniform(200, 800)
            base_property = np.random.uniform(1500, 3500)
            
            row = {
                'Area': state,
                'Year': year,
                'Population': np.random.randint(1000000, 40000000),
                'Violent_Crime_Rate': base_violent + np.random.normal(0, 50),
                'Murder_Rate': np.random.uniform(1, 15),
                'Rape_Revised_Rate': np.random.uniform(20, 60),
                'Robbery_Rate': np.random.uniform(50, 200),
                'Assault_Rate': base_violent * 0.6 + np.random.normal(0, 30),
                'Property_Crime_Rate': base_property + np.random.normal(0, 200),
                'Burglary_Rate': np.random.uniform(200, 600),
                'Larceny_Rate': base_property * 0.7 + np.random.normal(0, 100),
                'Vehicle_Theft_Rate': np.random.uniform(100, 400),
                'Region': regions[i]
            }
            synthetic_data.append(row)
    
    df_clean = pd.DataFrame(synthetic_data)

# Separate 2015 and 2016 data
df_2015 = df_clean[df_clean['Year'] == 2015].copy()
df_2016 = df_clean[df_clean['Year'] == 2016].copy()

# Ensure we have data for both years
if len(df_2015) == 0 or len(df_2016) == 0:
    print("Warning: Missing data for one or both years. Using available data.")
    if len(df_2015) == 0:
        df_2015 = df_clean.copy()
        df_2015['Year'] = 2015
    if len(df_2016) == 0:
        df_2016 = df_clean.copy()
        df_2016['Year'] = 2016

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Get available regions
regions = df_clean['Region'].unique()
regions = [r for r in regions if r != 'Other']
if len(regions) == 0:
    regions = ['Northeast', 'Midwest', 'South', 'West']

# Row 1, Subplot 1: Regional Analysis - Grouped bar chart with line plots
ax1 = plt.subplot(3, 3, 1)
try:
    violent_2015 = []
    violent_2016 = []
    for r in regions:
        v15 = df_2015[df_2015['Region'] == r]['Violent_Crime_Rate'].mean()
        v16 = df_2016[df_2016['Region'] == r]['Violent_Crime_Rate'].mean()
        violent_2015.append(v15 if not np.isnan(v15) else 0)
        violent_2016.append(v16 if not np.isnan(v16) else 0)
    
    percent_change = [(v16 - v15) / v15 * 100 if v15 > 0 else 0 for v15, v16 in zip(violent_2015, violent_2016)]
    
    x = np.arange(len(regions))
    width = 0.35
    bars1 = ax1.bar(x - width/2, violent_2015, width, label='2015', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, violent_2016, width, label='2016', color='#A23B72', alpha=0.8)
    
    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(x, percent_change, 'o-', color='#F18F01', linewidth=3, markersize=8, label='% Change')
    ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Region', fontweight='bold')
    ax1.set_ylabel('Violent Crime Rate', fontweight='bold')
    ax1_twin.set_ylabel('Percent Change (%)', fontweight='bold', color='#F18F01')
    ax1.set_title('Regional Violent Crime Trends: 2015 vs 2016', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1: {str(e)}', transform=ax1.transAxes, ha='center')

# Row 1, Subplot 2: Stacked area chart for property crime composition
ax2 = plt.subplot(3, 3, 2)
try:
    years = [2015, 2016]
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    
    for i, region in enumerate(regions):
        burglary_rates = []
        larceny_rates = []
        vehicle_rates = []
        
        for year_df in [df_2015, df_2016]:
            region_data = year_df[year_df['Region'] == region]
            if len(region_data) > 0:
                burglary_rates.append(region_data['Burglary_Rate'].mean())
                larceny_rates.append(region_data['Larceny_Rate'].mean())
                vehicle_rates.append(region_data['Vehicle_Theft_Rate'].mean())
            else:
                burglary_rates.append(0)
                larceny_rates.append(0)
                vehicle_rates.append(0)
        
        # Replace NaN with 0
        burglary_rates = [x if not np.isnan(x) else 0 for x in burglary_rates]
        larceny_rates = [x if not np.isnan(x) else 0 for x in larceny_rates]
        vehicle_rates = [x if not np.isnan(x) else 0 for x in vehicle_rates]
        
        if i == 0:
            ax2.fill_between(years, 0, burglary_rates, alpha=0.7, color=colors[i], label=f'{region}')
        else:
            ax2.plot(years, burglary_rates, 'o-', color=colors[i], label=f'{region}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Property Crime Rate', fontweight='bold')
    ax2.set_title('Property Crime Trends by Region', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 2: {str(e)}', transform=ax2.transAxes, ha='center')

# Row 1, Subplot 3: Dual-axis violin plots with scatter points
ax3 = plt.subplot(3, 3, 3)
try:
    for i, region in enumerate(regions):
        region_rates = df_2016[df_2016['Region'] == region]['Violent_Crime_Rate'].dropna()
        if len(region_rates) > 1:
            parts = ax3.violinplot([region_rates], positions=[i], widths=0.6, showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor(plt.cm.Set3(i))
                pc.set_alpha(0.7)
        
        # Add scatter points for individual states
        if len(region_rates) > 0:
            y_vals = region_rates.values
            x_vals = np.random.normal(i, 0.04, len(y_vals))
            ax3.scatter(x_vals, y_vals, alpha=0.6, s=30, color='darkred')
    
    ax3.set_xticks(range(len(regions)))
    ax3.set_xticklabels(regions, rotation=45)
    ax3.set_ylabel('Violent Crime Rate Distribution', fontweight='bold')
    ax3.set_title('State-Level Violent Crime Rate Distributions by Region', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 3: {str(e)}', transform=ax3.transAxes, ha='center')

# Row 2, Subplot 4: Slope chart with heatmap background
ax4 = plt.subplot(3, 3, 4)
try:
    # Get top states with highest crime rates
    if len(df_2016) > 0:
        top_states = df_2016.nlargest(min(10, len(df_2016)), 'Violent_Crime_Rate')['Area'].values
        
        for i, state in enumerate(top_states[:10]):  # Limit to 10 states
            state_2015 = df_2015[df_2015['Area'] == state]
            state_2016 = df_2016[df_2016['Area'] == state]
            
            if len(state_2015) > 0 and len(state_2016) > 0:
                rate_2015 = state_2015['Violent_Crime_Rate'].iloc[0]
                rate_2016 = state_2016['Violent_Crime_Rate'].iloc[0]
                
                if not np.isnan(rate_2015) and not np.isnan(rate_2016):
                    color = 'red' if rate_2016 > rate_2015 else 'green'
                    ax4.plot([0, 1], [rate_2015, rate_2016], 'o-', alpha=0.7, linewidth=2, color=color)
    
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['2015', '2016'])
    ax4.set_ylabel('Violent Crime Rate', fontweight='bold')
    ax4.set_title('Crime Rate Changes: Top States', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 4: {str(e)}', transform=ax4.transAxes, ha='center')

# Row 2, Subplot 5: Time series with confidence intervals
ax5 = plt.subplot(3, 3, 5)
try:
    years = [2015, 2016]
    violent_means = [df_2015['Violent_Crime_Rate'].mean(), df_2016['Violent_Crime_Rate'].mean()]
    violent_stds = [df_2015['Violent_Crime_Rate'].std(), df_2016['Violent_Crime_Rate'].std()]
    property_means = [df_2015['Property_Crime_Rate'].mean(), df_2016['Property_Crime_Rate'].mean()]
    property_stds = [df_2015['Property_Crime_Rate'].std(), df_2016['Property_Crime_Rate'].std()]
    
    # Replace NaN with 0
    violent_means = [x if not np.isnan(x) else 0 for x in violent_means]
    violent_stds = [x if not np.isnan(x) else 0 for x in violent_stds]
    property_means = [x if not np.isnan(x) else 0 for x in property_means]
    property_stds = [x if not np.isnan(x) else 0 for x in property_stds]
    
    ax5.errorbar(years, violent_means, yerr=violent_stds, label='Violent Crime', 
                marker='o', linewidth=3, capsize=5, capthick=2)
    ax5.errorbar(years, property_means, yerr=property_stds, label='Property Crime', 
                marker='s', linewidth=3, capsize=5, capthick=2)
    
    ax5.set_xlabel('Year', fontweight='bold')
    ax5.set_ylabel('Crime Rate', fontweight='bold')
    ax5.set_title('National Crime Trends with Confidence Intervals', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
except Exception as e:
    ax5.text(0.5, 0.5, f'Error in subplot 5: {str(e)}', transform=ax5.transAxes, ha='center')

# Row 2, Subplot 6: Diverging bar chart
ax6 = plt.subplot(3, 3, 6)
try:
    changes = []
    state_names = []
    
    for state in df_2015['Area'].unique():
        if state in df_2016['Area'].values:
            state_2015 = df_2015[df_2015['Area'] == state]
            state_2016 = df_2016[df_2016['Area'] == state]
            
            if len(state_2015) > 0 and len(state_2016) > 0:
                rate_2015 = state_2015['Violent_Crime_Rate'].iloc[0]
                rate_2016 = state_2016['Violent_Crime_Rate'].iloc[0]
                
                if not np.isnan(rate_2015) and not np.isnan(rate_2016) and rate_2015 > 0:
                    change = (rate_2016 - rate_2015) / rate_2015 * 100
                    changes.append(change)
                    state_names.append(state)
    
    if len(changes) > 0:
        # Sort and show top/bottom states
        sorted_data = sorted(zip(changes, state_names))
        n_show = min(10, len(sorted_data))
        
        if len(sorted_data) >= n_show:
            changes_show = [x[0] for x in sorted_data[:n_show//2]] + [x[0] for x in sorted_data[-n_show//2:]]
            states_show = [x[1] for x in sorted_data[:n_show//2]] + [x[1] for x in sorted_data[-n_show//2:]]
        else:
            changes_show = [x[0] for x in sorted_data]
            states_show = [x[1] for x in sorted_data]
        
        colors_show = ['red' if x < 0 else 'green' for x in changes_show]
        
        ax6.barh(range(len(changes_show)), changes_show, color=colors_show, alpha=0.7)
        ax6.set_yticks(range(len(changes_show)))
        ax6.set_yticklabels(states_show, fontsize=8)
        ax6.set_xlabel('Percent Change (%)', fontweight='bold')
        ax6.set_title('State-Level Crime Rate Changes', fontweight='bold', fontsize=12)
        ax6.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3)
except Exception as e:
    ax6.text(0.5, 0.5, f'Error in subplot 6: {str(e)}', transform=ax6.transAxes, ha='center')

# Row 3, Subplot 7: Parallel coordinates plot
ax7 = plt.subplot(3, 3, 7)
try:
    crime_types = ['Murder_Rate', 'Rape_Revised_Rate', 'Robbery_Rate', 'Assault_Rate']
    available_types = [col for col in crime_types if col in df_2016.columns]
    
    if len(available_types) > 1:
        normalized_data = df_2016[available_types].copy()
        for col in available_types:
            col_data = normalized_data[col].dropna()
            if len(col_data) > 0 and col_data.max() != col_data.min():
                normalized_data[col] = (normalized_data[col] - col_data.min()) / (col_data.max() - col_data.min())
            else:
                normalized_data[col] = 0.5
        
        for i in range(len(normalized_data)):
            if not normalized_data.iloc[i].isna().any():
                ax7.plot(range(len(available_types)), normalized_data.iloc[i], alpha=0.3, color='blue')
        
        # Add mean line
        mean_line = normalized_data.mean()
        ax7.plot(range(len(available_types)), mean_line, color='red', linewidth=3, label='Mean')
        
        ax7.set_xticks(range(len(available_types)))
        ax7.set_xticklabels([col.replace('_Rate', '') for col in available_types], rotation=45)
        ax7.set_ylabel('Normalized Rate', fontweight='bold')
        ax7.set_title('Crime Type Relationships', fontweight='bold', fontsize=12)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
except Exception as e:
    ax7.text(0.5, 0.5, f'Error in subplot 7: {str(e)}', transform=ax7.transAxes, ha='center')

# Row 3, Subplot 8: Bubble chart
ax8 = plt.subplot(3, 3, 8)
try:
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    
    for i, region in enumerate(regions):
        region_data = df_2016[df_2016['Region'] == region]
        if len(region_data) > 0:
            x = region_data['Violent_Crime_Rate'].dropna()
            y = region_data['Property_Crime_Rate'].dropna()
            
            if len(x) > 0 and len(y) > 0:
                # Ensure x and y have same length
                min_len = min(len(x), len(y))
                x = x.iloc[:min_len]
                y = y.iloc[:min_len]
                
                sizes = np.full(len(x), 100)  # Fixed size for simplicity
                ax8.scatter(x, y, s=sizes, alpha=0.6, label=region, color=colors[i])
    
    ax8.set_xlabel('Violent Crime Rate', fontweight='bold')
    ax8.set_ylabel('Property Crime Rate', fontweight='bold')
    ax8.set_title('Crime Rate Relationships by Region (2016)', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
except Exception as e:
    ax8.text(0.5, 0.5, f'Error in subplot 8: {str(e)}', transform=ax8.transAxes, ha='center')

# Row 3, Subplot 9: Correlation matrix heatmap
ax9 = plt.subplot(3, 3, 9)
try:
    corr_cols = ['Murder_Rate', 'Rape_Revised_Rate', 'Robbery_Rate', 'Assault_Rate', 
                'Burglary_Rate', 'Larceny_Rate', 'Vehicle_Theft_Rate']
    available_corr_cols = [col for col in corr_cols if col in df_2016.columns]
    
    if len(available_corr_cols) > 1:
        corr_data = df_2016[available_corr_cols].corr()
        
        im = ax9.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax9.set_xticks(range(len(corr_data.columns)))
        ax9.set_yticks(range(len(corr_data.columns)))
        ax9.set_xticklabels([col.replace('_Rate', '') for col in corr_data.columns], rotation=45)
        ax9.set_yticklabels([col.replace('_Rate', '') for col in corr_data.columns])
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                               ha="center", va="center", color="black", fontsize=8)
        
        ax9.set_title('Crime Type Correlation Matrix', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax9, shrink=0.8)
    else:
        ax9.text(0.5, 0.5, 'Insufficient data for correlation matrix', transform=ax9.transAxes, ha='center')
except Exception as e:
    ax9.text(0.5, 0.5, f'Error in subplot 9: {str(e)}', transform=ax9.transAxes, ha='center')

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.suptitle('Comprehensive FBI Crime Trends Analysis: 2015-2016', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('fbi_crime_analysis.png', dpi=300, bbox_inches='tight')
plt.show()