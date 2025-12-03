import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('APY.csv')

# Clean column names and handle missing values
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Production', 'Area', 'Yield'])

# Sample data for faster processing (take every 10th row to reduce computation time)
df_sample = df.iloc[::10].copy()

# Create comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Prepare aggregated data for analysis
yearly_data = df_sample.groupby('Crop_Year').agg({
    'Production': ['sum', 'std'],
    'Area': 'sum',
    'Yield': 'mean'
}).round(2)

yearly_data.columns = ['Total_Production', 'Production_Std', 'Total_Area', 'Avg_Yield']
yearly_data = yearly_data.reset_index()
yearly_data['Production_Std'] = yearly_data['Production_Std'].fillna(0)

# 1. Top-left: Line chart with confidence bands and bar chart overlay
ax1 = plt.subplot(3, 3, 1)
years = yearly_data['Crop_Year']
production = yearly_data['Total_Production']
variance = yearly_data['Production_Std']

# Line chart with confidence bands
ax1.plot(years, production, color='#2E86AB', linewidth=2, label='Total Production')
ax1.fill_between(years, production - variance, production + variance, 
                alpha=0.3, color='#2E86AB')

# Secondary axis for variance bars
ax1_twin = ax1.twinx()
ax1_twin.bar(years, variance, alpha=0.5, color='#F24236', width=0.6)

ax1.set_title('Production Trends with Variance', fontweight='bold', fontsize=10)
ax1.set_xlabel('Year')
ax1.set_ylabel('Production', color='#2E86AB')
ax1_twin.set_ylabel('Variance', color='#F24236')
ax1.grid(True, alpha=0.3)

# 2. Top-center: Stacked area chart with seasonal data
ax2 = plt.subplot(3, 3, 2)
seasonal_data = df_sample.groupby(['Crop_Year', 'Season'])['Production'].sum().unstack(fill_value=0)

if not seasonal_data.empty and len(seasonal_data.columns) > 1:
    # Limit to top 4 seasons for clarity
    top_seasons = seasonal_data.sum().nlargest(4).index
    seasonal_subset = seasonal_data[top_seasons]
    
    ax2.stackplot(seasonal_subset.index, seasonal_subset.T, 
                 labels=seasonal_subset.columns, alpha=0.7)
    
    ax2.set_title('Seasonal Production Composition', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Production by Season')
    ax2.legend(loc='upper left', fontsize=8)

# 3. Top-right: Multi-line time series for top crops
ax3 = plt.subplot(3, 3, 3)
top_crops = df_sample.groupby('Crop')['Production'].sum().nlargest(5).index
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, crop in enumerate(top_crops):
    crop_data = df_sample[df_sample['Crop'] == crop].groupby('Crop_Year')['Production'].sum()
    if len(crop_data) > 1:
        ax3.plot(crop_data.index, crop_data.values, 
                color=colors[i], linewidth=2, label=crop[:15], marker='o', markersize=3)

ax3.set_title('Top 5 Crops Production Trends', fontweight='bold', fontsize=10)
ax3.set_xlabel('Year')
ax3.set_ylabel('Production')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Middle-left: Simple heatmap
ax4 = plt.subplot(3, 3, 4)
# Create a simplified monthly pattern based on seasons
season_month_map = {'Kharif': [6, 7, 8, 9], 'Rabi': [11, 12, 1, 2], 'Summer': [3, 4, 5], 'Autumn': [9, 10, 11]}
monthly_prod = np.zeros((len(yearly_data), 12))

for idx, year in enumerate(yearly_data['Crop_Year']):
    year_data = df_sample[df_sample['Crop_Year'] == year]
    for season in year_data['Season'].unique():
        if season in season_month_map:
            prod_val = year_data[year_data['Season'] == season]['Production'].sum()
            for month in season_month_map[season]:
                if month <= 12:
                    monthly_prod[idx, month-1] += prod_val / len(season_month_map[season])

im = ax4.imshow(monthly_prod, cmap='YlOrRd', aspect='auto')
ax4.set_title('Monthly Production Pattern', fontweight='bold', fontsize=10)
ax4.set_xlabel('Month')
ax4.set_ylabel('Year Index')
ax4.set_xticks(range(12))
ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

# 5. Middle-center: Slope chart for states
ax5 = plt.subplot(3, 3, 5)
state_yearly = df_sample.groupby(['State', 'Crop_Year'])['Production'].sum().unstack(fill_value=0)

if not state_yearly.empty and state_yearly.shape[1] > 1:
    # Select top 8 states and compare first vs last year
    state_totals = state_yearly.sum(axis=1).nlargest(8)
    
    first_year_col = state_yearly.columns[0]
    last_year_col = state_yearly.columns[-1]
    
    y_pos = range(len(state_totals))
    
    for i, state in enumerate(state_totals.index):
        first_val = state_yearly.loc[state, first_year_col]
        last_val = state_yearly.loc[state, last_year_col]
        
        ax5.plot([0, 1], [first_val, last_val], 'o-', linewidth=2, markersize=5, alpha=0.7)
        ax5.text(-0.1, first_val, state[:10], ha='right', va='center', fontsize=8)
    
    ax5.set_title('State Production: First vs Last Year', fontweight='bold', fontsize=10)
    ax5.set_xlim(-0.3, 1.3)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['First Year', 'Last Year'])
    ax5.set_ylabel('Production')

# 6. Middle-right: Simple trend decomposition
ax6 = plt.subplot(3, 3, 6)
if len(yearly_data) > 3:
    # Simple trend analysis
    x = np.arange(len(yearly_data))
    y = yearly_data['Total_Production'].values
    
    # Linear trend
    z = np.polyfit(x, y, 1)
    trend_line = np.poly1d(z)
    
    # Residuals
    residuals = y - trend_line(x)
    
    ax6.plot(yearly_data['Crop_Year'], y, 'o-', color='blue', label='Actual', linewidth=2)
    ax6.plot(yearly_data['Crop_Year'], trend_line(x), '--', color='red', label='Trend', linewidth=2)
    ax6.fill_between(yearly_data['Crop_Year'], trend_line(x), y, alpha=0.3, color='green', label='Residuals')
    
    ax6.set_title('Production Trend Analysis', fontweight='bold', fontsize=10)
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Production')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

# 7. Bottom-left: Cumulative and growth rate
ax7 = plt.subplot(3, 3, 7)
cumulative_prod = yearly_data['Total_Production'].cumsum()
growth_rates = yearly_data['Total_Production'].pct_change() * 100

# Area chart for cumulative
ax7.fill_between(years, cumulative_prod, alpha=0.6, color='#9B59B6')

# Line chart for growth rates
ax7_twin = ax7.twinx()
ax7_twin.plot(years, growth_rates, color='#E67E22', linewidth=2, marker='o', markersize=4)
ax7_twin.axhline(y=0, color='red', linestyle='--', alpha=0.7)

ax7.set_title('Cumulative Production & Growth', fontweight='bold', fontsize=10)
ax7.set_xlabel('Year')
ax7.set_ylabel('Cumulative', color='#9B59B6')
ax7_twin.set_ylabel('Growth %', color='#E67E22')

# 8. Bottom-center: Simple autocorrelation
ax8 = plt.subplot(3, 3, 8)
if len(yearly_data) > 3:
    production_values = yearly_data['Total_Production'].values
    n = len(production_values)
    
    # Calculate simple autocorrelation for lags 1-5
    max_lag = min(5, n-1)
    autocorr_values = []
    lags = range(1, max_lag + 1)
    
    for lag in lags:
        if n > lag:
            corr = np.corrcoef(production_values[:-lag], production_values[lag:])[0, 1]
            autocorr_values.append(corr)
        else:
            autocorr_values.append(0)
    
    ax8.bar(lags, autocorr_values, alpha=0.7, color='#1ABC9C')
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax8.set_title('Production Autocorrelation', fontweight='bold', fontsize=10)
    ax8.set_xlabel('Lag')
    ax8.set_ylabel('Correlation')
    ax8.grid(True, alpha=0.3)

# 9. Bottom-right: Cross-correlation between area and production
ax9 = plt.subplot(3, 3, 9)
if len(yearly_data) > 3:
    area_values = yearly_data['Total_Area'].values
    prod_values = yearly_data['Total_Production'].values
    
    # Simple cross-correlation
    max_lag = min(3, len(area_values) - 1)
    lags = range(-max_lag, max_lag + 1)
    cross_corr = []
    
    for lag in lags:
        if lag == 0:
            corr = np.corrcoef(area_values, prod_values)[0, 1]
        elif lag > 0 and len(area_values) > lag:
            corr = np.corrcoef(area_values[:-lag], prod_values[lag:])[0, 1]
        elif lag < 0 and len(prod_values) > abs(lag):
            corr = np.corrcoef(area_values[abs(lag):], prod_values[:lag])[0, 1]
        else:
            corr = 0
        cross_corr.append(corr)
    
    ax9.bar(lags, cross_corr, alpha=0.7, color='#E74C3C')
    ax9.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax9.set_title('Area-Production Cross-Correlation', fontweight='bold', fontsize=10)
    ax9.set_xlabel('Lag')
    ax9.set_ylabel('Correlation')
    ax9.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=1.5)
plt.suptitle('Comprehensive Temporal Analysis of Indian Crop Production', 
             fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.94)
plt.savefig('crop_production_analysis.png', dpi=300, bbox_inches='tight')
plt.show()