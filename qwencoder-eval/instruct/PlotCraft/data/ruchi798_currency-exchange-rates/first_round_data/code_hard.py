import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('exchange_rates.csv')
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df = df.sort_values(['currency', 'date'])

# Define currency regions and major currencies
currency_regions = {
    'European': ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'DKK'],
    'Asian': ['JPY', 'CNY', 'KRW', 'SGD', 'HKD', 'THB'],
    'American': ['USD', 'CAD', 'MXN', 'BRL', 'ARS'],
    'Oceanic': ['AUD', 'NZD']
}

major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']
region_colors = {'European': '#2E86AB', 'Asian': '#A23B72', 'American': '#F18F01', 'Oceanic': '#C73E1D'}

# Create figure with white background
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.suptitle('Comprehensive Currency Exchange Rate Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

# Helper function to get currency region
def get_currency_region(currency):
    for region, currencies in currency_regions.items():
        if currency in currencies:
            return region
    return 'Other'

# Add region column
df['region'] = df['currency'].apply(get_currency_region)

# Subplot 1: Line chart with moving averages for top 5 strongest currencies
ax1 = plt.subplot(3, 3, 1)
top_currencies = df.groupby('currency')['value'].mean().nsmallest(5).index
for i, curr in enumerate(top_currencies):
    curr_data = df[df['currency'] == curr].copy()
    curr_data = curr_data.sort_values('date')
    
    # Calculate moving averages
    curr_data['ma7'] = curr_data['value'].rolling(window=7, min_periods=1).mean()
    curr_data['ma30'] = curr_data['value'].rolling(window=30, min_periods=1).mean()
    
    color = plt.cm.Set1(i)
    ax1.plot(curr_data['date'], curr_data['value'], alpha=0.3, color=color, linewidth=0.5)
    ax1.plot(curr_data['date'], curr_data['ma7'], color=color, linewidth=1.5, label=f'{curr} (7-day MA)')
    ax1.plot(curr_data['date'], curr_data['ma30'], color=color, linewidth=2, linestyle='--', alpha=0.8)

ax1.set_title('Top 5 Strongest Currencies with Moving Averages', fontweight='bold', fontsize=12)
ax1.set_ylabel('Exchange Rate (vs EUR)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Area chart with confidence bands for major currency groups
ax2 = plt.subplot(3, 3, 2)
for region, color in region_colors.items():
    region_currencies = currency_regions[region]
    region_data = df[df['currency'].isin(region_currencies)]
    
    if len(region_data) > 0:
        daily_stats = region_data.groupby('date')['value'].agg(['mean', 'std']).reset_index()
        daily_stats['upper'] = daily_stats['mean'] + 1.96 * daily_stats['std']
        daily_stats['lower'] = daily_stats['mean'] - 1.96 * daily_stats['std']
        
        ax2.fill_between(daily_stats['date'], daily_stats['lower'], daily_stats['upper'], 
                        alpha=0.3, color=color, label=f'{region} (95% CI)')
        ax2.plot(daily_stats['date'], daily_stats['mean'], color=color, linewidth=2)

ax2.set_title('Regional Currency Volatility with Confidence Bands', fontweight='bold', fontsize=12)
ax2.set_ylabel('Exchange Rate (vs EUR)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Seasonal decomposition for EUR/USD equivalent
ax3 = plt.subplot(3, 3, 3)
usd_data = df[df['currency'] == 'USD'].copy()
usd_data = usd_data.sort_values('date')

# Create trend using smoothing
trend = savgol_filter(usd_data['value'], window_length=min(51, len(usd_data)//4*2+1), polyorder=3)
residual = usd_data['value'] - trend

ax3.plot(usd_data['date'], usd_data['value'], color='black', linewidth=1, label='Original', alpha=0.7)
ax3.plot(usd_data['date'], trend, color='red', linewidth=2, label='Trend')
ax3.fill_between(usd_data['date'], trend, trend + residual, alpha=0.3, color='blue', label='Residual')

ax3.set_title('USD Exchange Rate Decomposition', fontweight='bold', fontsize=12)
ax3.set_ylabel('Exchange Rate')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Histogram with KDE for regional distributions
ax4 = plt.subplot(3, 3, 4)
for region, color in region_colors.items():
    region_currencies = currency_regions[region]
    region_values = df[df['currency'].isin(region_currencies)]['value']
    
    if len(region_values) > 0:
        # Normalize values for comparison
        normalized_values = np.log10(region_values + 1)
        ax4.hist(normalized_values, bins=30, alpha=0.3, color=color, density=True, label=region)
        
        # Add KDE
        kde_x = np.linspace(normalized_values.min(), normalized_values.max(), 100)
        kde = stats.gaussian_kde(normalized_values)
        ax4.plot(kde_x, kde(kde_x), color=color, linewidth=2)

ax4.set_title('Regional Currency Distribution (Log Scale)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Log(Exchange Rate + 1)')
ax4.set_ylabel('Density')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Scatter plot with regression lines
ax5 = plt.subplot(3, 3, 5)
# Calculate volatility and mean for each currency
currency_stats = df.groupby('currency').agg({
    'value': ['mean', 'std', 'count']
}).round(4)
currency_stats.columns = ['mean_rate', 'volatility', 'trading_volume']
currency_stats = currency_stats.reset_index()

# Add region information
currency_stats['region'] = currency_stats['currency'].apply(get_currency_region)

for region, color in region_colors.items():
    region_data = currency_stats[currency_stats['region'] == region]
    if len(region_data) > 0:
        scatter = ax5.scatter(region_data['mean_rate'], region_data['volatility'], 
                            s=region_data['trading_volume']/100, alpha=0.6, 
                            color=color, label=region, edgecolors='black', linewidth=0.5)

# Add regression line
x = currency_stats['mean_rate']
y = currency_stats['volatility']
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax5.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

ax5.set_title('Currency Strength vs Volatility (Bubble = Volume)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Mean Exchange Rate')
ax5.set_ylabel('Volatility (Std Dev)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 6: Box plots with violin overlays
ax6 = plt.subplot(3, 3, 6)
major_data = df[df['currency'].isin(major_currencies)]['value']
minor_data = df[~df['currency'].isin(major_currencies)]['value']

# Create violin plots
parts1 = ax6.violinplot([np.log10(major_data + 1)], positions=[1], widths=0.6, 
                       showmeans=True, showmedians=True)
parts2 = ax6.violinplot([np.log10(minor_data + 1)], positions=[2], widths=0.6, 
                       showmeans=True, showmedians=True)

# Color the violins
for pc in parts1['bodies']:
    pc.set_facecolor('#2E86AB')
    pc.set_alpha(0.7)
for pc in parts2['bodies']:
    pc.set_facecolor('#F18F01')
    pc.set_alpha(0.7)

# Add box plots
bp1 = ax6.boxplot([np.log10(major_data + 1)], positions=[1], widths=0.3, patch_artist=True)
bp2 = ax6.boxplot([np.log10(minor_data + 1)], positions=[2], widths=0.3, patch_artist=True)

bp1['boxes'][0].set_facecolor('#2E86AB')
bp2['boxes'][0].set_facecolor('#F18F01')

ax6.set_title('Major vs Minor Currencies Distribution', fontweight='bold', fontsize=12)
ax6.set_ylabel('Log(Exchange Rate + 1)')
ax6.set_xticks([1, 2])
ax6.set_xticklabels(['Major Currencies', 'Minor Currencies'])
ax6.grid(True, alpha=0.3)

# Subplot 7: Diverging bar chart with error bars
ax7 = plt.subplot(3, 3, 7)
# Calculate percentage change from EUR baseline
currency_change = df.groupby('currency')['value'].agg(['mean', 'std']).reset_index()
currency_change['pct_change'] = ((currency_change['mean'] - 1.0) / 1.0) * 100
currency_change['error'] = currency_change['std'] * 100
currency_change = currency_change.sort_values('pct_change')

# Select top and bottom currencies
top_bottom = pd.concat([currency_change.head(8), currency_change.tail(8)])
colors = ['red' if x < 0 else 'green' for x in top_bottom['pct_change']]

bars = ax7.barh(range(len(top_bottom)), top_bottom['pct_change'], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax7.errorbar(top_bottom['pct_change'], range(len(top_bottom)), 
            xerr=top_bottom['error'], fmt='none', color='black', capsize=3)

ax7.set_title('Currency Deviation from EUR Baseline', fontweight='bold', fontsize=12)
ax7.set_xlabel('Percentage Change from EUR (%)')
ax7.set_yticks(range(len(top_bottom)))
ax7.set_yticklabels(top_bottom['currency'], fontsize=8)
ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax7.grid(True, alpha=0.3)

# Subplot 8: Slope chart showing ranking changes
ax8 = plt.subplot(3, 3, 8)
# Calculate rankings for first and last periods
first_date = df['date'].min()
last_date = df['date'].max()
mid_date = first_date + (last_date - first_date) / 2

period1 = df[df['date'] <= mid_date].groupby('currency')['value'].mean()
period2 = df[df['date'] > mid_date].groupby('currency')['value'].mean()

common_currencies = set(period1.index) & set(period2.index) & set(major_currencies)
rank1 = period1[list(common_currencies)].rank()
rank2 = period2[list(common_currencies)].rank()

# Add volatility background
volatility_data = df[df['currency'].isin(common_currencies)].groupby('currency')['value'].std()
max_vol = volatility_data.max()

for i, curr in enumerate(common_currencies):
    vol_intensity = volatility_data[curr] / max_vol
    ax8.fill_between([0, 1], [rank1[curr]-0.4, rank2[curr]-0.4], 
                    [rank1[curr]+0.4, rank2[curr]+0.4], 
                    alpha=vol_intensity*0.3, color='gray')
    
    color = 'red' if rank2[curr] > rank1[curr] else 'green'
    ax8.plot([0, 1], [rank1[curr], rank2[curr]], 'o-', color=color, linewidth=2, markersize=6)
    ax8.text(-0.05, rank1[curr], curr, ha='right', va='center', fontsize=8)
    ax8.text(1.05, rank2[curr], curr, ha='left', va='center', fontsize=8)

ax8.set_title('Currency Ranking Changes Over Time', fontweight='bold', fontsize=12)
ax8.set_xlim(-0.2, 1.2)
ax8.set_xticks([0, 1])
ax8.set_xticklabels(['Period 1', 'Period 2'])
ax8.set_ylabel('Ranking (1=Strongest)')
ax8.grid(True, alpha=0.3)

# Subplot 9: Radar chart with line plots
ax9 = plt.subplot(3, 3, 9, projection='polar')

# Calculate metrics for major currencies
metrics = ['Stability', 'Strength', 'Volume']
currency_metrics = {}

for curr in major_currencies[:6]:  # Limit to 6 for clarity
    curr_data = df[df['currency'] == curr]
    if len(curr_data) > 0:
        stability = 1 / (curr_data['value'].std() + 0.001)  # Inverse of volatility
        strength = 1 / curr_data['value'].mean()  # Inverse of exchange rate
        volume = len(curr_data)  # Number of observations
        
        # Normalize metrics
        currency_metrics[curr] = [stability, strength, volume]

# Normalize all metrics to 0-1 scale
all_values = np.array(list(currency_metrics.values()))
normalized_metrics = {}
for i, curr in enumerate(currency_metrics.keys()):
    normalized_metrics[curr] = (all_values[i] - all_values.min(axis=0)) / (all_values.max(axis=0) - all_values.min(axis=0))

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, (curr, values) in enumerate(normalized_metrics.items()):
    values = values.tolist()
    values += values[:1]  # Complete the circle
    
    color = plt.cm.Set1(i)
    ax9.plot(angles, values, 'o-', linewidth=2, label=curr, color=color)
    ax9.fill(angles, values, alpha=0.1, color=color)

ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(metrics)
ax9.set_title('Multi-Metric Currency Performance', fontweight='bold', fontsize=12, pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
plt.show()