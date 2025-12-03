import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('Credit card transactions - India - Simple.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Extract time components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palettes
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
city_colors = {'Delhi, India': '#2E86AB', 'Greater Mumbai, India': '#A23B72', 
               'Bengaluru, India': '#F18F01', 'Ahmedabad, India': '#C73E1D'}

# (1,1) Monthly spending trends with moving average and confidence bands
monthly_spending = df.groupby([df['Date'].dt.to_period('M')])['Amount'].agg(['sum', 'std', 'count']).reset_index()
monthly_spending['Date'] = monthly_spending['Date'].dt.to_timestamp()
monthly_spending = monthly_spending.sort_values('Date')

# Calculate moving average
window = 3
monthly_spending['MA'] = monthly_spending['sum'].rolling(window=window, center=True).mean()
monthly_spending['std_err'] = monthly_spending['std'] / np.sqrt(monthly_spending['count'])
monthly_spending['upper_ci'] = monthly_spending['sum'] + 1.96 * monthly_spending['std_err']
monthly_spending['lower_ci'] = monthly_spending['sum'] - 1.96 * monthly_spending['std_err']

axes[0,0].plot(monthly_spending['Date'], monthly_spending['sum'], 'o-', color='#2E86AB', alpha=0.7, linewidth=2, label='Monthly Spending')
axes[0,0].plot(monthly_spending['Date'], monthly_spending['MA'], '-', color='#C73E1D', linewidth=3, label=f'{window}-Month MA')
axes[0,0].fill_between(monthly_spending['Date'], monthly_spending['lower_ci'], monthly_spending['upper_ci'], 
                       alpha=0.2, color='#2E86AB', label='95% CI')
axes[0,0].set_title('Monthly Spending Trends with Moving Average', fontweight='bold', fontsize=12)
axes[0,0].set_ylabel('Amount (₹)', fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# (1,2) Seasonal spending patterns by card type with stacked area chart
seasonal_card = df.groupby(['Month', 'Card Type'])['Amount'].sum().unstack(fill_value=0)
months = range(1, 13)

# Create stacked area chart
bottom = np.zeros(len(months))
for i, card_type in enumerate(seasonal_card.columns):
    values = [seasonal_card.loc[month, card_type] if month in seasonal_card.index else 0 for month in months]
    axes[0,1].fill_between(months, bottom, bottom + values, alpha=0.7, color=colors[i % len(colors)], label=card_type)
    # Add trend line
    z = np.polyfit(months, values, 1)
    p = np.poly1d(z)
    axes[0,1].plot(months, p(months), '--', color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    bottom += values

axes[0,1].set_title('Seasonal Spending by Card Type with Trends', fontweight='bold', fontsize=12)
axes[0,1].set_xlabel('Month', fontweight='bold')
axes[0,1].set_ylabel('Amount (₹)', fontweight='bold')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].grid(True, alpha=0.3)

# (1,3) Daily spending distribution with histogram and KDE overlay
daily_spending = df.groupby('Date')['Amount'].sum()

axes[0,2].hist(daily_spending, bins=30, alpha=0.7, color='#2E86AB', density=True, label='Histogram')
kde = gaussian_kde(daily_spending)
x_range = np.linspace(daily_spending.min(), daily_spending.max(), 100)
axes[0,2].plot(x_range, kde(x_range), '-', color='#C73E1D', linewidth=3, label='KDE')
axes[0,2].axvline(daily_spending.mean(), color='#F18F01', linestyle='--', linewidth=2, label=f'Mean: ₹{daily_spending.mean():.0f}')
axes[0,2].set_title('Daily Spending Distribution', fontweight='bold', fontsize=12)
axes[0,2].set_xlabel('Daily Amount (₹)', fontweight='bold')
axes[0,2].set_ylabel('Density', fontweight='bold')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# (2,1) City-wise spending evolution with filled areas
city_monthly = df.groupby([df['Date'].dt.to_period('M'), 'City'])['Amount'].sum().unstack(fill_value=0)
city_monthly.index = city_monthly.index.to_timestamp()

for i, city in enumerate(city_monthly.columns):
    axes[1,0].plot(city_monthly.index, city_monthly[city], '-', linewidth=2, 
                   color=city_colors.get(city, colors[i % len(colors)]), label=city.split(',')[0])
    axes[1,0].fill_between(city_monthly.index, 0, city_monthly[city], alpha=0.3, 
                           color=city_colors.get(city, colors[i % len(colors)]))

axes[1,0].set_title('City-wise Spending Evolution', fontweight='bold', fontsize=12)
axes[1,0].set_ylabel('Amount (₹)', fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# (2,2) Card type performance with bars and cumulative line
card_monthly = df.groupby([df['Date'].dt.to_period('M'), 'Card Type'])['Amount'].sum().unstack(fill_value=0)
card_totals = card_monthly.sum()

# Bar chart
bars = axes[1,1].bar(range(len(card_totals)), card_totals.values, color=colors[:len(card_totals)], alpha=0.7)
axes[1,1].set_xticks(range(len(card_totals)))
axes[1,1].set_xticklabels(card_totals.index, rotation=45)

# Cumulative line overlay
ax2 = axes[1,1].twinx()
cumulative = np.cumsum(sorted(card_totals.values, reverse=True))
ax2.plot(range(len(cumulative)), cumulative, 'o-', color='#C73E1D', linewidth=3, markersize=8)
ax2.set_ylabel('Cumulative Amount (₹)', fontweight='bold', color='#C73E1D')

axes[1,1].set_title('Card Type Performance', fontweight='bold', fontsize=12)
axes[1,1].set_ylabel('Total Amount (₹)', fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

# (2,3) Weekly spending patterns with box plots and violin overlay
weekly_data = []
week_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for day in range(7):
    day_amounts = df[df['DayOfWeek'] == day]['Amount'].values
    weekly_data.append(day_amounts)

# Box plots (removed alpha parameter)
bp = axes[1,2].boxplot(weekly_data, positions=range(7), patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)  # Set alpha on the patch object instead

# Violin plot overlay
parts = axes[1,2].violinplot(weekly_data, positions=range(7), alpha=0.3, showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor('#2E86AB')
    pc.set_alpha(0.3)

axes[1,2].set_title('Weekly Spending Patterns', fontweight='bold', fontsize=12)
axes[1,2].set_xlabel('Day of Week', fontweight='bold')
axes[1,2].set_ylabel('Amount (₹)', fontweight='bold')
axes[1,2].set_xticks(range(7))
axes[1,2].set_xticklabels(week_labels)
axes[1,2].grid(True, alpha=0.3)

# (3,1) Quarterly spending analysis with grouped bars and trend
quarterly_data = df.groupby(['Year', 'Quarter'])['Amount'].sum().reset_index()
quarterly_data['Period'] = quarterly_data['Year'].astype(str) + '-Q' + quarterly_data['Quarter'].astype(str)

bars = axes[2,0].bar(range(len(quarterly_data)), quarterly_data['Amount'], 
                     color=colors[:len(quarterly_data)], alpha=0.7)

# Trend line
z = np.polyfit(range(len(quarterly_data)), quarterly_data['Amount'], 1)
p = np.poly1d(z)
axes[2,0].plot(range(len(quarterly_data)), p(range(len(quarterly_data))), 
               '-', color='#C73E1D', linewidth=3, label='Trend')

axes[2,0].set_title('Quarterly Spending Analysis', fontweight='bold', fontsize=12)
axes[2,0].set_xlabel('Quarter', fontweight='bold')
axes[2,0].set_ylabel('Amount (₹)', fontweight='bold')
axes[2,0].set_xticks(range(len(quarterly_data)))
axes[2,0].set_xticklabels(quarterly_data['Period'], rotation=45)
axes[2,0].legend()
axes[2,0].grid(True, alpha=0.3)

# (3,2) Month-over-month growth rates with error bars
monthly_amounts = monthly_spending['sum'].values
growth_rates = [(monthly_amounts[i] - monthly_amounts[i-1]) / monthly_amounts[i-1] * 100 
                for i in range(1, len(monthly_amounts))]
growth_dates = monthly_spending['Date'].iloc[1:].values

# Calculate error bars (standard error)
growth_std = np.std(growth_rates)
error_bars = [growth_std / np.sqrt(len(growth_rates))] * len(growth_rates)

axes[2,1].errorbar(range(len(growth_rates)), growth_rates, yerr=error_bars, 
                   fmt='o-', color='#2E86AB', linewidth=2, capsize=5, capthick=2)
axes[2,1].axhline(y=0, color='#C73E1D', linestyle='--', linewidth=2, alpha=0.7)
axes[2,1].fill_between(range(len(growth_rates)), growth_rates, 0, 
                       where=np.array(growth_rates) > 0, alpha=0.3, color='green', label='Positive Growth')
axes[2,1].fill_between(range(len(growth_rates)), growth_rates, 0, 
                       where=np.array(growth_rates) < 0, alpha=0.3, color='red', label='Negative Growth')

axes[2,1].set_title('Month-over-Month Growth Rates', fontweight='bold', fontsize=12)
axes[2,1].set_xlabel('Month', fontweight='bold')
axes[2,1].set_ylabel('Growth Rate (%)', fontweight='bold')
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

# (3,3) Time series decomposition simulation
monthly_ts = monthly_spending['sum'].values
x = np.arange(len(monthly_ts))

# Trend component (polynomial fit)
trend = np.polyval(np.polyfit(x, monthly_ts, 2), x)

# Seasonal component (simplified)
seasonal = np.sin(2 * np.pi * x / 12) * np.std(monthly_ts) * 0.3

# Residual component
residual = monthly_ts - trend - seasonal

# Plot components
axes[2,2].plot(x, monthly_ts, 'o-', color='#2E86AB', linewidth=2, label='Original', alpha=0.8)
axes[2,2].plot(x, trend, '-', color='#C73E1D', linewidth=3, label='Trend')
axes[2,2].plot(x, seasonal + np.mean(monthly_ts), '-', color='#F18F01', linewidth=2, label='Seasonal')
axes[2,2].plot(x, residual + np.mean(monthly_ts), '-', color='#6A994E', linewidth=1, alpha=0.7, label='Residual')

axes[2,2].set_title('Time Series Decomposition', fontweight='bold', fontsize=12)
axes[2,2].set_xlabel('Time Period', fontweight='bold')
axes[2,2].set_ylabel('Amount (₹)', fontweight='bold')
axes[2,2].legend()
axes[2,2].grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.savefig('credit_card_spending_analysis.png', dpi=300, bbox_inches='tight')
plt.show()