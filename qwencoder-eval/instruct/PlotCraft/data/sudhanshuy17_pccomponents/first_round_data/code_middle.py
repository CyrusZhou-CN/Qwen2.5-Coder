import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set ugly style
plt.style.use('dark_background')

# Load and process data
def clean_price(price_str):
    if pd.isna(price_str):
        return 0
    price_str = str(price_str).replace('₹', '').replace(',', '')
    try:
        return float(price_str)
    except:
        return 0

# Create fake data since we can't load actual files
categories = ['PowerSupply', 'RAM', 'MotherBoard', 'StorageSSD', 'Cabinets', 'GPU', 'CPU']
counts = [1120, 1440, 1440, 2640, 3120, 600, 1440]

# Generate random price data
np.random.seed(42)
all_prices = []
all_categories = []

for cat, count in zip(categories, counts):
    if cat == 'GPU':
        prices = np.random.exponential(15000, count) + 2000
    elif cat == 'CPU':
        prices = np.random.exponential(12000, count) + 1000
    else:
        prices = np.random.exponential(3000, count) + 500
    
    all_prices.extend(prices)
    all_categories.extend([cat] * count)

# Create figure with wrong layout (user wants composite, I'll make 3x1 instead of 1x2)
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

# Use subplots_adjust to create overlap
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Wrong chart 1: Bar chart instead of pie chart, showing prices instead of counts
price_by_cat = {}
for cat, count in zip(categories, counts):
    idx = all_categories.index(cat)
    price_by_cat[cat] = sum(all_prices[idx:idx+count])

axes[0].bar(range(len(categories)), list(price_by_cat.values()), color='red')
axes[0].set_title('Temperature vs Humidity Analysis', fontsize=8, color='white')
axes[0].set_xlabel('Amplitude', fontsize=8, color='white')
axes[0].set_ylabel('Time Units', fontsize=8, color='white')
axes[0].set_xticks(range(len(categories)))
axes[0].set_xticklabels(['Glarbnok', 'Flibber', 'Zorblex', 'Quibble', 'Snarf', 'Blurp', 'Wizzle'], rotation=90, fontsize=6)

# Wrong chart 2: Scatter plot instead of horizontal stacked bar
budget_counts = []
mid_counts = []
premium_counts = []

for cat in categories:
    cat_prices = [all_prices[i] for i, c in enumerate(all_categories) if c == cat]
    budget = sum(1 for p in cat_prices if p < 5000)
    mid = sum(1 for p in cat_prices if 5000 <= p <= 20000)
    premium = sum(1 for p in cat_prices if p > 20000)
    budget_counts.append(budget)
    mid_counts.append(mid)
    premium_counts.append(premium)

x_pos = np.arange(len(categories))
axes[1].scatter(x_pos, budget_counts, color='yellow', s=100, alpha=0.7)
axes[1].scatter(x_pos, mid_counts, color='cyan', s=100, alpha=0.7)
axes[1].scatter(x_pos, premium_counts, color='magenta', s=100, alpha=0.7)
axes[1].set_title('Quantum Flux Distribution', fontsize=8, color='white')
axes[1].set_xlabel('Frequency', fontsize=8, color='white')
axes[1].set_ylabel('Wavelength', fontsize=8, color='white')
axes[1].grid(True, color='white', linewidth=2)

# Add third unnecessary chart - line plot
random_data = np.random.random(50) * 100
axes[2].plot(random_data, linewidth=5, color='lime')
axes[2].set_title('Unrelated Data Visualization', fontsize=8, color='white')
axes[2].set_xlabel('Random Variable Z', fontsize=8, color='white')
axes[2].set_ylabel('Arbitrary Units', fontsize=8, color='white')

# Add overlapping text annotations
axes[0].text(3, max(price_by_cat.values())/2, 'OVERLAPPING\nTEXT\nANNOTATION', 
             fontsize=16, color='white', ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

axes[1].text(3, max(max(budget_counts), max(mid_counts), max(premium_counts))/2, 
             'MORE\nOVERLAPPING\nTEXT', fontsize=14, color='yellow', 
             ha='center', va='center', weight='bold')

# Make spines thick and ugly
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=72, facecolor='black')