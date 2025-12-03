import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('Console_Data.csv')

# Clean and prepare data
df['Discontinuation Year'] = pd.to_numeric(df['Discontinuation Year'], errors='coerce')
df['Console Lifespan'] = df['Discontinuation Year'] - df['Released Year']
df['Console Lifespan'] = df['Console Lifespan'].fillna(2024 - df['Released Year'])  # For ongoing consoles

# Create generation mapping
gen_mapping = {1: '1st Gen', 2: '2nd Gen', 3: '3rd Gen', 4: '4th Gen', 5: '5th Gen', 6: '6th Gen', 7: '7th Gen', 8: '8th Gen'}
df['Gen_Label'] = df['Generation'].map(gen_mapping)

# Set up the 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Color palettes
gen_colors = plt.cm.Set3(np.linspace(0, 1, len(df['Generation'].unique())))
company_colors = {'Nintendo': '#E60012', 'Sony': '#003087', 'Microsoft': '#00BCF2', 
                 'Sega': '#1E90FF', 'Atari': '#FF6B35', 'Magnavox': '#8B4513', 'Mattel': '#FF1493'}

# Subplot 1: Stacked area chart with market share lines
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Prepare data for stacked area
years = range(1972, 2025)
gen_data = {}
for gen in sorted(df['Generation'].unique()):
    gen_consoles = df[df['Generation'] == gen]
    yearly_sales = []
    for year in years:
        active_consoles = gen_consoles[
            (gen_consoles['Released Year'] <= year) & 
            (gen_consoles['Discontinuation Year'].fillna(2024) >= year)
        ]
        yearly_sales.append(active_consoles['Units sold (million)'].sum())
    gen_data[f'Gen {gen}'] = yearly_sales

# Create stacked area
bottom = np.zeros(len(years))
for i, (gen, sales) in enumerate(gen_data.items()):
    ax1.fill_between(years, bottom, bottom + sales, alpha=0.7, 
                    color=gen_colors[i], label=gen)
    bottom += sales

# Add market share lines
ax1_twin = ax1.twinx()
for i, (gen, sales) in enumerate(gen_data.items()):
    market_share = np.array(sales) / (bottom + 1e-6) * 100
    ax1_twin.plot(years, market_share, color=gen_colors[i], linewidth=2, alpha=0.8)

ax1.set_title('Console Market Evolution: Cumulative Sales & Market Share', fontweight='bold', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Cumulative Units Sold (millions)')
ax1_twin.set_ylabel('Market Share (%)')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Bar chart with scatter overlay
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

gen_stats = df.groupby('Generation').agg({
    'Units sold (million)': 'sum',
    'Console Lifespan': 'mean'
}).reset_index()

bars = ax2.bar(gen_stats['Generation'], gen_stats['Units sold (million)'], 
               color=gen_colors[:len(gen_stats)], alpha=0.7, label='Total Sales')

ax2_twin = ax2.twinx()
scatter = ax2_twin.scatter(gen_stats['Generation'], gen_stats['Console Lifespan'], 
                          color='red', s=100, alpha=0.8, label='Avg Lifespan', zorder=5)

ax2.set_title('Generation Performance: Sales vs Console Lifespan', fontweight='bold', fontsize=12)
ax2.set_xlabel('Generation')
ax2.set_ylabel('Total Units Sold (millions)')
ax2_twin.set_ylabel('Average Lifespan (years)')
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Subplot 3: Box plots with line chart
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Box plots for sales distribution
sales_by_gen = [df[df['Generation'] == gen]['Units sold (million)'].values 
                for gen in sorted(df['Generation'].unique())]
box_plot = ax3.boxplot(sales_by_gen, positions=sorted(df['Generation'].unique()), 
                       patch_artist=True, widths=0.6)

for patch, color in zip(box_plot['boxes'], gen_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Line chart for number of consoles per generation
ax3_twin = ax3.twinx()
console_counts = df.groupby('Generation').size()
ax3_twin.plot(console_counts.index, console_counts.values, 
              color='darkred', marker='o', linewidth=3, markersize=8, label='Console Count')

ax3.set_title('Sales Distribution & Console Release Patterns', fontweight='bold', fontsize=12)
ax3.set_xlabel('Generation')
ax3.set_ylabel('Units Sold (millions)')
ax3_twin.set_ylabel('Number of Consoles Released')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Subplot 4: Stacked bar with line overlay
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# Market share by company across generations
company_gen_data = df.groupby(['Generation', 'Company'])['Units sold (million)'].sum().unstack(fill_value=0)
company_gen_data.plot(kind='bar', stacked=True, ax=ax4, 
                     color=[company_colors.get(col, plt.cm.tab10(i)) 
                           for i, col in enumerate(company_gen_data.columns)])

# Add trend lines for major companies
ax4_twin = ax4.twinx()
for company in ['Nintendo', 'Sony', 'Sega', 'Atari']:
    if company in company_gen_data.columns:
        trend_data = company_gen_data[company]
        ax4_twin.plot(range(len(trend_data)), trend_data.values, 
                     marker='o', linewidth=2, label=f'{company} Trend',
                     color=company_colors.get(company, 'black'))

ax4.set_title('Company Market Share Evolution', fontweight='bold', fontsize=12)
ax4.set_xlabel('Generation')
ax4.set_ylabel('Market Share (millions)')
ax4_twin.set_ylabel('Individual Company Sales')
ax4.legend(loc='upper left', fontsize=8)
ax4_twin.legend(loc='upper right', fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Bubble chart with trend lines
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

for company in df['Company'].unique():
    company_data = df[df['Company'] == company]
    color = company_colors.get(company, plt.cm.tab10(hash(company) % 10))
    
    scatter = ax5.scatter(company_data['Released Year'], 
                         company_data['Units sold (million)'],
                         s=company_data['Console Lifespan'] * 20,
                         alpha=0.6, color=color, label=company, edgecolors='black')
    
    # Add trend line
    if len(company_data) > 1:
        z = np.polyfit(company_data['Released Year'], company_data['Units sold (million)'], 1)
        p = np.poly1d(z)
        ax5.plot(company_data['Released Year'], p(company_data['Released Year']), 
                color=color, linestyle='--', alpha=0.8)

ax5.set_title('Console Performance: Sales vs Release Year\n(Bubble size = Lifespan)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Release Year')
ax5.set_ylabel('Units Sold (millions)')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 6: Radar chart with bar overlay
ax6 = plt.subplot(3, 3, 6, projection='polar')
ax6.set_facecolor('white')

# Prepare radar chart data
companies = ['Nintendo', 'Sony', 'Sega', 'Atari']
metrics = ['Total Sales', 'Console Count', 'Avg Lifespan']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, company in enumerate(companies):
    if company in df['Company'].values:
        company_data = df[df['Company'] == company]
        values = [
            company_data['Units sold (million)'].sum() / 100,  # Normalized
            len(company_data) / 5,  # Normalized
            company_data['Console Lifespan'].mean() / 20  # Normalized
        ]
        values += values[:1]  # Complete the circle
        
        color = company_colors.get(company, plt.cm.tab10(i))
        ax6.plot(angles, values, 'o-', linewidth=2, label=company, color=color)
        ax6.fill(angles, values, alpha=0.25, color=color)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics)
ax6.set_title('Company Performance Radar\n(Normalized Metrics)', fontweight='bold', fontsize=12, pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
ax6.grid(True)

# Subplot 7: Heatmap with line overlay
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Create release intensity heatmap
year_gen_matrix = df.groupby(['Released Year', 'Generation']).size().unstack(fill_value=0)
im = ax7.imshow(year_gen_matrix.T, cmap='YlOrRd', aspect='auto', alpha=0.8)

# Add cumulative market growth line
ax7_twin = ax7.twinx()
cumulative_sales = df.groupby('Released Year')['Units sold (million)'].sum().cumsum()
ax7_twin.plot(range(len(cumulative_sales)), cumulative_sales.values, 
              color='blue', linewidth=3, label='Cumulative Market Growth')

ax7.set_title('Console Release Intensity & Market Growth', fontweight='bold', fontsize=12)
ax7.set_xlabel('Release Year Index')
ax7.set_ylabel('Generation')
ax7_twin.set_ylabel('Cumulative Sales (millions)')
ax7_twin.legend(loc='upper left')

# Subplot 8: Slope chart with area background
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Background area chart showing overall market expansion
gen_totals = df.groupby('Generation')['Units sold (million)'].sum()
ax8.fill_between(gen_totals.index, 0, gen_totals.values, alpha=0.3, color='lightblue')

# Slope chart for company performance between generations
for company in ['Nintendo', 'Sony', 'Sega']:
    if company in df['Company'].values:
        company_gen_sales = df[df['Company'] == company].groupby('Generation')['Units sold (million)'].sum()
        if len(company_gen_sales) > 1:
            ax8.plot(company_gen_sales.index, company_gen_sales.values, 
                    marker='o', linewidth=2, markersize=8, 
                    label=company, color=company_colors.get(company, 'black'))

ax8.set_title('Company Performance Across Generations', fontweight='bold', fontsize=12)
ax8.set_xlabel('Generation')
ax8.set_ylabel('Sales (millions)')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Subplot 9: Multi-layered area chart with scatter overlay
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Console lifecycle overlaps
years_extended = range(1970, 2030)
lifecycle_data = np.zeros((len(df), len(years_extended)))

for i, (_, console) in enumerate(df.iterrows()):
    start_idx = console['Released Year'] - 1970
    end_idx = int(console['Discontinuation Year']) - 1970 if pd.notna(console['Discontinuation Year']) else 2024 - 1970
    if start_idx >= 0 and end_idx < len(years_extended):
        lifecycle_data[i, start_idx:end_idx] = console['Units sold (million)']

# Create stacked area for lifecycle overlaps
for i in range(len(df)):
    ax9.fill_between(years_extended, 0, lifecycle_data[i], alpha=0.1)

# Add scatter for peak sales years and trend line
peak_years = df['Released Year'] + df['Console Lifespan'] / 2
ax9.scatter(peak_years, df['Units sold (million)'], 
           s=100, alpha=0.7, color='red', label='Peak Sales Period')

# Market maturation trend line
z = np.polyfit(df['Released Year'], df['Units sold (million)'], 2)
p = np.poly1d(z)
trend_years = range(1972, 2025)
ax9.plot(trend_years, p(trend_years), color='darkblue', linewidth=3, 
         label='Market Maturation Trend')

ax9.set_title('Console Lifecycle Overlaps & Market Maturation', fontweight='bold', fontsize=12)
ax9.set_xlabel('Year')
ax9.set_ylabel('Sales Impact (millions)')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.show()