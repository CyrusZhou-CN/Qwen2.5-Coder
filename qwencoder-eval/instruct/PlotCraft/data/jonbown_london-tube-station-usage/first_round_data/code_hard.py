import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load datasets efficiently
years_data = {}
available_years = []

# Try to load each year's data
for year in [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]:
    try:
        df = pd.read_csv(f'{year}_Entry_Exit.csv')
        
        # Quick data cleaning
        numeric_cols = ['Entry_Week', 'Entry_Saturday', 'Entry_Sunday', 'AnnualEntryExit_Mill']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived columns
        df['Weekend_Activity'] = df['Entry_Saturday'].fillna(0) + df['Entry_Sunday'].fillna(0)
        df['Weekday_Weekend_Ratio'] = df['Entry_Week'] / (df['Weekend_Activity'] + 1)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Entry_Week', 'AnnualEntryExit_Mill'])
        
        if len(df) > 0:
            years_data[year] = df
            available_years.append(year)
            
    except Exception as e:
        print(f"Could not load {year}: {e}")

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color schemes
colors_early = ['#2E86AB', '#A23B72', '#F18F01']
colors_middle = ['#6A994E', '#A7C957', '#BC6C25']
colors_late = ['#F72585', '#B5179E', '#7209B7']

# Top row (2007-2009): Scatter plots
target_years_top = [2007, 2008, 2009]
for i in range(3):
    ax = axes[0, i]
    
    if i < len(target_years_top) and target_years_top[i] in years_data:
        year = target_years_top[i]
        df = years_data[year]
        
        # Sample data if too large to avoid timeout
        if len(df) > 200:
            df = df.sample(n=200, random_state=42)
        
        x = df['Entry_Week'].values
        y = df['AnnualEntryExit_Mill'].values
        sizes = np.clip(df['Weekend_Activity'].values / 200, 10, 200)
        
        # Scatter plot
        scatter = ax.scatter(x, y, s=sizes, alpha=0.6, c=colors_early[i], 
                           edgecolors='white', linewidth=0.5)
        
        # Simple regression line
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color='darkred', linewidth=2, alpha=0.8)
            
            # Correlation annotation
            ax.text(0.05, 0.95, f'r = {r_value:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, fontweight='bold')
        except:
            pass
            
        ax.set_title(f'{year} Usage Patterns', fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Year {target_years_top[i]} - No Data', fontsize=12)
        ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Weekday Entries', fontsize=10)
    ax.set_ylabel('Annual Usage (Millions)', fontsize=10)
    ax.grid(True, alpha=0.3)

# Middle row (2010-2012): Histograms with KDE
target_years_middle = [2010, 2011, 2012]
for i in range(3):
    ax = axes[1, i]
    
    if i < len(target_years_middle) and target_years_middle[i] in years_data:
        year = target_years_middle[i]
        df = years_data[year]
        usage_data = df['AnnualEntryExit_Mill'].dropna()
        
        if len(usage_data) > 0:
            # Histogram
            n, bins, patches = ax.hist(usage_data, bins=20, alpha=0.7,
                                     color=colors_middle[i], edgecolor='white')
            
            # KDE overlay
            try:
                kde = gaussian_kde(usage_data)
                x_range = np.linspace(usage_data.min(), usage_data.max(), 50)
                kde_values = kde(x_range)
                
                ax2 = ax.twinx()
                ax2.plot(x_range, kde_values, color='darkgreen', linewidth=2)
                ax2.set_ylabel('Density', fontsize=10)
                ax2.tick_params(labelsize=8)
            except:
                pass
            
            # Percentiles
            try:
                percentiles = [25, 50, 75]
                perc_values = np.percentile(usage_data, percentiles)
                colors_perc = ['orange', 'red', 'purple']
                
                for j, (perc, val, color) in enumerate(zip(percentiles, perc_values, colors_perc)):
                    ax.axvline(val, color=color, linestyle='--', linewidth=2, alpha=0.8)
                    ax.text(val, ax.get_ylim()[1] * 0.8, f'{perc}th: {val:.1f}M',
                           rotation=90, fontsize=8, ha='right')
            except:
                pass
                
        ax.set_title(f'{year} Usage Distribution', fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Year {target_years_middle[i]} - No Data', fontsize=12)
        ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Annual Usage (Millions)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3)

# Bottom row (2013-2015): Box plots by borough
target_years_bottom = [2013, 2014, 2015]
for i in range(3):
    ax = axes[2, i]
    
    if i < len(target_years_bottom) and target_years_bottom[i] in years_data:
        year = target_years_bottom[i]
        df = years_data[year]
        
        if 'Borough' in df.columns:
            # Get top boroughs
            borough_counts = df['Borough'].value_counts().head(8)  # Reduced for performance
            top_boroughs = borough_counts.index.tolist()
            
            borough_data = []
            borough_labels = []
            
            for borough in top_boroughs:
                borough_subset = df[df['Borough'] == borough]['Weekday_Weekend_Ratio'].dropna()
                if len(borough_subset) > 0:
                    borough_data.append(borough_subset.values)
                    borough_labels.append(borough[:8])  # Truncate names
            
            if borough_data:
                # Box plot
                bp = ax.boxplot(borough_data, labels=borough_labels, patch_artist=True)
                
                # Color boxes
                for patch in bp['boxes']:
                    patch.set_facecolor(colors_late[i])
                    patch.set_alpha(0.7)
                
                # Add median values
                medians = [np.median(data) for data in borough_data]
                for j, median_val in enumerate(medians):
                    ax.text(j+1, median_val, f'{median_val:.1f}', ha='center', va='bottom',
                           fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Borough', fontsize=10)
                ax.set_ylabel('Weekday/Weekend Ratio', fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax.text(0.5, 0.5, 'Insufficient Borough Data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
        else:
            # Fallback: simple ratio histogram
            ratios = df['Weekday_Weekend_Ratio'].dropna()
            if len(ratios) > 0:
                ax.hist(ratios, bins=15, alpha=0.7, color=colors_late[i], edgecolor='white')
                ax.set_xlabel('Weekday/Weekend Ratio', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Ratio Data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'{year} Borough Patterns', fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Year {target_years_bottom[i]} - No Data', fontsize=12)
        ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    
    ax.grid(True, alpha=0.3)

# Overall styling
fig.suptitle('London Underground Station Usage Evolution (2007-2015)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.subplots_adjust(hspace=0.35, wspace=0.3)

plt.savefig('london_tube_evolution.png', dpi=300, bbox_inches='tight')
plt.show()