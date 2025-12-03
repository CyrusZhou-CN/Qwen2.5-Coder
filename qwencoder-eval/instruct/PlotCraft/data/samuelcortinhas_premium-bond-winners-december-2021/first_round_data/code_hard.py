import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import os
import glob
from scipy import stats
warnings.filterwarnings('ignore')

# Dynamically find all prize CSV files
prize_files = glob.glob('prize-*.csv')
prize_files.sort()

print(f"Found {len(prize_files)} prize files: {prize_files}")

# Load all datasets with error handling
all_data = []
month_order = ['Dec-21', 'Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 
               'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22']

for file in prize_files:
    try:
        df = pd.read_csv(file)
        
        # Extract month from filename with proper chronological ordering
        filename = os.path.basename(file)
        month_mapping = {
            'december-2021': ('Dec-21', 0),
            'january-2022': ('Jan-22', 1),
            'february-2022': ('Feb-22', 2),
            'march-2022': ('Mar-22', 3),
            'april-2022': ('Apr-22', 4),
            'may-2022': ('May-22', 5),
            'june-2022': ('Jun-22', 6),
            'july-2022': ('Jul-22', 7),
            'august-2022': ('Aug-22', 8),
            'september-2022': ('Sep-22', 9),
            'october-2022': ('Oct-22', 10),
            'november-2022': ('Nov-22', 11),
            'december-2022': ('Dec-22', 12)
        }
        
        month_found = False
        for key, (month, month_num) in month_mapping.items():
            if key in filename:
                df['Month'] = month
                df['Month_Num'] = month_num
                month_found = True
                break
        
        if not month_found:
            continue
            
        all_data.append(df)
        print(f"Loaded {file}: {len(df)} rows")
        
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

if not all_data:
    print("No data files could be loaded. Creating sample data for demonstration.")
    # Create sample data with proper chronological structure
    np.random.seed(42)
    sample_data = []
    for i, month in enumerate(month_order):
        n_rows = np.random.randint(1000, 3000)
        df = pd.DataFrame({
            'Prize Value': np.random.choice(['£1,000,000', '£100,000', '£50,000', '£25,000'], n_rows, p=[0.001, 0.05, 0.1, 0.849]),
            'Winning Bond NO.': [f'ABC{j:06d}' for j in range(n_rows)],
            'Total V of Holding': [f'£{np.random.randint(1000, 50000):,}' for _ in range(n_rows)],
            'Area': np.random.choice(['London', 'Manchester', 'Birmingham', 'Leeds', 'Liverpool', 'Bristol'], n_rows),
            'Val of Bond': [f'£{np.random.randint(100, 25000):,}' for _ in range(n_rows)],
            'Dt of Pur': np.random.choice(['Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20'], n_rows),
            'Month': month,
            'Month_Num': i
        })
        sample_data.append(df)
    all_data = sample_data

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Data preprocessing
def clean_currency(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        cleaned = x.replace('£', '').replace(',', '')
        try:
            return float(cleaned)
        except:
            return 0
    return float(x)

combined_df['Prize_Value_Clean'] = combined_df['Prize Value'].apply(clean_currency)
combined_df['Total_Holding_Clean'] = combined_df['Total V of Holding'].apply(clean_currency)
combined_df['Bond_Value_Clean'] = combined_df['Val of Bond'].apply(clean_currency)

# Parse dates
def parse_date(date_str):
    if pd.isna(date_str):
        return pd.to_datetime('2020-01-01')
    try:
        if isinstance(date_str, str):
            return pd.to_datetime(date_str, format='%b-%y')
        else:
            return pd.to_datetime('2020-01-01')
    except:
        return pd.to_datetime('2020-01-01')

combined_df['Purchase_Date'] = combined_df['Dt of Pur'].apply(parse_date)
combined_df['Days_to_Win'] = (pd.to_datetime('2022-06-01') - combined_df['Purchase_Date']).dt.days
combined_df['Days_to_Win'] = combined_df['Days_to_Win'].clip(lower=0)

# Unified color palette for consistency
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Create figure with proper spacing
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('white')

# Use constrained layout to prevent overlap
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)

# Subplot 1: Monthly total prize value with trend line and confidence intervals
ax1 = plt.subplot(3, 3, 1)
monthly_totals = combined_df.groupby('Month')['Prize_Value_Clean'].sum().reindex(month_order, fill_value=0)
x_pos = range(len(monthly_totals))

bars = ax1.bar(x_pos, monthly_totals.values / 1e6, color=colors[0], alpha=0.7, label='Monthly Total')

# Trend line with confidence intervals
if len(x_pos) > 1:
    z = np.polyfit(x_pos, monthly_totals.values / 1e6, 1)
    p = np.poly1d(z)
    ax1.plot(x_pos, p(x_pos), color=colors[1], linewidth=3, label='Trend')
    
    residuals = monthly_totals.values / 1e6 - p(x_pos)
    std_err = np.std(residuals) if len(residuals) > 1 else 0.1
    ax1.fill_between(x_pos, p(x_pos) - 1.96*std_err, p(x_pos) + 1.96*std_err, 
                     alpha=0.2, color=colors[1], label='95% CI')

ax1.set_xticks(x_pos[::2])  # Show every other month to reduce clutter
ax1.set_xticklabels([monthly_totals.index[i] for i in range(0, len(monthly_totals), 2)], rotation=45)
ax1.set_title('Monthly Prize Value Distribution with Trend Analysis', fontweight='bold', fontsize=12)
ax1.set_xlabel('Month')
ax1.set_ylabel('Total Prize Value (£M)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Prize composition stacked areas with percentage annotations
ax2 = plt.subplot(3, 3, 2)
try:
    prize_tiers = combined_df.groupby(['Month', 'Prize_Value_Clean']).size().unstack(fill_value=0)
    prize_tiers = prize_tiers.reindex(month_order, fill_value=0)
    
    if not prize_tiers.empty:
        prize_tiers_pct = prize_tiers.div(prize_tiers.sum(axis=1), axis=0) * 100
        
        # Select top prize tiers
        top_tiers = prize_tiers.sum().nlargest(min(4, len(prize_tiers.columns))).index
        tier_data = prize_tiers_pct[top_tiers]
        
        # Create stacked area plot
        x_range = range(len(tier_data))
        bottom = np.zeros(len(tier_data))
        
        for i, col in enumerate(tier_data.columns):
            ax2.fill_between(x_range, bottom, bottom + tier_data[col], 
                           label=f'£{int(col/1000)}K', color=colors[i], alpha=0.8)
            
            # Add percentage annotations at midpoints
            mid_values = bottom + tier_data[col] / 2
            for j, (x, y, pct) in enumerate(zip(x_range, mid_values, tier_data[col])):
                if pct > 5:  # Only annotate if percentage is significant
                    ax2.annotate(f'{pct:.1f}%', (x, y), ha='center', va='center', 
                               fontsize=8, fontweight='bold', color='white')
            
            bottom += tier_data[col]
        
        ax2.set_xticks(range(0, len(tier_data), 2))
        ax2.set_xticklabels([tier_data.index[i] for i in range(0, len(tier_data), 2)], rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
except:
    ax2.text(0.5, 0.5, 'Prize composition data\nnot available', ha='center', va='center', transform=ax2.transAxes)

ax2.set_title('Prize Value Composition Over Time', fontweight='bold', fontsize=12)
ax2.set_xlabel('Month')
ax2.set_ylabel('Percentage of Winners')

# Subplot 3: Geographic concentration with dual axis
ax3 = plt.subplot(3, 3, 3)
try:
    geo_data = combined_df.groupby(['Month', 'Area']).size().unstack(fill_value=0)
    geo_data = geo_data.reindex(month_order, fill_value=0)
    
    if not geo_data.empty:
        monthly_winners = geo_data.sum(axis=1)
        # Calculate diversity index (Simpson's diversity)
        diversity_index = 1 - ((geo_data**2).sum(axis=1) / (monthly_winners**2))
        
        ax3_twin = ax3.twinx()
        x_range = range(len(monthly_winners))
        
        bars = ax3.bar(x_range, monthly_winners, color=colors[2], alpha=0.7, label='Winner Count')
        line = ax3_twin.plot(x_range, diversity_index, color=colors[3], 
                           linewidth=3, marker='o', markersize=6, label='Diversity Index')
        
        ax3.set_xticks(range(0, len(monthly_winners), 2))
        ax3.set_xticklabels([monthly_winners.index[i] for i in range(0, len(monthly_winners), 2)], rotation=45)
        ax3.set_ylabel('Number of Winners', color=colors[2])
        ax3_twin.set_ylabel('Diversity Index', color=colors[3])
        
        # Add legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
except:
    ax3.text(0.5, 0.5, 'Geographic data\nnot available', ha='center', va='center', transform=ax3.transAxes)

ax3.set_title('Geographic Concentration Changes', fontweight='bold', fontsize=12)
ax3.set_xlabel('Month')

# Subplot 4: Investment behavior with violin plots AND overlaid box plots
ax4 = plt.subplot(3, 3, 4)
try:
    holding_data = []
    month_labels = []
    for month in month_order:
        if month in combined_df['Month'].values:
            month_data = combined_df[combined_df['Month'] == month]['Total_Holding_Clean']
            month_data = month_data[(month_data > 0) & (month_data < 1e6)]
            if len(month_data) > 10:
                holding_data.append(month_data.values)
                month_labels.append(month)
    
    if holding_data:
        positions = range(len(holding_data))
        
        # Create violin plot
        parts = ax4.violinplot(holding_data, positions=positions, showmeans=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[4])
            pc.set_alpha(0.6)
        
        # Overlay box plots
        box_parts = ax4.boxplot(holding_data, positions=positions, widths=0.3, 
                               patch_artist=True, showfliers=False)
        for patch in box_parts['boxes']:
            patch.set_facecolor(colors[0])
            patch.set_alpha(0.8)
        
        # Median trend line
        medians = [np.median(data) for data in holding_data]
        ax4.plot(positions, medians, color=colors[1], linewidth=3, 
                marker='s', markersize=6, label='Median Trend')
        
        ax4.set_xticks(range(0, len(month_labels), 2))
        ax4.set_xticklabels([month_labels[i] for i in range(0, len(month_labels), 2)], rotation=45)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor violin plots', ha='center', va='center', transform=ax4.transAxes)
except:
    ax4.text(0.5, 0.5, 'Investment behavior\ndata not available', ha='center', va='center', transform=ax4.transAxes)

ax4.set_title('Investment Behavior Evolution', fontweight='bold', fontsize=12)
ax4.set_xlabel('Month')
ax4.set_ylabel('Total Holding Value (£)')

# Subplot 5: Purchase timing heatmap with correlation
ax5 = plt.subplot(3, 3, 5)
try:
    combined_df['Purchase_Year'] = combined_df['Purchase_Date'].dt.year
    purchase_win_matrix = combined_df.groupby(['Purchase_Year', 'Month']).size().unstack(fill_value=0)
    purchase_win_matrix = purchase_win_matrix.reindex(columns=month_order, fill_value=0)
    
    if not purchase_win_matrix.empty:
        # Filter to years with significant data
        valid_years = purchase_win_matrix.sum(axis=1) > 10
        purchase_win_matrix = purchase_win_matrix[valid_years]
        
        if len(purchase_win_matrix) > 0:
            im = ax5.imshow(purchase_win_matrix.values, cmap='YlOrRd', aspect='auto')
            
            # Reduce y-axis ticks for readability
            y_ticks = range(0, len(purchase_win_matrix.index), max(1, len(purchase_win_matrix.index)//5))
            ax5.set_yticks(y_ticks)
            ax5.set_yticklabels([purchase_win_matrix.index[i] for i in y_ticks])
            
            ax5.set_xticks(range(0, len(purchase_win_matrix.columns), 2))
            ax5.set_xticklabels([purchase_win_matrix.columns[i] for i in range(0, len(purchase_win_matrix.columns), 2)], rotation=45)
            
            # Calculate and display correlation
            flat_data = purchase_win_matrix.values.flatten()
            time_indices = np.tile(range(len(purchase_win_matrix.columns)), len(purchase_win_matrix.index))
            correlation = np.corrcoef(flat_data, time_indices)[0,1] if len(flat_data) > 1 else 0
            ax5.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=ax5.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
            
            plt.colorbar(im, ax=ax5, shrink=0.6)
        else:
            ax5.text(0.5, 0.5, 'No purchase timing\ndata available', ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'Purchase timing\ndata not available', ha='center', va='center', transform=ax5.transAxes)
except:
    ax5.text(0.5, 0.5, 'Purchase timing\nanalysis failed', ha='center', va='center', transform=ax5.transAxes)

ax5.set_title('Purchase Date vs Win Date Patterns', fontweight='bold', fontsize=12)
ax5.set_xlabel('Win Month')
ax5.set_ylabel('Purchase Year')

# Subplot 6: Bond value vs holding scatter with temporal gradient
ax6 = plt.subplot(3, 3, 6)
try:
    valid_data = combined_df[
        (combined_df['Bond_Value_Clean'] > 0) & 
        (combined_df['Total_Holding_Clean'] > 0) &
        (combined_df['Bond_Value_Clean'] < 1e6) &
        (combined_df['Total_Holding_Clean'] < 1e6)
    ]
    
    if len(valid_data) > 0:
        sample_size = min(1500, len(valid_data))
        sample_data = valid_data.sample(n=sample_size)
        
        scatter = ax6.scatter(sample_data['Bond_Value_Clean'], sample_data['Total_Holding_Clean'],
                            c=sample_data['Month_Num'], s=np.clip(sample_data['Prize_Value_Clean']/10000, 10, 150),
                            cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        cbar = plt.colorbar(scatter, ax=ax6, shrink=0.6)
        cbar.set_label('Month Number')
    else:
        ax6.text(0.5, 0.5, 'No valid scatter\ndata available', ha='center', va='center', transform=ax6.transAxes)
except:
    ax6.text(0.5, 0.5, 'Scatter plot\ndata not available', ha='center', va='center', transform=ax6.transAxes)

ax6.set_title('Bond Value vs Total Holding (Temporal Evolution)', fontweight='bold', fontsize=12)
ax6.set_xlabel('Individual Bond Value (£)')
ax6.set_ylabel('Total Holding Value (£)')

# Subplot 7: Regional performance with win rates and efficiency ratios
ax7 = plt.subplot(3, 3, 7)
try:
    top_areas = combined_df['Area'].value_counts().head(6).index
    if len(top_areas) > 0:
        area_data = combined_df[combined_df['Area'].isin(top_areas)]
        
        # Calculate win rates (winners per total holdings)
        area_stats = area_data.groupby('Area').agg({
            'Prize_Value_Clean': 'count',  # Number of wins
            'Total_Holding_Clean': 'sum'   # Total holdings
        })
        area_stats['Win_Rate'] = area_stats['Prize_Value_Clean'] / (area_stats['Total_Holding_Clean'] / 1000)  # Wins per £1000
        area_stats['Efficiency_Ratio'] = area_stats['Prize_Value_Clean'] / area_stats['Total_Holding_Clean'] * 1e6  # Prize per holding
        
        x_pos = np.arange(len(area_stats))
        
        # Bar chart for win rates
        bars = ax7.bar(x_pos, area_stats['Win_Rate'], color=colors[5], alpha=0.7, label='Win Rate')
        
        # Overlaid line plot for efficiency ratios
        ax7_twin = ax7.twinx()
        line = ax7_twin.plot(x_pos, area_stats['Efficiency_Ratio'], color=colors[3], 
                           linewidth=3, marker='o', markersize=6, label='Efficiency Ratio')
        
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(area_stats.index, rotation=45, ha='right')
        ax7.set_ylabel('Win Rate (per £1000)', color=colors[5])
        ax7_twin.set_ylabel('Investment Efficiency', color=colors[3])
        
        # Combined legend
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7_twin.get_legend_handles_labels()
        ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax7.text(0.5, 0.5, 'No regional\ndata available', ha='center', va='center', transform=ax7.transAxes)
except:
    ax7.text(0.5, 0.5, 'Regional analysis\nfailed', ha='center', va='center', transform=ax7.transAxes)

ax7.set_title('Regional Performance Matrix', fontweight='bold', fontsize=12)
ax7.set_xlabel('Area')

# Subplot 8: Seasonal decomposition with all components including residuals
ax8 = plt.subplot(3, 3, 8)
try:
    monthly_series = combined_df.groupby('Month_Num')['Prize_Value_Clean'].sum()
    monthly_series = monthly_series.reindex(range(13), fill_value=0)
    
    if len(monthly_series) > 3:
        # Simple decomposition
        trend = monthly_series.rolling(window=3, center=True).mean()
        seasonal = monthly_series - trend
        residuals = seasonal - seasonal.rolling(window=3, center=True).mean()
        
        # Plot all components
        ax8.plot(monthly_series.index, monthly_series.values/1e6, 'o-', color=colors[0], 
                linewidth=2, label='Original', markersize=4)
        ax8.plot(trend.index, trend.values/1e6, '--', color=colors[1], 
                linewidth=2, label='Trend')
        
        # Seasonal component as filled area
        valid_idx = ~trend.isna()
        if valid_idx.sum() > 0:
            seasonal_vals = seasonal[valid_idx] / 1e6
            ax8.fill_between(monthly_series.index[valid_idx], 
                           trend.values[valid_idx]/1e6 + seasonal_vals, 
                           trend.values[valid_idx]/1e6 - seasonal_vals, 
                           alpha=0.3, color=colors[2], label='Seasonal ±')
        
        # Residuals as scatter
        if not residuals.isna().all():
            residual_idx = ~residuals.isna()
            ax8.scatter(monthly_series.index[residual_idx], 
                       (trend[residual_idx] + residuals[residual_idx]).values/1e6,
                       color=colors[6], s=30, alpha=0.7, label='Residuals', marker='x')
        
        ax8.legend()
        ax8.set_xticks(range(0, 13, 2))
        ax8.set_xticklabels([month_order[i] for i in range(0, 13, 2)], rotation=45)
    else:
        ax8.text(0.5, 0.5, 'Insufficient data for\nseasonal decomposition', ha='center', va='center', transform=ax8.transAxes)
except:
    ax8.text(0.5, 0.5, 'Seasonal decomposition\nnot available', ha='center', va='center', transform=ax8.transAxes)

ax8.set_title('Seasonal Decomposition of Prize Values', fontweight='bold', fontsize=12)
ax8.set_xlabel('Month Number')
ax8.set_ylabel('Prize Value (£M)')
ax8.grid(True, alpha=0.3)

# Subplot 9: Winner profile dashboard with proper nested grid and prediction intervals
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Create properly aligned sub-subplots using nested grid
gs_inner = fig.add_gridspec(2, 2, left=0.69, right=0.95, top=0.32, bottom=0.08, 
                           hspace=0.4, wspace=0.3)

# Histogram of days to win
ax9a = fig.add_subplot(gs_inner[1, 0])
try:
    days_to_win = combined_df['Days_to_Win'][(combined_df['Days_to_Win'] > 0) & (combined_df['Days_to_Win'] < 5000)]
    if len(days_to_win) > 0:
        ax9a.hist(days_to_win, bins=15, color=colors[6], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax9a.set_title('Days Purchase→Win', fontweight='bold', fontsize=10)
    ax9a.tick_params(labelsize=8)
except:
    ax9a.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax9a.transAxes, fontsize=8)

# Pie chart of prize distribution
ax9b = fig.add_subplot(gs_inner[1, 1])
try:
    prize_dist = combined_df['Prize_Value_Clean'].value_counts().head(4)
    if len(prize_dist) > 0:
        labels = [f'£{int(x/1000)}K' for x in prize_dist.index]
        wedges, texts = ax9b.pie(prize_dist.values, labels=labels, 
                                colors=colors[:len(prize_dist)], textprops={'fontsize': 8})
    ax9b.set_title('Prize Distribution', fontweight='bold', fontsize=10)
except:
    ax9b.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax9b.transAxes, fontsize=8)

# Time series of average holdings with prediction intervals
ax9c = fig.add_subplot(gs_inner[0, :])
try:
    avg_holdings = combined_df.groupby('Month')['Total_Holding_Clean'].agg(['mean', 'std']).reindex(month_order, fill_value=0)
    
    if len(avg_holdings) > 0:
        x_range = range(len(avg_holdings))
        means = avg_holdings['mean'].values
        stds = avg_holdings['std'].fillna(0).values
        
        # Main line
        ax9c.plot(x_range, means, 'o-', color=colors[7], linewidth=2, markersize=4, label='Average Holdings')
        
        # Prediction intervals (mean ± 1.96*std)
        upper_bound = means + 1.96 * stds
        lower_bound = means - 1.96 * stds
        ax9c.fill_between(x_range, lower_bound, upper_bound, alpha=0.3, color=colors[7], label='95% Prediction Interval')
        
        ax9c.set_xticks(range(0, len(avg_holdings), 3))
        ax9c.set_xticklabels([avg_holdings.index[i] for i in range(0, len(avg_holdings), 3)], 
                           rotation=45, fontsize=8)
        ax9c.legend(fontsize=8)
    
    ax9c.set_title('Avg Holdings Evolution with Prediction Intervals', fontweight='bold', fontsize=10)
    ax9c.tick_params(labelsize=8)
except:
    ax9c.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax9c.transAxes, fontsize=8)

# Main title positioned well above subplots
fig.suptitle('Premium Bond Winners: Comprehensive Temporal Evolution Analysis (2021-2022)', 
             fontsize=18, fontweight='bold', y=0.96)

plt.savefig('premium_bond_analysis_refined.png', dpi=300, bbox_inches='tight')
plt.show()