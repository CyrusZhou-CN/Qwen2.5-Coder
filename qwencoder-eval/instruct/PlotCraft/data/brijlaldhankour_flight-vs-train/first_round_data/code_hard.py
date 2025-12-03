import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('trains vs planes.csv')

# Data preprocessing
df['date'] = pd.to_datetime(df['date '], format='%d-%m-%Y')
df = df.drop('date ', axis=1)

# Create figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 24), facecolor='white')
fig.patch.set_facecolor('white')

# Color palette
colors = {'Plane': '#2E86AB', 'Train': '#A23B72'}

# Subplot 1: Diverging bar chart with error bars and jitter plots
ax1 = plt.subplot(3, 2, 1, facecolor='white')

# Calculate mean differences and statistics
route_stats = []
for route in df['Route'].unique():
    route_data = df[df['Route'] == route]
    plane_prices = route_data[route_data['Mode'] == 'Plane']['Ticket Price']
    train_prices = route_data[route_data['Mode'] == 'Train']['Ticket Price']
    
    if len(plane_prices) > 0 and len(train_prices) > 0:
        mean_diff = plane_prices.mean() - train_prices.mean()
        std_diff = np.sqrt(plane_prices.var() + train_prices.var())
        
        route_stats.append({
            'Route': route,
            'Mean_Diff': mean_diff,
            'Std_Diff': std_diff,
            'Plane_Mean': plane_prices.mean(),
            'Train_Mean': train_prices.mean()
        })

route_stats_df = pd.DataFrame(route_stats)

# Create diverging bar chart
y_pos = np.arange(len(route_stats_df))
bars = ax1.barh(y_pos, route_stats_df['Mean_Diff'], 
                color=['#2E86AB' if x > 0 else '#A23B72' for x in route_stats_df['Mean_Diff']],
                alpha=0.7, height=0.6)

# Add error bars
ax1.errorbar(route_stats_df['Mean_Diff'], y_pos, 
             xerr=route_stats_df['Std_Diff'], fmt='none', 
             color='black', capsize=5, alpha=0.8)

# Add jitter plots
for i, route in enumerate(route_stats_df['Route']):
    route_data = df[df['Route'] == route]
    for mode in ['Plane', 'Train']:
        mode_data = route_data[route_data['Mode'] == mode]['Ticket Price']
        if len(mode_data) > 0:
            y_jitter = np.random.normal(i, 0.1, len(mode_data))
            ax1.scatter(mode_data, y_jitter, alpha=0.6, s=30, 
                       color=colors[mode], edgecolors='white', linewidth=0.5)

ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(route_stats_df['Route'])
ax1.set_xlabel('Price Difference (Plane - Train)', fontweight='bold')
ax1.set_title('Mean Price Differences with Individual Data Points', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3)

# Subplot 2: Dumbbell plot with area chart envelope
ax2 = plt.subplot(3, 2, 2, facecolor='white')

# Prepare data for dumbbell plot
weekly_data = df.groupby(['week', 'Mode'])['Ticket Price'].mean().unstack(fill_value=0)

# Ensure both modes exist
if 'Plane' not in weekly_data.columns:
    weekly_data['Plane'] = 0
if 'Train' not in weekly_data.columns:
    weekly_data['Train'] = 0

# Create area chart envelope (price range)
weeks = weekly_data.index
plane_prices = weekly_data['Plane']
train_prices = weekly_data['Train']

# Fill area between min and max prices
min_prices = np.minimum(plane_prices, train_prices)
max_prices = np.maximum(plane_prices, train_prices)
ax2.fill_between(weeks, min_prices, max_prices, alpha=0.2, color='gray', label='Price Range Envelope')

# Create dumbbell plot
for week in weeks:
    plane_price = plane_prices[week]
    train_price = train_prices[week]
    
    if plane_price > 0 and train_price > 0:
        # Connect with line
        ax2.plot([week, week], [train_price, plane_price], 'k-', alpha=0.6, linewidth=2)
        
        # Add points
        ax2.scatter(week, plane_price, color=colors['Plane'], s=80, zorder=5, 
                   edgecolors='white', linewidth=2, label='Plane' if week == weeks[0] else "")
        ax2.scatter(week, train_price, color=colors['Train'], s=80, zorder=5,
                   edgecolors='white', linewidth=2, label='Train' if week == weeks[0] else "")

ax2.set_xlabel('Week', fontweight='bold')
ax2.set_ylabel('Average Ticket Price', fontweight='bold')
ax2.set_title('Weekly Price Comparison with Range Envelope', fontweight='bold', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Slope chart with confidence intervals
ax3 = plt.subplot(3, 2, 3, facecolor='white')

# Calculate confidence intervals
for mode in ['Plane', 'Train']:
    mode_data = df[df['Mode'] == mode]
    if len(mode_data) > 0:
        weekly_means = mode_data.groupby('week')['Ticket Price'].mean()
        weekly_std = mode_data.groupby('week')['Ticket Price'].std().fillna(0)
        weekly_count = mode_data.groupby('week').size()
        weekly_sem = weekly_std / np.sqrt(weekly_count)
        
        # Confidence interval (95%)
        ci = 1.96 * weekly_sem
        
        # Plot line
        ax3.plot(weekly_means.index, weekly_means.values, 'o-', 
                 color=colors[mode], linewidth=3, markersize=8, label=mode)
        
        # Add confidence interval
        ax3.fill_between(weekly_means.index, 
                         weekly_means.values - ci, 
                         weekly_means.values + ci,
                         alpha=0.3, color=colors[mode])

ax3.set_xlabel('Week', fontweight='bold')
ax3.set_ylabel('Average Ticket Price', fontweight='bold')
ax3.set_title('Price Trends with 95% Confidence Intervals', fontweight='bold', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Radar chart with scatter overlay
ax4 = plt.subplot(3, 2, 4, facecolor='white', projection='polar')

# Prepare data for radar chart
overall_mean = df['Ticket Price'].mean()
route_mode_stats = df.groupby(['Route', 'Mode'])['Ticket Price'].agg(['mean', 'std']).reset_index()
route_mode_stats['deviation'] = (route_mode_stats['mean'] - overall_mean) / overall_mean * 100

# Get unique combinations and create consistent angles
unique_routes = df['Route'].unique()
n_categories = len(unique_routes)
angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# Plot radar chart for each mode
for mode in ['Plane', 'Train']:
    mode_data = route_mode_stats[route_mode_stats['Mode'] == mode]
    
    # Create values array matching the angles
    values = []
    for route in unique_routes:
        route_data = mode_data[mode_data['Route'] == route]
        if len(route_data) > 0:
            values.append(route_data['deviation'].iloc[0])
        else:
            values.append(0)
    
    values = np.array(values)
    values = np.concatenate((values, [values[0]]))
    
    ax4.plot(angles, values, 'o-', linewidth=2, label=mode, color=colors[mode])
    ax4.fill(angles, values, alpha=0.25, color=colors[mode])
    
    # Add scatter points for actual values
    for i, val in enumerate(values[:-1]):
        ax4.scatter(angles[i], val, s=60, color=colors[mode], 
                   edgecolors='white', linewidth=2, zorder=5)

# Customize radar chart
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(unique_routes, fontsize=10)
ax4.set_title('Normalized Price Deviations from Overall Mean', fontweight='bold', fontsize=14, pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax4.grid(True, alpha=0.3)

# Subplot 5: Diverging lollipop chart with histogram
ax5 = plt.subplot(3, 2, 5, facecolor='white')

# Calculate percentage differences from median
dataset_median = df['Ticket Price'].median()
route_mode_pct_diff = []

for _, row in df.iterrows():
    pct_diff = (row['Ticket Price'] - dataset_median) / dataset_median * 100
    route_mode_pct_diff.append({
        'Route_Mode': f"{row['Route']}-{row['Mode']}",
        'Pct_Diff': pct_diff,
        'Mode': row['Mode']
    })

pct_diff_df = pd.DataFrame(route_mode_pct_diff)
avg_pct_diff = pct_diff_df.groupby('Route_Mode')['Pct_Diff'].mean().reset_index()
avg_pct_diff['Mode'] = avg_pct_diff['Route_Mode'].str.split('-').str[1]

# Create lollipop chart
y_pos = np.arange(len(avg_pct_diff))
for i, (_, row) in enumerate(avg_pct_diff.iterrows()):
    color = colors.get(row['Mode'], '#666666')
    ax5.plot([0, row['Pct_Diff']], [i, i], color=color, linewidth=3, alpha=0.7)
    ax5.scatter(row['Pct_Diff'], i, color=color, s=100, zorder=5, 
               edgecolors='white', linewidth=2)

ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(avg_pct_diff['Route_Mode'], fontsize=10)
ax5.set_xlabel('Percentage Difference from Dataset Median', fontweight='bold')
ax5.set_title('Price Deviations from Median with Distribution', fontweight='bold', fontsize=14)

# Add histogram on the right side
ax5_hist = ax5.twinx()
ax5_hist.hist(pct_diff_df['Pct_Diff'], bins=15, alpha=0.3, color='gray', 
              orientation='horizontal', density=True)
ax5_hist.set_ylabel('Distribution Density', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Subplot 6: Box plots with violin overlays and trend lines
ax6 = plt.subplot(3, 2, 6, facecolor='white')

# Prepare data for box plots
routes = df['Route'].unique()
modes = df['Mode'].unique()

# Create positions for box plots
positions = []
labels = []
box_data = []

pos = 0
for route in routes:
    for mode in modes:
        data = df[(df['Route'] == route) & (df['Mode'] == mode)]['Ticket Price']
        if len(data) > 0:
            box_data.append(data.values)
            positions.append(pos)
            labels.append(f"{route}\n{mode}")
            pos += 1
    pos += 0.5  # Add space between routes

if len(box_data) > 0:
    # Create violin plots
    try:
        parts = ax6.violinplot(box_data, positions=positions, widths=0.6, showmeans=False, 
                               showmedians=False, showextrema=False)
        
        # Color violin plots
        for i, pc in enumerate(parts['bodies']):
            mode = modes[i % len(modes)]
            pc.set_facecolor(colors[mode])
            pc.set_alpha(0.3)
    except:
        pass  # Skip violin plots if they fail
    
    # Create box plots
    bp = ax6.boxplot(box_data, positions=positions, widths=0.3, patch_artist=True,
                     showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.6))
    
    # Color box plots
    for i, patch in enumerate(bp['boxes']):
        mode = modes[i % len(modes)]
        patch.set_facecolor(colors[mode])
        patch.set_alpha(0.7)
    
    # Add mean markers and trend lines
    means = [np.mean(data) for data in box_data]
    ax6.scatter(positions, means, color='red', s=50, zorder=5, marker='D', 
               edgecolors='white', linewidth=1, label='Mean')
    
    # Connect means with trend lines for each route
    for i in range(0, len(positions), len(modes)):
        if i + len(modes) - 1 < len(positions):
            route_positions = positions[i:i+len(modes)]
            route_means = means[i:i+len(modes)]
            if len(route_positions) == len(route_means):
                ax6.plot(route_positions, route_means, 'r--', alpha=0.7, linewidth=2)

ax6.set_xticks(positions)
ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax6.set_ylabel('Ticket Price', fontweight='bold')
ax6.set_title('Price Distributions with Violin Overlays and Trend Lines', fontweight='bold', fontsize=14)
ax6.legend()
ax6.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('comprehensive_flight_train_analysis.png', dpi=300, bbox_inches='tight')
plt.show()