import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('All 1.csv')

# Data preprocessing
df['PassengerId_bin'] = pd.cut(df['PassengerId'], bins=10, labels=False)
df['PassengerId_quartile'] = pd.qcut(df['PassengerId'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Calculate deviations from 50% baseline
baseline = 0.5
bin_stats = df.groupby('PassengerId_bin').agg({
    'Survived': ['mean', 'count', 'std'],
    'PassengerId': ['min', 'max', 'mean']
}).round(3)

bin_stats.columns = ['survival_rate', 'count', 'std', 'id_min', 'id_max', 'id_mean']
bin_stats['deviation'] = bin_stats['survival_rate'] - baseline
bin_stats['std_err'] = bin_stats['std'].fillna(0) / np.sqrt(bin_stats['count'])

# Quartile statistics
quartile_stats = df.groupby('PassengerId_quartile').agg({
    'Survived': ['mean', 'count'],
    'PassengerId': 'mean'
}).round(3)
quartile_stats.columns = ['actual_rate', 'count', 'avg_id']
quartile_stats['expected_rate'] = baseline
quartile_stats['deviation'] = quartile_stats['actual_rate'] - baseline

# Create 3x3 grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Subplot 1: Diverging bar chart with error bars and cumulative trend line
ax1 = plt.subplot(3, 3, 1)
colors = ['red' if x < 0 else 'green' for x in bin_stats['deviation']]
bars = ax1.bar(range(len(bin_stats)), bin_stats['deviation'], 
               yerr=bin_stats['std_err'], capsize=5, color=colors, alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1_twin = ax1.twinx()
cumulative_dev = np.cumsum(bin_stats['deviation'])
ax1_twin.plot(range(len(bin_stats)), cumulative_dev, 'b-o', linewidth=2, markersize=4)
ax1.set_title('Survival Rate Deviations by Passenger ID Bins', fontweight='bold', fontsize=12)
ax1.set_xlabel('Passenger ID Bin')
ax1.set_ylabel('Deviation from 50% Baseline', color='black')
ax1_twin.set_ylabel('Cumulative Deviation', color='blue')
ax1.grid(True, alpha=0.3)

# Subplot 2: Dumbbell plot with scatter points
ax2 = plt.subplot(3, 3, 2)
y_pos = range(len(quartile_stats))
for i, (idx, row) in enumerate(quartile_stats.iterrows()):
    ax2.plot([row['expected_rate'], row['actual_rate']], [i, i], 'k-', linewidth=2)
    ax2.scatter(row['expected_rate'], i, color='red', s=100, label='Expected' if i == 0 else "")
    ax2.scatter(row['actual_rate'], i, color='blue', s=100, label='Actual' if i == 0 else "")

# Add individual passenger deviations as scatter
for q_idx, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    q_data = df[df['PassengerId_quartile'] == quartile]
    individual_devs = q_data['Survived'] - baseline
    ax2.scatter(individual_devs + baseline, [q_idx] * len(individual_devs), 
               alpha=0.3, s=20, color='gray')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(quartile_stats.index)
ax2.set_title('Actual vs Expected Survival Rates by Quartile', fontweight='bold', fontsize=12)
ax2.set_xlabel('Survival Rate')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Area chart with violin plot overlay
ax3 = plt.subplot(3, 3, 3)
deviation_magnitudes = np.abs(bin_stats['deviation'])
x_area = range(len(deviation_magnitudes))
ax3.fill_between(x_area, 0, deviation_magnitudes, alpha=0.6, color='lightblue')
ax3.plot(x_area, deviation_magnitudes, 'b-', linewidth=2)

# Violin plot overlay
ax3_twin = ax3.twinx()
all_deviations = df['Survived'] - baseline
violin_parts = ax3_twin.violinplot([all_deviations], positions=[5], widths=2, showmeans=True)
for pc in violin_parts['bodies']:
    pc.set_facecolor('orange')
    pc.set_alpha(0.7)

ax3.set_title('Deviation Magnitude Distribution', fontweight='bold', fontsize=12)
ax3.set_xlabel('Passenger ID Bin')
ax3.set_ylabel('Deviation Magnitude', color='blue')
ax3_twin.set_ylabel('Individual Deviations', color='orange')
ax3.grid(True, alpha=0.3)

# Subplot 4: Radar chart with polar scatter
ax4 = plt.subplot(3, 3, 4, projection='polar')
categories = ['Early Passengers', 'Late Passengers', 'High ID Cluster', 'Low ID Cluster', 'Mid Range']
early_dev = df[df['PassengerId'] < df['PassengerId'].median()]['Survived'].mean() - baseline
late_dev = df[df['PassengerId'] >= df['PassengerId'].median()]['Survived'].mean() - baseline
high_cluster = df[df['PassengerId'] > df['PassengerId'].quantile(0.75)]['Survived'].mean() - baseline
low_cluster = df[df['PassengerId'] < df['PassengerId'].quantile(0.25)]['Survived'].mean() - baseline
mid_range = df[(df['PassengerId'] >= df['PassengerId'].quantile(0.25)) & 
               (df['PassengerId'] <= df['PassengerId'].quantile(0.75))]['Survived'].mean() - baseline

values = [early_dev, late_dev, high_cluster, low_cluster, mid_range]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

ax4.plot(angles, values, 'o-', linewidth=2, color='red')
ax4.fill(angles, values, alpha=0.25, color='red')

# Polar scatter overlay
scatter_angles = np.random.uniform(0, 2*np.pi, len(df))
scatter_radii = np.abs(df['Survived'] - baseline)
ax4.scatter(scatter_angles, scatter_radii, alpha=0.3, s=10, color='blue')

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=8)
ax4.set_title('Multi-Dimensional Deviation Analysis', fontweight='bold', fontsize=12, pad=20)

# Subplot 5: Slope chart with error bands
ax5 = plt.subplot(3, 3, 5)
x_positions = [0, 1]
for i, (idx, row) in enumerate(quartile_stats.iterrows()):
    ax5.plot(x_positions, [baseline, row['actual_rate']], 'o-', linewidth=2, alpha=0.7)
    # Error band
    std_err = np.sqrt(row['actual_rate'] * (1 - row['actual_rate']) / row['count'])
    ax5.fill_between(x_positions, 
                     [baseline - 0.02, row['actual_rate'] - std_err],
                     [baseline + 0.02, row['actual_rate'] + std_err],
                     alpha=0.2)

ax5.set_xlim(-0.1, 1.1)
ax5.set_xticks(x_positions)
ax5.set_xticklabels(['Expected (50%)', 'Actual Rate'])
ax5.set_title('Baseline to Actual Survival Transitions', fontweight='bold', fontsize=12)
ax5.set_ylabel('Survival Rate')
ax5.grid(True, alpha=0.3)

# Subplot 6: Stacked area chart with trend line
ax6 = plt.subplot(3, 3, 6)
positive_devs = np.maximum(bin_stats['deviation'], 0)
negative_devs = np.minimum(bin_stats['deviation'], 0)
x_range = range(len(bin_stats))

ax6.fill_between(x_range, 0, positive_devs, alpha=0.7, color='green', label='Positive Deviation')
ax6.fill_between(x_range, 0, negative_devs, alpha=0.7, color='red', label='Negative Deviation')

# Trend line
z = np.polyfit(x_range, bin_stats['deviation'], 1)
p = np.poly1d(z)
ax6.plot(x_range, p(x_range), 'k--', linewidth=2, label='Trend')

# Confidence intervals
conf_interval = 1.96 * bin_stats['std_err']
ax6.fill_between(x_range, 
                 bin_stats['deviation'] - conf_interval,
                 bin_stats['deviation'] + conf_interval,
                 alpha=0.2, color='gray', label='95% CI')

ax6.set_title('Cumulative Deviation Patterns', fontweight='bold', fontsize=12)
ax6.set_xlabel('Passenger ID Bin')
ax6.set_ylabel('Deviation from Baseline')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Subplot 7: Diverging lollipop with histogram
ax7 = plt.subplot(3, 3, 7)
for i, dev in enumerate(bin_stats['deviation']):
    color = 'red' if dev < 0 else 'green'
    ax7.plot([i, i], [0, dev], color=color, linewidth=3)
    ax7.scatter(i, dev, color=color, s=100, zorder=5)

# Histogram overlay
ax7_twin = ax7.twinx()
counts, bins, patches = ax7_twin.hist(df['Survived'] - baseline, bins=20, alpha=0.3, color='blue')

ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.set_title('Deviation Scores with Distribution', fontweight='bold', fontsize=12)
ax7.set_xlabel('Passenger ID Bin')
ax7.set_ylabel('Deviation Score', color='black')
ax7_twin.set_ylabel('Frequency', color='blue')
ax7.grid(True, alpha=0.3)

# Subplot 8: Error bar with strip plot
ax8 = plt.subplot(3, 3, 8)
quartile_means = [quartile_stats.loc[q, 'actual_rate'] for q in ['Q1', 'Q2', 'Q3', 'Q4']]
quartile_stds = [df[df['PassengerId_quartile'] == q]['Survived'].std() for q in ['Q1', 'Q2', 'Q3', 'Q4']]

ax8.errorbar(range(4), quartile_means, yerr=quartile_stds, 
             fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8)

# Strip plot overlay
for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    q_data = df[df['PassengerId_quartile'] == quartile]['Survived']
    x_jitter = np.random.normal(i, 0.05, len(q_data))
    ax8.scatter(x_jitter, q_data, alpha=0.4, s=20)

ax8.set_xticks(range(4))
ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax8.set_title('Survival Spread Patterns by Quartile', fontweight='bold', fontsize=12)
ax8.set_xlabel('Passenger ID Quartile')
ax8.set_ylabel('Survival Rate')
ax8.grid(True, alpha=0.3)

# Subplot 9: Multi-layered heatmap with contours and marginals (FIXED)
ax9 = plt.subplot(3, 3, 9)

# Create proper 2D grid for heatmap and contours
passenger_ids_norm = (df['PassengerId'] - df['PassengerId'].min()) / (df['PassengerId'].max() - df['PassengerId'].min())
survived_vals = df['Survived']

# Create meshgrid for proper 2D plotting
x_bins = np.linspace(0, 1, 15)
y_bins = np.linspace(0, 1, 10)

# Create 2D histogram
hist, x_edges, y_edges = np.histogram2d(passenger_ids_norm, survived_vals, bins=[x_bins, y_bins])

# Ensure we have at least 2x2 for contours
if hist.shape[0] >= 2 and hist.shape[1] >= 2:
    # Create proper meshgrid for contours
    X_centers = (x_edges[:-1] + x_edges[1:]) / 2
    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(X_centers, Y_centers)
    
    # Plot heatmap
    im = ax9.imshow(hist.T, origin='lower', aspect='auto', cmap='RdYlBu', alpha=0.8,
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    
    # Add contour lines only if we have enough data points
    if np.sum(hist) > 10:  # Only add contours if we have sufficient data
        try:
            ax9.contour(X, Y, hist.T, levels=3, colors='black', alpha=0.5, linewidths=1)
        except:
            pass  # Skip contours if they fail
else:
    # Fallback: simple scatter plot
    ax9.scatter(passenger_ids_norm, survived_vals, alpha=0.6, c=survived_vals, cmap='RdYlBu')

# Add marginal distributions
# Create inset axes for marginals
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Top marginal (PassengerId distribution)
ax_top = inset_axes(ax9, width="100%", height="30%", loc='upper center', 
                    bbox_to_anchor=(0, 1.02, 1, 0.3), bbox_transform=ax9.transAxes)
ax_top.hist(passenger_ids_norm, bins=20, alpha=0.7, color='blue', density=True)
ax_top.set_xlim(ax9.get_xlim())
ax_top.set_xticks([])

# Right marginal (Survival distribution)
ax_right = inset_axes(ax9, width="30%", height="100%", loc='center right',
                      bbox_to_anchor=(1.02, 0, 0.3, 1), bbox_transform=ax9.transAxes)
survival_counts = df['Survived'].value_counts().sort_index()
ax_right.barh(survival_counts.index, survival_counts.values, alpha=0.7, color='orange')
ax_right.set_ylim(ax9.get_ylim())
ax_right.set_yticks([])

ax9.set_title('PassengerId vs Survival Heatmap with Marginals', fontweight='bold', fontsize=12)
ax9.set_xlabel('Normalized Passenger ID')
ax9.set_ylabel('Survival Status')

# Adjust layout
plt.tight_layout(pad=2.0)
plt.savefig('titanic_deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()