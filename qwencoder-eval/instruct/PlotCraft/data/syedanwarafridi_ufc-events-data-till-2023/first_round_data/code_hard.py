import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('ufc.csv')

# Clean and prepare data
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Fill missing values with 0 for fighter statistics
numeric_cols = ['Fighter_1_KD', 'Fighter_2_KD', 'Fighter_1_STR', 'Fighter_2_STR', 
                'Fighter_1_TD', 'Fighter_2_TD', 'Fighter_1_SUB', 'Fighter_2_SUB']
df[numeric_cols] = df[numeric_cols].fillna(0)

# Get top 5 weight classes
top_weight_classes = df['Weight_Class'].value_counts().head(5).index.tolist()
df_top = df[df['Weight_Class'].isin(top_weight_classes)]

# Create color palette for weight classes
colors = plt.cm.Set3(np.linspace(0, 1, len(top_weight_classes)))
weight_class_colors = dict(zip(top_weight_classes, colors))

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Subplot 1: Scatter plot with KDE contours and marginal histograms
ax1 = plt.subplot(3, 3, 1)
for i, wc in enumerate(top_weight_classes):
    data = df_top[df_top['Weight_Class'] == wc]
    ax1.scatter(data['Fighter_1_STR'], data['Fighter_1_KD'], 
               c=[weight_class_colors[wc]], alpha=0.6, s=30, label=wc)

# Add KDE contours
x = df_top['Fighter_1_STR'].values
y = df_top['Fighter_1_KD'].values
valid_mask = ~(np.isnan(x) | np.isnan(y))
x, y = x[valid_mask], y[valid_mask]

if len(x) > 10:
    try:
        xx, yy = np.mgrid[x.min():x.max():.5, y.min():y.max():.1]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax1.contour(xx, yy, f, colors='black', alpha=0.3, linewidths=0.8)
    except:
        pass

ax1.set_xlabel('Fighter 1 Significant Strikes', fontweight='bold')
ax1.set_ylabel('Fighter 1 Knockdowns', fontweight='bold')
ax1.set_title('Fighter Performance Clustering: Strikes vs Knockdowns', fontweight='bold', fontsize=12)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Bubble chart with regression line
ax2 = plt.subplot(3, 3, 2)
for wc in top_weight_classes:
    data = df_top[df_top['Weight_Class'] == wc]
    bubble_sizes = (data['Fighter_1_STR'] + 1) * 2  # +1 to avoid zero sizes
    ax2.scatter(data['Fighter_1_TD'], data['Fighter_1_SUB'], 
               s=bubble_sizes, c=[weight_class_colors[wc]], alpha=0.6, label=wc)

# Add regression line
x_reg = df_top['Fighter_1_TD'].values
y_reg = df_top['Fighter_1_SUB'].values
valid_mask = ~(np.isnan(x_reg) | np.isnan(y_reg))
x_reg, y_reg = x_reg[valid_mask], y_reg[valid_mask]

if len(x_reg) > 1:
    z = np.polyfit(x_reg, y_reg, 1)
    p = np.poly1d(z)
    ax2.plot(x_reg, p(x_reg), "r--", alpha=0.8, linewidth=2)

ax2.set_xlabel('Fighter 1 Takedowns', fontweight='bold')
ax2.set_ylabel('Fighter 1 Submission Attempts', fontweight='bold')
ax2.set_title('Takedowns vs Submissions\n(Bubble size = Strikes)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Subplot 3: Violin plot with box plots
ax3 = plt.subplot(3, 3, 3)
violin_data = [df_top[df_top['Weight_Class'] == wc]['Fighter_1_STR'].values for wc in top_weight_classes]
parts = ax3.violinplot(violin_data, positions=range(len(top_weight_classes)), showmeans=True)

# Color the violins
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

# Add box plots - FIXED: removed alpha parameter
box_data = ax3.boxplot(violin_data, positions=range(len(top_weight_classes)), 
                       patch_artist=True, widths=0.3)
for patch, color in zip(box_data['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax3.set_xticks(range(len(top_weight_classes)))
ax3.set_xticklabels([wc.replace(' ', '\n') for wc in top_weight_classes], fontsize=8)
ax3.set_ylabel('Significant Strikes', fontweight='bold')
ax3.set_title('Strike Distribution by Weight Class', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Subplot 4: Stacked bar chart with line plot
ax4 = plt.subplot(3, 3, 4)
method_counts = df_top.groupby(['Weight_Class', 'Method']).size().unstack(fill_value=0)
method_counts.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
ax4.set_xlabel('Weight Class', fontweight='bold')
ax4.set_ylabel('Number of Fights', fontweight='bold')
ax4.set_title('Fight Methods by Weight Class', fontweight='bold', fontsize=12)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax4.tick_params(axis='x', rotation=45)

# Subplot 5: Heatmap with clustering
ax5 = plt.subplot(3, 3, 5)
heatmap_data = df_top.groupby(['Weight_Class', 'Method']).size().unstack(fill_value=0)
heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

sns.heatmap(heatmap_data_pct, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5, cbar_kws={'shrink': 0.8})
ax5.set_title('Method Distribution by Weight Class (%)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Method', fontweight='bold')
ax5.set_ylabel('Weight Class', fontweight='bold')

# Subplot 6: Radar chart
ax6 = plt.subplot(3, 3, 6, projection='polar')
metrics = ['Fighter_1_KD', 'Fighter_1_STR', 'Fighter_1_TD', 'Fighter_1_SUB']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, wc in enumerate(top_weight_classes[:3]):  # Limit to 3 for clarity
    values = df_top[df_top['Weight_Class'] == wc][metrics].mean().tolist()
    values += values[:1]  # Complete the circle
    ax6.plot(angles, values, 'o-', linewidth=2, label=wc, color=colors[i])
    ax6.fill(angles, values, alpha=0.25, color=colors[i])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(['KD', 'STR', 'TD', 'SUB'])
ax6.set_title('Performance Metrics Comparison\n(Top 3 Weight Classes)', fontweight='bold', fontsize=12, pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 7: Time series
ax7 = plt.subplot(3, 3, 7)
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_counts = df.groupby(['YearMonth', 'Location']).size().unstack(fill_value=0)

# Plot top 3 locations
top_locations = df['Location'].value_counts().head(3).index
for i, loc in enumerate(top_locations):
    if loc in monthly_counts.columns:
        monthly_counts[loc].plot(ax=ax7, label=loc[:15], linewidth=2, color=plt.cm.tab10(i))

ax7.set_xlabel('Date', fontweight='bold')
ax7.set_ylabel('Number of Fights', fontweight='bold')
ax7.set_title('Monthly Fight Frequency by Location', fontweight='bold', fontsize=12)
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax7.grid(True, alpha=0.3)

# Subplot 8: Network graph
ax8 = plt.subplot(3, 3, 8)
location_weight_counts = df_top.groupby(['Location', 'Weight_Class']).size().reset_index(name='count')
location_weight_counts = location_weight_counts[location_weight_counts['count'] >= 10]  # Filter for clarity

G = nx.Graph()
for _, row in location_weight_counts.iterrows():
    loc_short = row['Location'].split(',')[0][:8]  # Shorten location names
    G.add_edge(loc_short, row['Weight_Class'][:8], weight=row['count'])

if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1, iterations=50)
    node_sizes = [G.degree(node) * 50 + 100 for node in G.nodes()]
    nx.draw(G, pos, ax=ax8, node_size=node_sizes, node_color='lightblue', 
            with_labels=True, font_size=6, font_weight='bold', edge_color='gray')

ax8.set_title('Location-Weight Class Network', fontweight='bold', fontsize=12)

# Subplot 9: Parallel coordinates
ax9 = plt.subplot(3, 3, 9)
parallel_data = df_top[metrics + ['Weight_Class']].copy()
parallel_data_norm = parallel_data.copy()

# Normalize the data
scaler = StandardScaler()
parallel_data_norm[metrics] = scaler.fit_transform(parallel_data_norm[metrics])

for i, wc in enumerate(top_weight_classes):
    data = parallel_data_norm[parallel_data_norm['Weight_Class'] == wc]
    # Sample data for clarity
    sample_data = data.sample(min(20, len(data))) if len(data) > 0 else data
    for idx, row in sample_data.iterrows():
        ax9.plot(range(len(metrics)), row[metrics], color=colors[i], alpha=0.3, linewidth=0.5)

# Plot means
for i, wc in enumerate(top_weight_classes):
    data = parallel_data_norm[parallel_data_norm['Weight_Class'] == wc]
    if len(data) > 0:
        means = data[metrics].mean()
        ax9.plot(range(len(metrics)), means, color=colors[i], linewidth=3, label=wc[:10])

ax9.set_xticks(range(len(metrics)))
ax9.set_xticklabels(['KD', 'STR', 'TD', 'SUB'])
ax9.set_ylabel('Normalized Values', fontweight='bold')
ax9.set_title('Parallel Coordinates: Fighter Statistics', fontweight='bold', fontsize=12)
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig('ufc_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()