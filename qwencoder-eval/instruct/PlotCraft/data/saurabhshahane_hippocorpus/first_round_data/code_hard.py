import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.patches as patches
from matplotlib.patches import Circle
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('hippoCorpusV2.csv')

# Data preprocessing
# Handle missing values for key variables
df = df.dropna(subset=['memType', 'openness', 'importance', 'annotatorAge', 'annotatorGender'])
df['annotatorRace'] = df['annotatorRace'].fillna('Unknown')

# Define Likert scale variables
likert_vars = ['distracted', 'draining', 'frequency', 'importance', 'similarity', 'stressful']
for var in likert_vars:
    df[var] = df[var].fillna(df[var].median())

# Create age groups
df['age_group'] = pd.cut(df['annotatorAge'], bins=[0, 25, 35, 50, 100], 
                        labels=['18-25', '26-35', '36-50', '50+'])

# Set up the figure with white background and better spacing
fig = plt.figure(figsize=(26, 22))
fig.patch.set_facecolor('white')
fig.suptitle('Comprehensive Analysis of Clustering Patterns in the Hippocorpus Dataset', 
             fontsize=20, fontweight='bold', y=0.98)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Define improved color palettes
memtype_colors = {'recalled': '#2E86AB', 'imagined': '#A23B72', 'retold': '#F18F01'}
gender_colors = {'man': '#4A90E2', 'woman': '#E94B3C', 'other': '#50C878'}
race_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Subplot 1: Scatter plot with marginal histograms and density contours
from matplotlib.gridspec import GridSpec

# Create custom grid for subplot 1 with marginal histograms
gs1 = GridSpec(3, 3, figure=fig, left=0.05, right=0.32, top=0.85, bottom=0.68, 
               width_ratios=[1, 0.2, 0.05], height_ratios=[0.2, 1, 0.05])

# Main scatter plot
ax1_main = fig.add_subplot(gs1[1, 0])
ax1_main.set_facecolor('white')

# Marginal histograms
ax1_top = fig.add_subplot(gs1[0, 0], sharex=ax1_main)
ax1_right = fig.add_subplot(gs1[1, 1], sharey=ax1_main)

# Main scatter plot with different markers for each memType
markers = {'recalled': 'o', 'imagined': 's', 'retold': '^'}
for memtype in df['memType'].unique():
    subset = df[df['memType'] == memtype]
    ax1_main.scatter(subset['openness'], subset['importance'], 
                    c=memtype_colors[memtype], alpha=0.6, s=30, 
                    marker=markers[memtype], label=memtype, edgecolors='white', linewidth=0.5)

# Add density contours
for memtype in df['memType'].unique():
    subset = df[df['memType'] == memtype]
    if len(subset) > 10:
        try:
            x = subset['openness'].values
            y = subset['importance'].values
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            
            from scipy.stats import gaussian_kde
            positions = np.vstack([xi.ravel(), yi.ravel()])
            values = np.vstack([x, y])
            kernel = gaussian_kde(values)
            zi = np.reshape(kernel(positions).T, xi.shape)
            
            ax1_main.contour(xi, yi, zi, levels=3, colors=memtype_colors[memtype], alpha=0.7, linewidths=1.5)
        except:
            pass

# Marginal histograms
for memtype in df['memType'].unique():
    subset = df[df['memType'] == memtype]
    ax1_top.hist(subset['openness'], bins=20, alpha=0.6, color=memtype_colors[memtype], density=True)
    ax1_right.hist(subset['importance'], bins=20, alpha=0.6, color=memtype_colors[memtype], 
                   orientation='horizontal', density=True)

ax1_main.set_xlabel('Openness', fontweight='bold', fontsize=12)
ax1_main.set_ylabel('Importance', fontweight='bold', fontsize=12)
ax1_main.set_title('Openness vs Importance by Memory Type\nwith Density Contours & Marginal Histograms', 
                   fontweight='bold', pad=20, fontsize=14)
ax1_main.legend(title='Memory Type', title_fontsize=11, fontsize=10)
ax1_main.grid(True, alpha=0.3)

# Style marginal plots
ax1_top.set_ylabel('Density', fontweight='bold', fontsize=10)
ax1_right.set_xlabel('Density', fontweight='bold', fontsize=10)
ax1_top.tick_params(labelbottom=False)
ax1_right.tick_params(labelleft=False)

# Subplot 2: Violin plot with strip plot and box plots
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Prepare data for violin plot
age_groups = df['age_group'].dropna().unique()
violin_data = [df[df['age_group'] == group]['logTimeSinceEvent'].dropna() for group in age_groups]

# Create violin plot
parts = ax2.violinplot(violin_data, positions=range(len(age_groups)), 
                      showmeans=False, showmedians=False, showextrema=False, widths=0.8)

for pc in parts['bodies']:
    pc.set_facecolor('#E8F4FD')
    pc.set_alpha(0.7)
    pc.set_edgecolor('navy')

# Add box plots inside violins
box_parts = ax2.boxplot(violin_data, positions=range(len(age_groups)), 
                       widths=0.3, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.9, edgecolor='black'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))

# Add strip plot
for i, group in enumerate(age_groups):
    group_data = df[df['age_group'] == group]['logTimeSinceEvent'].dropna()
    if len(group_data) > 0:
        # Sample to avoid overcrowding
        sample_size = min(50, len(group_data))
        sample_data = group_data.sample(n=sample_size, random_state=42)
        y_vals = sample_data.values
        x_vals = np.random.normal(i, 0.08, size=len(y_vals))
        ax2.scatter(x_vals, y_vals, alpha=0.5, s=12, color='darkblue', edgecolors='white', linewidth=0.3)

ax2.set_xticks(range(len(age_groups)))
ax2.set_xticklabels(age_groups)
ax2.set_xlabel('Age Group', fontweight='bold', fontsize=12)
ax2.set_ylabel('Log Time Since Event', fontweight='bold', fontsize=12)
ax2.set_title('Distribution of Log Time Since Event by Age Group\n(Violin + Box + Strip Plot)', 
              fontweight='bold', pad=20, fontsize=14)
ax2.grid(True, alpha=0.3)

# Subplot 3: Stacked bar chart with line overlay (improved colors)
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Calculate race composition by gender
gender_race_counts = df.groupby(['annotatorGender', 'annotatorRace']).size().unstack(fill_value=0)
gender_totals = gender_race_counts.sum(axis=1)

# Create stacked bar chart with percentages using improved colors
bottom = np.zeros(len(gender_race_counts))
distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for i, race in enumerate(gender_race_counts.columns):
    values = gender_race_counts[race].values
    percentages = (values / gender_totals.values) * 100
    color = distinct_colors[i % len(distinct_colors)]
    bars = ax3.bar(gender_race_counts.index, percentages, bottom=bottom, 
                   color=color, label=race, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add percentage labels for larger segments
    for j, (bar, pct) in enumerate(zip(bars, percentages)):
        if pct > 8:  # Only show labels for segments > 8%
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2.,
                    f'{pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    bottom += percentages

# Add line plot overlay for total counts (improved styling)
ax3_twin = ax3.twinx()
ax3_twin.plot(gender_race_counts.index, gender_totals.values, 
              color='#2c3e50', marker='o', linewidth=2.5, markersize=8, 
              label='Total Count', markerfacecolor='white', markeredgecolor='#2c3e50', markeredgewidth=2)
ax3_twin.set_ylabel('Total Count', fontweight='bold', color='#2c3e50', fontsize=12)
ax3_twin.tick_params(axis='y', labelcolor='#2c3e50')

# Wrap long labels
wrapped_labels = []
for label in gender_race_counts.index:
    if len(label) > 8:
        wrapped_labels.append(label[:8] + '\n' + label[8:])
    else:
        wrapped_labels.append(label)

ax3.set_xticks(range(len(gender_race_counts.index)))
ax3.set_xticklabels(wrapped_labels, fontsize=10)
ax3.set_xlabel('Gender', fontweight='bold', fontsize=12)
ax3.set_ylabel('Percentage', fontweight='bold', fontsize=12)
ax3.set_title('Race Composition by Gender with Total Counts\n(Stacked Bar + Line Overlay)', 
              fontweight='bold', pad=20, fontsize=14)
ax3.legend(title='Race', bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Subplot 4: Radar chart
ax4 = plt.subplot(3, 3, 4, projection='polar')
ax4.set_facecolor('white')

# Calculate mean values for radar chart
radar_data = df.groupby('memType')[likert_vars].mean()

# Set up radar chart
angles = np.linspace(0, 2 * np.pi, len(likert_vars), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, memtype in enumerate(radar_data.index):
    values = radar_data.loc[memtype].tolist()
    values += values[:1]  # Complete the circle
    
    ax4.plot(angles, values, 'o-', linewidth=3, label=memtype, 
             color=memtype_colors[memtype], markersize=6)
    ax4.fill(angles, values, alpha=0.25, color=memtype_colors[memtype])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(likert_vars, fontsize=10, fontweight='bold')
ax4.set_ylim(0, 5)
ax4.set_title('Likert Scale Variables by Memory Type\n(Radar Chart with Filled Areas)', 
              fontweight='bold', pad=30, fontsize=14)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
ax4.grid(True, alpha=0.3)

# Subplot 5: Parallel coordinates plot
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Prepare data for parallel coordinates
parallel_vars = ['WorkTimeInSeconds', 'openness'] + likert_vars
parallel_data = df[parallel_vars + ['memType']].dropna()

# Normalize data for parallel coordinates
normalized_data = parallel_data.copy()
for var in parallel_vars:
    normalized_data[var] = (parallel_data[var] - parallel_data[var].min()) / \
                          (parallel_data[var].max() - parallel_data[var].min())

# Sample data to avoid overcrowding
sample_size = min(800, len(normalized_data))
sample_data = normalized_data.sample(n=sample_size, random_state=42)

# Plot parallel coordinates with transparency based on density
for memtype in sample_data['memType'].unique():
    subset = sample_data[sample_data['memType'] == memtype]
    alpha_val = min(0.6, 200 / len(subset))  # Adjust transparency based on density
    
    for idx, row in subset.iterrows():
        ax5.plot(range(len(parallel_vars)), [row[var] for var in parallel_vars], 
                color=memtype_colors[memtype], alpha=alpha_val, linewidth=1)

ax5.set_xticks(range(len(parallel_vars)))
ax5.set_xticklabels(parallel_vars, rotation=45, ha='right', fontsize=10)
ax5.set_ylabel('Normalized Values', fontweight='bold', fontsize=12)
ax5.set_title('Parallel Coordinates Plot by Memory Type\n(Transparency ∝ Density)', 
              fontweight='bold', pad=20, fontsize=14)
ax5.grid(True, alpha=0.3)

# Create custom legend
legend_elements = [plt.Line2D([0], [0], color=memtype_colors[mt], lw=3, label=mt) 
                  for mt in memtype_colors.keys()]
ax5.legend(handles=legend_elements, title='Memory Type', fontsize=10)

# Subplot 6: Correlation heatmap with hierarchical clustering dendrograms
from matplotlib.gridspec import GridSpec

# Create custom grid for subplot 6 with dendrograms
gs6 = GridSpec(3, 3, figure=fig, left=0.38, right=0.65, top=0.52, bottom=0.35, 
               width_ratios=[0.2, 1, 0.05], height_ratios=[0.2, 1, 0.05])

# Main heatmap
ax6_main = fig.add_subplot(gs6[1, 1])
ax6_main.set_facecolor('white')

# Dendrogram axes
ax6_top = fig.add_subplot(gs6[0, 1], sharex=ax6_main)
ax6_left = fig.add_subplot(gs6[1, 0], sharey=ax6_main)

# Select numerical variables for correlation
numerical_vars = ['WorkTimeInSeconds', 'annotatorAge', 'openness', 'logTimeSinceEvent'] + likert_vars
corr_data = df[numerical_vars].corr()

# Perform hierarchical clustering
linkage_matrix = linkage(pdist(corr_data), method='ward')
dendro_top = dendrogram(linkage_matrix, ax=ax6_top, orientation='top', 
                       labels=corr_data.columns, leaf_rotation=90, no_plot=False)
dendro_left = dendrogram(linkage_matrix, ax=ax6_left, orientation='left', 
                        labels=corr_data.columns, no_plot=False)

cluster_order = dendro_top['leaves']
corr_data_ordered = corr_data.iloc[cluster_order, cluster_order]

# Create heatmap
im = ax6_main.imshow(corr_data_ordered, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values with better formatting
for i in range(len(corr_data_ordered)):
    for j in range(len(corr_data_ordered)):
        corr_val = corr_data_ordered.iloc[i, j]
        text_color = 'white' if abs(corr_val) > 0.5 else 'black'
        ax6_main.text(j, i, f'{corr_val:.2f}', ha="center", va="center", 
                     color=text_color, fontsize=9, fontweight='bold')

ax6_main.set_xticks(range(len(corr_data_ordered)))
ax6_main.set_yticks(range(len(corr_data_ordered)))
ax6_main.set_xticklabels(corr_data_ordered.columns, rotation=45, ha='right', fontsize=9)
ax6_main.set_yticklabels(corr_data_ordered.columns, fontsize=9)
ax6_main.set_title('Correlation Matrix with Hierarchical Clustering\n(Dendrograms + Annotated Values)', 
                   fontweight='bold', pad=20, fontsize=14)

# Style dendrograms
ax6_top.set_xticks([])
ax6_top.set_yticks([])
ax6_left.set_xticks([])
ax6_left.set_yticks([])
ax6_top.tick_params(labelbottom=False, labeltop=False)
ax6_left.tick_params(labelleft=False, labelright=False)

# Add colorbar
cbar = plt.colorbar(im, ax=ax6_main, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=11)

# Subplot 7: Grouped bar chart with scatter overlay
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Calculate mean work time by age group and memory type
work_time_stats = df.groupby(['age_group', 'memType'])['WorkTimeInSeconds'].agg(['mean', 'std']).reset_index()

# Create grouped bar chart
age_groups_list = work_time_stats['age_group'].unique()
x = np.arange(len(age_groups_list))
width = 0.25

for i, memtype in enumerate(['recalled', 'imagined', 'retold']):
    subset = work_time_stats[work_time_stats['memType'] == memtype]
    means = [subset[subset['age_group'] == ag]['mean'].iloc[0] if len(subset[subset['age_group'] == ag]) > 0 else 0 
             for ag in age_groups_list]
    
    bars = ax7.bar(x + i*width, means, width, label=memtype, 
                   color=memtype_colors[memtype], alpha=0.8, edgecolor='white', linewidth=0.5)

# Add individual data points with different markers
markers = {'recalled': 'o', 'imagined': 's', 'retold': '^'}
for i, memtype in enumerate(['recalled', 'imagined', 'retold']):
    for j, age_group in enumerate(age_groups_list):
        subset = df[(df['memType'] == memtype) & (df['age_group'] == age_group)]
        if len(subset) > 0:
            # Sample points to avoid overcrowding
            sample_subset = subset.sample(n=min(25, len(subset)), random_state=42)
            y_vals = sample_subset['WorkTimeInSeconds'].values
            x_vals = np.random.normal(j + i*width, 0.06, size=len(y_vals))
            ax7.scatter(x_vals, y_vals, alpha=0.6, s=20, color=memtype_colors[memtype], 
                       marker=markers[memtype], edgecolors='white', linewidth=0.5)

ax7.set_xlabel('Age Group', fontweight='bold', fontsize=12)
ax7.set_ylabel('Work Time (Seconds)', fontweight='bold', fontsize=12)
ax7.set_title('Work Time by Age Group and Memory Type\n(Grouped Bars + Individual Points)', 
              fontweight='bold', pad=20, fontsize=14)
ax7.set_xticks(x + width)
ax7.set_xticklabels(age_groups_list)
ax7.legend(title='Memory Type', fontsize=10)
ax7.grid(True, alpha=0.3)

# Subplot 8: Treemap (improved implementation)
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Calculate hierarchical composition
race_gender_stats = df.groupby(['annotatorRace', 'annotatorGender']).agg({
    'openness': 'mean',
    'AssignmentId': 'count'
}).reset_index()
race_gender_stats.columns = ['race', 'gender', 'mean_openness', 'count']
race_gender_stats = race_gender_stats.sort_values('count', ascending=False)

# Create improved treemap using rectangles
total_count = race_gender_stats['count'].sum()
colors = plt.cm.viridis(race_gender_stats['mean_openness'] / race_gender_stats['mean_openness'].max())

# Calculate positions for treemap
positions = []
current_x, current_y = 0, 0
row_height = 0
max_width = 10

for i, row in race_gender_stats.iterrows():
    area = (row['count'] / total_count) * 100
    width = min(area * 0.5, max_width - current_x)
    height = area / width if width > 0 else 1
    
    if current_x + width > max_width:
        current_x = 0
        current_y += row_height
        row_height = 0
        width = min(area * 0.5, max_width)
        height = area / width if width > 0 else 1
    
    positions.append((current_x, current_y, width, height))
    current_x += width
    row_height = max(row_height, height)

# Draw rectangles
for i, (row_idx, row) in enumerate(race_gender_stats.iterrows()):
    x, y, width, height = positions[i]
    
    rect = patches.Rectangle((x, y), width, height, 
                           facecolor=colors[i], alpha=0.8, edgecolor='white', linewidth=2)
    ax8.add_patch(rect)
    
    # Add labels if rectangle is large enough
    if width * height > 2:
        label_text = f"{row['race'][:8]}\n{row['gender']}\n({row['count']})"
        ax8.text(x + width/2, y + height/2, label_text, 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax8.set_xlim(0, max_width)
ax8.set_ylim(0, current_y + row_height)
ax8.set_title('Hierarchical Composition: Race & Gender\n(Area ∝ Count, Color ∝ Mean Openness)', 
              fontweight='bold', pad=20, fontsize=14)
ax8.set_xticks([])
ax8.set_yticks([])

# Add colorbar for openness
sm = plt.cm.ScalarMappable(cmap='viridis', 
                          norm=plt.Normalize(vmin=race_gender_stats['mean_openness'].min(),
                                           vmax=race_gender_stats['mean_openness'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax8, shrink=0.8)
cbar.set_label('Mean Openness', fontweight='bold', fontsize=11)

# Subplot 9: Network-style plot (properly implemented)
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Create network data from pair IDs
pair_data = df[['recAgnPairId', 'recImgPairId', 'memType']].dropna()
pair_data = pair_data[pair_data['recAgnPairId'] != pair_data['recImgPairId']]

# Sample data to make network manageable
if len(pair_data) > 1000:
    pair_data = pair_data.sample(n=1000, random_state=42)

# Count connections and dominant memory types
connections = {}
node_info = {}

for _, row in pair_data.iterrows():
    agn_id = row['recAgnPairId']
    img_id = row['recImgPairId']
    memtype = row['memType']
    
    # Track node information
    for node_id in [agn_id, img_id]:
        if node_id not in node_info:
            node_info[node_id] = {'count': 0, 'memTypes': []}
        node_info[node_id]['count'] += 1
        node_info[node_id]['memTypes'].append(memtype)
    
    # Track connections
    pair = tuple(sorted([agn_id, img_id]))
    if pair not in connections:
        connections[pair] = {'count': 0, 'memTypes': []}
    connections[pair]['count'] += 1
    connections[pair]['memTypes'].append(memtype)

# Filter to most frequent connections and nodes
top_connections = sorted(connections.items(), key=lambda x: x[1]['count'], reverse=True)[:30]
top_nodes = sorted(node_info.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
top_node_ids = [node[0] for node in top_nodes]

# Create network graph
G = nx.Graph()

# Add nodes with attributes
for node_id, info in node_info.items():
    if node_id in top_node_ids:
        memtype_counts = Counter(info['memTypes'])
        dominant_memtype = memtype_counts.most_common(1)[0][0]
        G.add_node(node_id, memType=dominant_memtype, frequency=info['count'])

# Add edges
for (node1, node2), data in top_connections:
    if node1 in top_node_ids and node2 in top_node_ids:
        G.add_edge(node1, node2, weight=data['count'])

if len(G.nodes()) > 0:
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    for node in G.nodes():
        memtype = G.nodes[node].get('memType', 'recalled')
        frequency = G.nodes[node].get('frequency', 1)
        ax9.scatter(pos[node][0], pos[node][1], 
                   s=frequency * 100, c=memtype_colors[memtype], alpha=0.8,
                   edgecolors='white', linewidth=2)
    
    # Draw edges
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']
        ax9.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=weight)

ax9.set_title('Network of Pair ID Connections\n(Node Size ∝ Frequency, Color = Dominant MemType)', 
              fontweight='bold', pad=20, fontsize=14)
ax9.set_xticks([])
ax9.set_yticks([])

# Create legend for network
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=memtype_colors[mt], markersize=12, 
                             markeredgecolor='white', markeredgewidth=2, label=mt) 
                  for mt in memtype_colors.keys()]
ax9.legend(handles=legend_elements, title='Dominant Memory Type', loc='upper right', fontsize=10)

# Final layout adjustment
plt.tight_layout(pad=2.0)
plt.show()