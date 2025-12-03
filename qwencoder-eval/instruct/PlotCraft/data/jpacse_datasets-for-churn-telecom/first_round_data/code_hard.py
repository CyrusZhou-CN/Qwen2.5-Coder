import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('cell2celltrain.csv')

# Data preprocessing
df = df.dropna(subset=['MonthlyRevenue', 'MonthlyMinutes', 'DroppedCalls', 'UnansweredCalls', 'CustomerCareCalls'])
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# Create the comprehensive 3x3 subplot grid with improved layout
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Improved color palette - more harmonious and professional
colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
churn_colors = {'Yes': '#E63946', 'No': '#457B9D'}
treemap_colors = ['#F8F9FA', '#E9ECEF', '#DEE2E6', '#CED4DA', '#ADB5BD', '#6C757D']

# (1) Top-left: Scatter plot with KDE contours and MARGINAL HISTOGRAMS
ax1 = plt.subplot(3, 3, 1)
sample_data = df.sample(n=2000, random_state=42)

# Main scatter plot
for churn_status in ['Yes', 'No']:
    data_subset = sample_data[sample_data['Churn'] == churn_status]
    ax1.scatter(data_subset['MonthlyMinutes'], data_subset['MonthlyRevenue'], 
               c=churn_colors[churn_status], alpha=0.6, s=20, label=f'Churn: {churn_status}')

# Add KDE contours
try:
    from scipy.stats import gaussian_kde
    x = sample_data['MonthlyMinutes'].values
    y = sample_data['MonthlyRevenue'].values
    xx, yy = np.mgrid[x.min():x.max():(x.max()-x.min())/50, y.min():y.max():(y.max()-y.min())/50]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax1.contour(xx, yy, f, colors='gray', alpha=0.5, linewidths=0.8)
except:
    pass

# Create marginal histograms
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax1_top = fig.add_subplot(gs[0, 0], frame_on=False)
ax1_right = fig.add_subplot(gs[0, 0], frame_on=False)

# Replot main scatter
for churn_status in ['Yes', 'No']:
    data_subset = sample_data[sample_data['Churn'] == churn_status]
    ax1.scatter(data_subset['MonthlyMinutes'], data_subset['MonthlyRevenue'], 
               c=churn_colors[churn_status], alpha=0.6, s=20, label=f'Churn: {churn_status}')

# Top marginal histogram (MonthlyMinutes)
ax1_top.hist(sample_data[sample_data['Churn'] == 'Yes']['MonthlyMinutes'], 
            bins=30, alpha=0.7, color=churn_colors['Yes'], density=True)
ax1_top.hist(sample_data[sample_data['Churn'] == 'No']['MonthlyMinutes'], 
            bins=30, alpha=0.7, color=churn_colors['No'], density=True)
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.axis('off')

# Right marginal histogram (MonthlyRevenue)
ax1_right.hist(sample_data[sample_data['Churn'] == 'Yes']['MonthlyRevenue'], 
              bins=30, alpha=0.7, color=churn_colors['Yes'], density=True, orientation='horizontal')
ax1_right.hist(sample_data[sample_data['Churn'] == 'No']['MonthlyRevenue'], 
              bins=30, alpha=0.7, color=churn_colors['No'], density=True, orientation='horizontal')
ax1_right.set_ylim(ax1.get_ylim())
ax1_right.axis('off')

ax1.set_xlabel('Monthly Minutes', fontweight='bold')
ax1.set_ylabel('Monthly Revenue ($)', fontweight='bold')
ax1.set_title('Revenue vs Minutes with Marginal Distributions', fontweight='bold', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# (2) Top-center: Stacked bar chart with overlaid line plot
ax2 = fig.add_subplot(gs[0, 1])
income_churn = df.groupby(['IncomeGroup', 'Churn']).size().unstack(fill_value=0)
income_rates = df.groupby('IncomeGroup')['Churn_Binary'].agg(['mean', 'std', 'count'])

# Stacked bar chart
income_churn.plot(kind='bar', stacked=True, ax=ax2, color=[churn_colors['No'], churn_colors['Yes']], alpha=0.7)

# Overlaid line plot with error bars
ax2_twin = ax2.twinx()
x_pos = range(len(income_rates))
errors = 1.96 * income_rates['std'] / np.sqrt(income_rates['count'])
ax2_twin.errorbar(x_pos, income_rates['mean'], yerr=errors, 
                 color='#F77F00', marker='o', linewidth=2, markersize=6, capsize=4)

ax2.set_xlabel('Income Group', fontweight='bold')
ax2.set_ylabel('Customer Count', fontweight='bold')
ax2_twin.set_ylabel('Churn Rate', fontweight='bold', color='#F77F00')
ax2.set_title('Churn Distribution by Income Group', fontweight='bold', fontsize=11)
ax2.tick_params(axis='x', rotation=45)

# (3) Top-right: Violin plot with box plot overlay and strip plot
ax3 = fig.add_subplot(gs[0, 2])
# Order credit ratings logically
credit_order = ['1-Highest', '2-High', '3-Good', '4-Medium', '5-Low']
available_ratings = [cr for cr in credit_order if cr in df['CreditRating'].values]
violin_data = [df[df['CreditRating'] == cr]['MonthlyRevenue'].dropna().values for cr in available_ratings]

# Violin plot
parts = ax3.violinplot(violin_data, positions=range(len(available_ratings)), showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#A8DADC')
    pc.set_alpha(0.7)

# Box plot overlay
bp = ax3.boxplot(violin_data, positions=range(len(available_ratings)), widths=0.3, 
                patch_artist=True, boxprops=dict(facecolor='white', alpha=0.8))

# Strip plot (sample points)
for i, data in enumerate(violin_data):
    if len(data) > 100:
        sample_points = np.random.choice(data, 100, replace=False)
    else:
        sample_points = data
    y_jitter = np.random.normal(i, 0.05, len(sample_points))
    ax3.scatter(y_jitter, sample_points, alpha=0.4, s=8, color='#E63946')

ax3.set_xticks(range(len(available_ratings)))
ax3.set_xticklabels(available_ratings, rotation=45)
ax3.set_xlabel('Credit Rating', fontweight='bold')
ax3.set_ylabel('Monthly Revenue ($)', fontweight='bold')
ax3.set_title('Revenue Distribution by Credit Rating', fontweight='bold', fontsize=11)

# (4) Middle-left: CLUSTERMAP with hierarchical clustering dendrogram
ax4 = fig.add_subplot(gs[1, 0])
corr_features = ['MonthlyRevenue', 'MonthlyMinutes', 'DroppedCalls', 'UnansweredCalls', 'CustomerCareCalls']
corr_matrix = df[corr_features].corr()

# Create clustermap using seaborn
plt.subplot(3, 3, 4)
clustermap = sns.clustermap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                           figsize=(6, 6))
plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
clustermap.ax_heatmap.set_title('Feature Correlation with Clustering', fontweight='bold', fontsize=11)

# (5) Middle-center: Parallel coordinates plot with density curves
ax5 = fig.add_subplot(gs[1, 1])
parallel_features = ['MonthlyRevenue', 'MonthsInService', 'Handsets', 'IncomeGroup']
sample_parallel = df[parallel_features + ['Churn']].dropna().sample(n=500, random_state=42)

# Normalize features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(sample_parallel[parallel_features])
normalized_df = pd.DataFrame(normalized_data, columns=parallel_features)
normalized_df['Churn'] = sample_parallel['Churn'].values

# Plot parallel coordinates
for churn_status in ['Yes', 'No']:
    subset = normalized_df[normalized_df['Churn'] == churn_status]
    for idx, row in subset.iterrows():
        ax5.plot(range(len(parallel_features)), row[parallel_features], 
                color=churn_colors[churn_status], alpha=0.3, linewidth=0.8)

# Add density curves for each feature
for i, feature in enumerate(parallel_features):
    for churn_status in ['Yes', 'No']:
        data = normalized_df[normalized_df['Churn'] == churn_status][feature]
        if len(data) > 1:
            density = stats.gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 50)
            ys = density(xs) * 0.3 + i
            ax5.plot(xs * 0.1 + i, ys, color=churn_colors[churn_status], linewidth=2, alpha=0.7)

ax5.set_xticks(range(len(parallel_features)))
ax5.set_xticklabels(parallel_features, rotation=45)
ax5.set_ylabel('Normalized Values', fontweight='bold')
ax5.set_title('Parallel Coordinates with Density Curves', fontweight='bold', fontsize=11)

# (6) Middle-right: TREEMAP with NESTED PIE CHARTS
ax6 = fig.add_subplot(gs[1, 2])
service_areas = df['ServiceArea'].value_counts().head(4)

# Calculate positions for treemap
total_area = 1.0
positions = []
y_pos = 0

for i, (area, count) in enumerate(service_areas.items()):
    height = (count / service_areas.sum()) * 0.9
    width = 0.9
    positions.append((0.05, y_pos, width, height))
    
    # Main rectangle for service area
    rect = Rectangle((0.05, y_pos), width, height, 
                    facecolor=treemap_colors[i % len(treemap_colors)], 
                    edgecolor='white', linewidth=2, alpha=0.7)
    ax6.add_patch(rect)
    
    # Get occupation data for this service area
    area_data = df[df['ServiceArea'] == area]
    occupations = area_data['Occupation'].value_counts().head(4)
    
    if len(occupations) > 0:
        # Create nested pie chart
        pie_center_x = 0.05 + width * 0.75
        pie_center_y = y_pos + height * 0.5
        pie_radius = min(width, height) * 0.15
        
        # Pie chart data
        pie_colors = plt.cm.Set3(np.linspace(0, 1, len(occupations)))
        wedges, texts = ax6.pie(occupations.values, center=(pie_center_x, pie_center_y), 
                               radius=pie_radius, colors=pie_colors, 
                               startangle=90, textprops={'fontsize': 6})
    
    # Label for service area
    ax6.text(0.05 + width * 0.25, y_pos + height/2, f'{area}\n({count})', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    y_pos += height + 0.02

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.set_title('Service Area Treemap with Occupation Breakdown', fontweight='bold', fontsize=11)
ax6.axis('off')

# (7) Bottom-left: NETWORK GRAPH
ax7 = fig.add_subplot(gs[2, 0])
network_features = ['MonthlyMinutes', 'DroppedCalls', 'CustomerCareCalls']
network_sample = df[network_features + ['Churn_Binary']].dropna().sample(n=100, random_state=42)

# Create network graph
G = nx.Graph()

# Add nodes
for idx, row in network_sample.iterrows():
    G.add_node(idx, **row.to_dict())

# Add edges based on similarity
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(network_sample[network_features], metric='euclidean'))
threshold = np.percentile(distances, 10)

for i in range(len(network_sample)):
    for j in range(i+1, len(network_sample)):
        if distances[i, j] < threshold:
            G.add_edge(network_sample.index[i], network_sample.index[j])

# Layout and draw network
pos = nx.spring_layout(G, k=0.5, iterations=50)
node_colors = [churn_colors['Yes'] if network_sample.loc[node, 'Churn_Binary'] == 1 
               else churn_colors['No'] for node in G.nodes()]
node_sizes = [network_sample.loc[node, 'CustomerCareCalls'] * 50 + 50 for node in G.nodes()]

nx.draw(G, pos, ax=ax7, node_color=node_colors, node_size=node_sizes, 
        edge_color='gray', alpha=0.7, with_labels=False)

ax7.set_title('Customer Network Graph by Usage Patterns', fontweight='bold', fontsize=11)

# (8) Bottom-center: Radar chart with OVERLAID AREA PLOT
ax8 = fig.add_subplot(gs[2, 1], projection='polar')
radar_features = ['MonthlyRevenue', 'MonthlyMinutes', 'DroppedCalls', 
                 'CustomerCareCalls', 'MonthsInService', 'Handsets']

# Calculate means for churned vs non-churned
churned_means = []
non_churned_means = []

for feature in radar_features:
    churned_mean = df[df['Churn'] == 'Yes'][feature].mean()
    non_churned_mean = df[df['Churn'] == 'No'][feature].mean()
    
    # Normalize to 0-1 scale
    max_val = df[feature].max()
    min_val = df[feature].min()
    churned_means.append((churned_mean - min_val) / (max_val - min_val))
    non_churned_means.append((non_churned_mean - min_val) / (max_val - min_val))

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

churned_means += churned_means[:1]
non_churned_means += non_churned_means[:1]

# Plot with overlaid area
ax8.plot(angles, churned_means, 'o-', linewidth=2, label='Churned', color='#E63946')
ax8.fill(angles, churned_means, alpha=0.25, color='#E63946')  # Area plot overlay
ax8.plot(angles, non_churned_means, 'o-', linewidth=2, label='Non-Churned', color='#457B9D')
ax8.fill(angles, non_churned_means, alpha=0.25, color='#457B9D')  # Area plot overlay

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(radar_features, fontsize=9)
ax8.set_ylim(0, 1)
ax8.set_title('Customer Profile Radar with Area Overlay', fontweight='bold', fontsize=11, pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# (9) Bottom-right: 3D SCATTER PLOT
ax9 = fig.add_subplot(gs[2, 2], projection='3d')
cluster_features = ['MonthlyRevenue', 'MonthlyMinutes', 'MonthsInService']
cluster_sample = df[cluster_features].dropna().sample(n=500, random_state=42)

# Perform 3D clustering
kmeans_3d = KMeans(n_clusters=3, random_state=42)
clusters_3d = kmeans_3d.fit_predict(cluster_sample)

# 3D scatter plot
for i in range(3):
    cluster_data = cluster_sample[clusters_3d == i]
    ax9.scatter(cluster_data['MonthlyRevenue'], cluster_data['MonthlyMinutes'], 
               cluster_data['MonthsInService'], c=colors[i], alpha=0.6, s=30, label=f'Cluster {i+1}')

# Plot centroids with smaller markers
centroids_3d = kmeans_3d.cluster_centers_
ax9.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], 
           c='black', marker='x', s=100, linewidths=2, label='Centroids')  # Reduced size

ax9.set_xlabel('Monthly Revenue', fontweight='bold')
ax9.set_ylabel('Monthly Minutes', fontweight='bold')
ax9.set_zlabel('Months in Service', fontweight='bold')
ax9.set_title('3D Customer Clusters', fontweight='bold', fontsize=11)
ax9.legend()

# Overall layout adjustment with improved spacing
plt.tight_layout(pad=3.0)
plt.suptitle('Comprehensive Telecom Customer Segmentation Analysis', 
             fontsize=16, fontweight='bold', y=0.98)  # Reduced font size and adjusted position
plt.subplots_adjust(top=0.94, hspace=0.4, wspace=0.3)
plt.show()