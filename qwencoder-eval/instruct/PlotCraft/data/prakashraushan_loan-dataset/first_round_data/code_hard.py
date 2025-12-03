import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx
import squarify
from matplotlib.patches import Ellipse
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Find the correct CSV file
csv_files = glob.glob('*.csv')
loan_file = None
for file in csv_files:
    if 'loan' in file.lower():
        loan_file = file
        break

if loan_file is None:
    # Try common variations
    possible_names = ['LoanDataset.csv', 'LoansDatasest.csv', 'loan_dataset.csv', 'loans_dataset.csv']
    for name in possible_names:
        if os.path.exists(name):
            loan_file = name
            break

if loan_file is None:
    raise FileNotFoundError("Could not find loan dataset CSV file")

# Load and preprocess data
df = pd.read_csv(loan_file)

# Clean and convert data types
df['customer_income'] = pd.to_numeric(df['customer_income'].astype(str).str.replace(',', '').str.replace('£', ''), errors='coerce')
df['loan_amnt'] = pd.to_numeric(df['loan_amnt'].astype(str).str.replace('£', '').str.replace(',', ''), errors='coerce')
df['historical_default'] = df['historical_default'].fillna('N')
df = df.dropna(subset=['customer_id', 'customer_age', 'customer_income', 'Current_loan_status'])

# Create binary default indicator
df['is_default'] = (df['Current_loan_status'] == 'DEFAULT').astype(int)

# Set up the 3x3 subplot grid with white background
fig = plt.figure(figsize=(24, 20), facecolor='white')
fig.suptitle('Comprehensive Loan Default Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

# 1. Top-left: Scatter plot with marginal histograms and density contours
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Sample data for performance
sample_df = df.sample(n=min(5000, len(df)), random_state=42)
colors = ['#2E86AB' if x == 0 else '#F24236' for x in sample_df['is_default']]

scatter = ax1.scatter(sample_df['customer_age'], sample_df['customer_income'], 
                     c=colors, alpha=0.6, s=20)

# Add simple density representation using hexbin
ax1.hexbin(sample_df['customer_age'], sample_df['customer_income'], 
          gridsize=15, alpha=0.3, cmap='Blues')

ax1.set_xlabel('Customer Age', fontweight='bold')
ax1.set_ylabel('Customer Income', fontweight='bold')
ax1.set_title('Age vs Income Distribution by Default Status', fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                         markersize=8, label='No Default'),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='#F24236', 
                         markersize=8, label='Default')]
ax1.legend(handles=legend_elements, loc='upper right')

# 2. Top-center: Grouped violin plots with box plots overlay
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Create grouped data for violin plot
grade_default_data = []
grade_labels = []
positions = []
pos = 0

for grade in sorted(df['loan_grade'].unique()):
    default_rates = df[df['loan_grade'] == grade]['loan_int_rate']
    no_default_rates = df[(df['loan_grade'] == grade) & (df['is_default'] == 0)]['loan_int_rate']
    default_rates_only = df[(df['loan_grade'] == grade) & (df['is_default'] == 1)]['loan_int_rate']
    
    if len(no_default_rates) > 0:
        grade_default_data.append(no_default_rates.values)
        positions.append(pos)
        grade_labels.append(f'{grade}\nNo Default')
        pos += 1
    
    if len(default_rates_only) > 0:
        grade_default_data.append(default_rates_only.values)
        positions.append(pos)
        grade_labels.append(f'{grade}\nDefault')
        pos += 1

if grade_default_data:
    parts = ax2.violinplot(grade_default_data, positions=positions, widths=0.6, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        if 'Default' in grade_labels[i] and 'No Default' not in grade_labels[i]:
            pc.set_facecolor('#F24236')
        else:
            pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.7)

ax2.set_xticks(positions[::2])
ax2.set_xticklabels([grade_labels[i].split('\n')[0] for i in range(0, len(grade_labels), 2)], fontweight='bold')
ax2.set_ylabel('Loan Interest Rate (%)', fontweight='bold')
ax2.set_title('Interest Rates by Loan Grade and Default Status', fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3)

# 3. Top-right: Stacked bar chart with line plot overlay
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Calculate data for stacked bar and line plot
home_intent = df.groupby(['home_ownership', 'loan_intent']).size().unstack(fill_value=0)
avg_amounts = df.groupby('home_ownership')['loan_amnt'].mean()

# Create stacked bar chart
home_intent.plot(kind='bar', stacked=True, ax=ax3, colormap='Set3', alpha=0.8)

# Add line plot for average amounts
ax3_twin = ax3.twinx()
ax3_twin.plot(range(len(avg_amounts)), avg_amounts.values, 'ro-', linewidth=3, markersize=8, label='Avg Loan Amount')
ax3_twin.set_ylabel('Average Loan Amount (£)', fontweight='bold', color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')

ax3.set_xlabel('Home Ownership Status', fontweight='bold')
ax3.set_ylabel('Number of Loans', fontweight='bold')
ax3.set_title('Loan Intent Distribution by Home Ownership', fontweight='bold', pad=20)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. Middle-left: Radar chart
ax4 = plt.subplot(3, 3, 4, projection='polar')
ax4.set_facecolor('white')

# Prepare numerical features for radar chart
numerical_features = ['customer_age', 'customer_income', 'employment_duration', 
                     'loan_amnt', 'loan_int_rate', 'term_years', 'cred_hist_length']

# Calculate normalized means for defaulted and non-defaulted loans
default_means = []
no_default_means = []

for feature in numerical_features:
    if feature in df.columns:
        default_mean = df[df['is_default'] == 1][feature].mean()
        no_default_mean = df[df['is_default'] == 0][feature].mean()
        
        # Normalize to 0-1 scale
        max_val = df[feature].max()
        min_val = df[feature].min()
        
        if max_val != min_val:
            default_norm = (default_mean - min_val) / (max_val - min_val)
            no_default_norm = (no_default_mean - min_val) / (max_val - min_val)
        else:
            default_norm = 0.5
            no_default_norm = 0.5
        
        default_means.append(default_norm)
        no_default_means.append(no_default_norm)

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(default_means), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

default_means += default_means[:1]
no_default_means += no_default_means[:1]

ax4.plot(angles, default_means, 'o-', linewidth=2, label='Default', color='#F24236')
ax4.fill(angles, default_means, alpha=0.25, color='#F24236')
ax4.plot(angles, no_default_means, 'o-', linewidth=2, label='No Default', color='#2E86AB')
ax4.fill(angles, no_default_means, alpha=0.25, color='#2E86AB')

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels([f.replace('_', ' ').title() for f in numerical_features[:len(default_means)-1]], fontweight='bold')
ax4.set_title('Feature Comparison: Default vs No Default', fontweight='bold', pad=30)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 5. Middle-center: Correlation heatmap with hierarchical clustering
ax5 = plt.subplot(3, 3, 5)
ax5.set_facecolor('white')

# Calculate correlation matrix for numerical features
corr_features = ['customer_age', 'customer_income', 'employment_duration', 
                'loan_amnt', 'loan_int_rate', 'term_years', 'cred_hist_length', 'is_default']
available_features = [f for f in corr_features if f in df.columns]
corr_matrix = df[available_features].corr()

# Create hierarchical clustering
try:
    linkage_matrix = linkage(pdist(corr_matrix), method='ward')
    dendro_order = dendrogram(linkage_matrix, no_plot=True)['leaves']
    corr_ordered = corr_matrix.iloc[dendro_order, dendro_order]
except:
    corr_ordered = corr_matrix

# Create heatmap
im = ax5.imshow(corr_ordered, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax5.set_xticks(range(len(corr_ordered.columns)))
ax5.set_yticks(range(len(corr_ordered.columns)))
ax5.set_xticklabels([col.replace('_', ' ').title() for col in corr_ordered.columns], 
                   rotation=45, ha='right', fontweight='bold')
ax5.set_yticklabels([col.replace('_', ' ').title() for col in corr_ordered.columns], 
                   fontweight='bold')
ax5.set_title('Hierarchical Clustering of Feature Correlations', fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# 6. Middle-right: Parallel coordinates plot
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Prepare data for parallel coordinates
parallel_features = ['loan_grade', 'term_years', 'employment_duration', 'cred_hist_length']
sample_parallel = df.sample(n=min(1000, len(df)), random_state=42)

# Convert categorical to numerical
grade_map = {grade: i for i, grade in enumerate(sorted(df['loan_grade'].unique()))}
sample_parallel = sample_parallel.copy()
sample_parallel['loan_grade_num'] = sample_parallel['loan_grade'].map(grade_map)

# Normalize features
parallel_data = sample_parallel[['loan_grade_num', 'term_years', 'employment_duration', 'cred_hist_length']].copy()
for col in parallel_data.columns:
    col_min = parallel_data[col].min()
    col_max = parallel_data[col].max()
    if col_max != col_min:
        parallel_data[col] = (parallel_data[col] - col_min) / (col_max - col_min)
    else:
        parallel_data[col] = 0.5

# Plot parallel coordinates
for idx, row in parallel_data.iterrows():
    color = '#F24236' if sample_parallel.loc[idx, 'is_default'] == 1 else '#2E86AB'
    alpha = 0.6 if sample_parallel.loc[idx, 'is_default'] == 1 else 0.3
    ax6.plot(range(len(parallel_data.columns)), row.values, color=color, alpha=alpha, linewidth=0.5)

ax6.set_xticks(range(len(parallel_data.columns)))
ax6.set_xticklabels(['Loan Grade', 'Term Years', 'Employment Duration', 'Credit History'], 
                   fontweight='bold')
ax6.set_ylabel('Normalized Values', fontweight='bold')
ax6.set_title('Parallel Coordinates: Multi-dimensional Relationships', fontweight='bold', pad=20)
ax6.grid(True, alpha=0.3)

# Add legend
legend_elements = [Line2D([0], [0], color='#2E86AB', linewidth=2, label='No Default'),
                  Line2D([0], [0], color='#F24236', linewidth=2, label='Default')]
ax6.legend(handles=legend_elements, loc='upper right')

# 7. Bottom-left: Network graph
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Create network graph
G = nx.Graph()

# Add nodes
intents = df['loan_intent'].unique()[:5]  # Limit for readability
ownerships = df['home_ownership'].unique()
grades = df['loan_grade'].unique()[:4]  # Limit for readability

for intent in intents:
    G.add_node(f"I_{intent[:4]}", node_type='intent')
for ownership in ownerships:
    G.add_node(f"O_{ownership[:4]}", node_type='ownership')
for grade in grades:
    G.add_node(f"G_{grade}", node_type='grade')

# Add edges with default rates as weights
for intent in intents:
    for ownership in ownerships:
        subset = df[(df['loan_intent'] == intent) & (df['home_ownership'] == ownership)]
        if len(subset) > 10:
            default_rate = subset['is_default'].mean()
            G.add_edge(f"I_{intent[:4]}", f"O_{ownership[:4]}", weight=default_rate)

for grade in grades:
    for ownership in ownerships:
        subset = df[(df['loan_grade'] == grade) & (df['home_ownership'] == ownership)]
        if len(subset) > 10:
            default_rate = subset['is_default'].mean()
            G.add_edge(f"G_{grade}", f"O_{ownership[:4]}", weight=default_rate)

# Create layout
pos = nx.spring_layout(G, k=1, iterations=50)

# Draw network
node_colors = []
for node in G.nodes():
    if 'I_' in node:
        node_colors.append('#FF9999')
    elif 'O_' in node:
        node_colors.append('#99FF99')
    else:
        node_colors.append('#9999FF')

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax7)
nx.draw_networkx_labels(G, pos, {node: node.split('_')[1] for node in G.nodes()}, 
                       font_size=8, font_weight='bold', ax=ax7)

# Draw edges with thickness based on default rates
if G.edges():
    edges = G.edges()
    weights = [G[u][v]['weight'] * 10 + 1 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, ax=ax7)

ax7.set_title('Network: Loan Intent, Ownership & Grade Relationships', fontweight='bold', pad=20)
ax7.axis('off')

# 8. Bottom-center: Treemap
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Prepare treemap data
treemap_data = df.groupby(['loan_grade', 'loan_intent']).agg({
    'loan_amnt': 'sum',
    'is_default': 'mean'
}).reset_index()

# Filter for top combinations
treemap_data = treemap_data.nlargest(12, 'loan_amnt')

# Create treemap
sizes = treemap_data['loan_amnt'].values
labels = [f"{row['loan_grade']}\n{row['loan_intent'][:6]}" for _, row in treemap_data.iterrows()]
colors = treemap_data['is_default'].values

# Normalize colors
norm_colors = plt.cm.RdYlBu_r(colors)

squarify.plot(sizes=sizes, label=labels, color=norm_colors, alpha=0.8, ax=ax8, text_kwargs={'fontsize': 8})
ax8.set_title('Treemap: Loan Amounts by Grade & Intent\n(Color = Default Rate)', fontweight='bold', pad=20)
ax8.axis('off')

# 9. Bottom-right: PCA cluster scatter plot
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Prepare data for PCA
pca_features = ['customer_age', 'customer_income', 'employment_duration', 
               'loan_amnt', 'loan_int_rate', 'term_years', 'cred_hist_length']
available_pca_features = [f for f in pca_features if f in df.columns]
pca_data = df[available_pca_features].dropna()
pca_labels = df.loc[pca_data.index, 'is_default']
pca_grades = df.loc[pca_data.index, 'loan_grade']

# Sample for performance
if len(pca_data) > 2000:
    sample_idx = np.random.choice(pca_data.index, 2000, replace=False)
    pca_data = pca_data.loc[sample_idx]
    pca_labels = pca_labels.loc[sample_idx]
    pca_grades = pca_grades.loc[sample_idx]

# Standardize and apply PCA
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_scaled)

# Create scatter plot
colors = ['#2E86AB' if x == 0 else '#F24236' for x in pca_labels]
markers = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D', 'E': 'v', 'F': 'p', 'G': '*'}

for grade in pca_grades.unique():
    mask = pca_grades == grade
    marker = markers.get(grade, 'o')
    grade_colors = [colors[i] for i in range(len(colors)) if mask.iloc[i]]
    ax9.scatter(pca_result[mask, 0], pca_result[mask, 1], 
               c=grade_colors, marker=marker, s=30, alpha=0.6, label=f'Grade {grade}')

# Add cluster centroids
try:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_result)
    centroids = kmeans.cluster_centers_
    ax9.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, linewidths=3)
except:
    pass

ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
ax9.set_title('PCA Clustering with Loan Grades', fontweight='bold', pad=20)
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax9.grid(True, alpha=0.3)

# Final layout adjustment
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
plt.savefig('comprehensive_loan_analysis.png', dpi=300, bbox_inches='tight')
plt.show()