import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('csv_data.csv')

# Data preprocessing functions
def extract_employee_count(emp_str):
    """Extract numeric employee count from string ranges"""
    if pd.isna(emp_str):
        return 50
    emp_str = str(emp_str).strip()
    if emp_str == '1':
        return 1
    elif emp_str == '2-10':
        return 6
    elif emp_str == '11-50':
        return 30
    elif emp_str == '51-200':
        return 125
    elif emp_str == '201-500':
        return 350
    elif emp_str == '501-1000':
        return 750
    elif emp_str == '1001-5000':
        return 3000
    elif emp_str == '5001-10000':
        return 7500
    else:
        return 100

def extract_job_count(jobs_str):
    """Extract total job count from jobs string"""
    if pd.isna(jobs_str):
        return 5
    try:
        import re
        numbers = re.findall(r'\((\d+)\)', str(jobs_str))
        total = sum(int(n) for n in numbers)
        return max(total, 1)
    except:
        return 5

def extract_funding_status(tags_str):
    """Extract funding status from tags"""
    if pd.isna(tags_str):
        return 'Unknown'
    tags_str = str(tags_str).lower()
    if 'recently funded' in tags_str:
        return 'Recently Funded'
    elif 'actively hiring' in tags_str:
        return 'Actively Hiring'
    else:
        return 'Other'

def count_locations(loc_str):
    """Count number of locations"""
    if pd.isna(loc_str):
        return 1
    return max(len(str(loc_str).split(',')), 1)

def count_job_categories(jobs_str):
    """Count number of different job categories"""
    if pd.isna(jobs_str):
        return 1
    try:
        import re
        categories = re.findall(r'\(([^:]+):', str(jobs_str))
        return max(len(set(categories)), 1)
    except:
        return 1

def extract_shared_investors(tags_str):
    """Extract shared investor information from tags"""
    if pd.isna(tags_str):
        return []
    tags_str = str(tags_str).lower()
    investors = []
    if 'same investor as airbnb' in tags_str:
        investors.append('airbnb_investor')
    if 'same investor as uber' in tags_str:
        investors.append('uber_investor')
    if 'same investor as apple' in tags_str:
        investors.append('apple_investor')
    if 'same investor as snapchat' in tags_str:
        investors.append('snapchat_investor')
    return investors

# Apply preprocessing
df['employee_count'] = df['employees'].apply(extract_employee_count)
df['job_count'] = df['jobs'].apply(extract_job_count)
df['funding_status'] = df['tags'].apply(extract_funding_status)
df['location_count'] = df['locations'].apply(count_locations)
df['job_categories'] = df['jobs'].apply(count_job_categories)
df['shared_investors'] = df['tags'].apply(extract_shared_investors)

# Clean and prepare industry data
df['industries'] = df['industries'].fillna('Unknown')
df['primary_industry'] = df['industries'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
df['first_location'] = df['locations'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')

# Create size categories
df['size_category'] = pd.cut(df['employee_count'], 
                            bins=[0, 10, 50, 200, 1000, 10000], 
                            labels=['Micro', 'Small', 'Medium', 'Large', 'Enterprise'])

# Ensure we have valid data
df = df[(df['employee_count'] > 0) & (df['job_count'] > 0)].copy()

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Subplot (0,0): Scatter plot with bubble sizes and K-means clustering
ax1 = plt.subplot(3, 3, 1)
cluster_data = df[['employee_count', 'job_count']].copy()

if len(cluster_data) > 0:
    # Perform K-means clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=min(4, len(cluster_data)), random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Create scatter plot with bubble sizes proportional to employee count
    funding_colors = {'Recently Funded': '#FF6B6B', 'Actively Hiring': '#4ECDC4', 'Other': '#45B7D1'}
    for status in funding_colors.keys():
        mask = df['funding_status'] == status
        if mask.sum() > 0:
            # Bubble sizes proportional to employee count
            bubble_sizes = df[mask]['employee_count'] * 0.5  # Scale for visibility
            ax1.scatter(df[mask]['employee_count'], df[mask]['job_count'], 
                       s=bubble_sizes, alpha=0.6, 
                       c=funding_colors[status], label=status, edgecolors='white', linewidth=0.5)

    # Add cluster boundaries
    if len(cluster_data) > 10:
        x_range = np.linspace(df['employee_count'].min(), df['employee_count'].max(), 50)
        y_range = np.linspace(df['job_count'].min(), df['job_count'].max(), 50)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid_points)
        grid_clusters = kmeans.predict(grid_scaled)
        grid_clusters = grid_clusters.reshape(xx.shape)
        ax1.contour(xx, yy, grid_clusters, levels=3, colors='black', alpha=0.4, linewidths=1.5)

# Use log scale for better distribution
ax1.set_xscale('log')
ax1.set_xlabel('Employee Count (log scale)', fontweight='bold')
ax1.set_ylabel('Job Openings', fontweight='bold')
ax1.set_title('Company Clustering by Size and Hiring Activity', fontweight='bold', fontsize=12)
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Subplot (0,1): Dendrogram with heatmap overlay
ax2 = plt.subplot(3, 3, 2)
industry_counts = df['primary_industry'].value_counts()
top_industries = industry_counts.head(8).index.tolist()
industry_subset = df[df['primary_industry'].isin(top_industries)]

if len(top_industries) > 1:
    # Create feature matrix for industries
    industry_features = []
    valid_industries = []
    for industry in top_industries:
        industry_data = industry_subset[industry_subset['primary_industry'] == industry]
        if len(industry_data) > 0:
            features = [
                industry_data['employee_count'].mean(),
                industry_data['job_count'].mean(),
                industry_data['location_count'].mean(),
                len(industry_data)
            ]
            industry_features.append(features)
            valid_industries.append(industry)

    if len(industry_features) > 1:
        industry_features = np.array(industry_features)
        industry_features = StandardScaler().fit_transform(industry_features)

        # Calculate distance matrix for heatmap overlay
        distance_matrix_full = squareform(pdist(industry_features, metric='euclidean'))
        
        # Create heatmap overlay first
        im = ax2.imshow(distance_matrix_full, cmap='Blues', alpha=0.6, aspect='auto')
        
        # Calculate linkage for dendrogram
        distance_vector = pdist(industry_features, metric='euclidean')
        linkage_matrix = linkage(distance_vector, method='ward')

        # Create dendrogram on top
        dend = dendrogram(linkage_matrix, labels=[ind[:8] for ind in valid_industries], 
                         ax=ax2, orientation='top', leaf_rotation=45, leaf_font_size=8,
                         color_threshold=0)

ax2.set_title('Industry Hierarchical Clustering with Distance Heatmap', fontweight='bold', fontsize=12)

# Subplot (0,2): Network graph with proper node and edge sizing
ax3 = plt.subplot(3, 3, 3)
G = nx.Graph()
companies_sample = df.sample(min(25, len(df)), random_state=42)

# Add nodes with job count as attribute
for idx, company in companies_sample.iterrows():
    G.add_node(company['company_name'][:12], 
              job_count=company['job_count'],
              industry=company['primary_industry'])

# Add edges based on shared investors with connection strength
companies_list = list(companies_sample.itertuples())
for i, comp1 in enumerate(companies_list):
    for j, comp2 in enumerate(companies_list):
        if i < j:
            shared_count = len(set(comp1.shared_investors) & set(comp2.shared_investors))
            if shared_count > 0 or comp1.primary_industry == comp2.primary_industry:
                weight = shared_count + (1 if comp1.primary_industry == comp2.primary_industry else 0)
                G.add_edge(comp1.company_name[:12], comp2.company_name[:12], weight=weight)

if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Node sizes proportional to job openings
    node_sizes = [G.nodes[node]['job_count'] * 8 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7, ax=ax3)
    
    # Edge thickness proportional to connection strength
    if len(G.edges()) > 0:
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [w * 3 / max_weight for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax3)

ax3.set_title('Company Network: Shared Investors & Industries', fontweight='bold', fontsize=12)
ax3.axis('off')

# Subplot (1,0): Parallel coordinates plot with violin overlays
ax4 = plt.subplot(3, 3, 4)
# Prepare data for parallel coordinates
sample_companies = df.sample(min(100, len(df)), random_state=42)
parallel_features = ['employee_count', 'job_count', 'location_count', 'job_categories']
parallel_data = sample_companies[parallel_features].copy()

# Normalize data
scaler_parallel = StandardScaler()
parallel_normalized = scaler_parallel.fit_transform(parallel_data)

# Create parallel coordinates plot
x_positions = np.arange(len(parallel_features))
for i in range(len(parallel_normalized)):
    ax4.plot(x_positions, parallel_normalized[i], alpha=0.3, color='steelblue', linewidth=0.5)

# Add violin plots overlay
violin_data = [parallel_normalized[:, i] for i in range(len(parallel_features))]
violin_parts = ax4.violinplot(violin_data, positions=x_positions, widths=0.4)

for pc in violin_parts['bodies']:
    pc.set_alpha(0.6)
    pc.set_facecolor('lightcoral')

ax4.set_xticks(x_positions)
ax4.set_xticklabels(['Employees', 'Jobs', 'Locations', 'Job Types'], fontweight='bold')
ax4.set_ylabel('Standardized Values', fontweight='bold')
ax4.set_title('Parallel Coordinates: Company Profiles', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Subplot (1,1): Treemap with mini bar charts
ax5 = plt.subplot(3, 3, 5)
industry_stats = df.groupby('primary_industry').agg({
    'company_name': 'count',
    'employee_count': 'mean'
}).round(2)
industry_stats.columns = ['company_count', 'avg_employees']
industry_stats = industry_stats.sort_values('company_count', ascending=False).head(6)

# Create treemap with mini bar charts
colors = plt.cm.Set3(np.linspace(0, 1, len(industry_stats)))
y_pos = 0
total_companies = industry_stats['company_count'].sum()

for i, (industry, stats) in enumerate(industry_stats.iterrows()):
    height = stats['company_count'] / total_companies * 0.9  # Leave space for bars
    
    # Main rectangle
    rect = Rectangle((0, y_pos), 0.7, height, facecolor=colors[i], alpha=0.7, edgecolor='white', linewidth=2)
    ax5.add_patch(rect)
    
    # Industry label
    industry_short = industry[:12] + '...' if len(industry) > 12 else industry
    ax5.text(0.35, y_pos + height/2, f'{industry_short}\n({int(stats["company_count"])})', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Mini bar chart showing size distribution
    industry_data = df[df['primary_industry'] == industry]
    size_dist = industry_data['size_category'].value_counts()
    
    if len(size_dist) > 0:
        bar_width = 0.25 / len(size_dist)
        bar_x_start = 0.75
        for j, (size_cat, count) in enumerate(size_dist.items()):
            bar_height = (count / size_dist.sum()) * height * 0.8
            bar_y = y_pos + height * 0.1 + j * bar_height
            mini_rect = Rectangle((bar_x_start, bar_y), bar_width, bar_height, 
                                facecolor='darkblue', alpha=0.8, edgecolor='white')
            ax5.add_patch(mini_rect)
    
    y_pos += height

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 0.9)
ax5.set_title('Industry Treemap with Size Distribution', fontweight='bold', fontsize=12)
ax5.axis('off')

# Subplot (1,2): Radar chart with scatter overlay
ax6 = plt.subplot(3, 3, 6, projection='polar')
location_stats = df.groupby('first_location').agg({
    'employee_count': 'mean',
    'job_count': 'mean',
    'job_categories': 'mean',
    'company_name': 'count'
}).round(2)

top_locations = location_stats.nlargest(5, 'company_name')
categories = ['Avg Employees', 'Avg Jobs', 'Avg Job Types', 'Company Count']

if len(top_locations) > 0:
    # Normalize data for radar chart
    normalized_data = StandardScaler().fit_transform(top_locations.values)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot radar lines for each location
    for i, (location, _) in enumerate(top_locations.iterrows()):
        values = normalized_data[i].tolist()
        values += values[:1]
        location_short = location[:8] + '...' if len(location) > 8 else location
        ax6.plot(angles, values, 'o-', linewidth=2, label=location_short, color=colors_radar[i])
        ax6.fill(angles, values, alpha=0.15, color=colors_radar[i])
        
        # Add individual company scatter points for within-region variation
        location_companies = df[df['first_location'] == location].sample(min(5, len(df[df['first_location'] == location])))
        for _, company in location_companies.iterrows():
            company_values = [company['employee_count']/1000, company['job_count']/50, 
                            company['job_categories']/5, 1]
            company_values += company_values[:1]
            ax6.scatter(angles[:-1], company_values[:-1], alpha=0.4, s=20, color=colors_radar[i])

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontweight='bold')
    ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

ax6.set_title('Location Profiles with Company Variation', fontweight='bold', fontsize=12, pad=20)

# Subplot (2,0): Stacked area chart with line overlay
ax7 = plt.subplot(3, 3, 7)
size_job_diversity = df.groupby('size_category', observed=True).agg({
    'job_categories': ['mean', 'count'],
    'job_count': 'sum'
}).round(2)

size_job_diversity.columns = ['avg_diversity', 'company_count', 'total_jobs']

if len(size_job_diversity) > 0:
    x = range(len(size_job_diversity))
    
    # Create stacked areas
    cumulative_jobs = np.cumsum(size_job_diversity['total_jobs'])
    ax7.fill_between(x, 0, cumulative_jobs, alpha=0.6, color='lightblue', label='Cumulative Jobs')
    
    # Add diversity index line overlay
    ax7_twin = ax7.twinx()
    ax7_twin.plot(x, size_job_diversity['avg_diversity'], 'o-', color='darkred', 
                  linewidth=3, markersize=8, label='Diversity Index')

    ax7.set_xticks(x)
    ax7.set_xticklabels(size_job_diversity.index, fontweight='bold')
    ax7.set_xlabel('Company Size Category', fontweight='bold')
    ax7.set_ylabel('Cumulative Job Count', fontweight='bold')
    ax7_twin.set_ylabel('Job Diversity Index', fontweight='bold')
    
    ax7.legend(loc='upper left')
    ax7_twin.legend(loc='upper right')

ax7.set_title('Job Distribution & Diversity by Company Size', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3)

# Subplot (2,1): Correlation heatmap with dendrograms
ax8 = plt.subplot(3, 3, 8)
corr_data = df[['employee_count', 'job_count', 'location_count', 'job_categories']].corr()

# Create clustered heatmap with dendrograms
g = sns.clustermap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, cbar_kws={'shrink': 0.8}, 
                   figsize=(6, 6))
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

# Copy the clustered heatmap to our subplot
ax8.clear()
sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=ax8, cbar_kws={'shrink': 0.8})
ax8.set_title('Correlation Matrix with Hierarchical Clustering', fontweight='bold', fontsize=12)

# Subplot (2,2): Sankey diagram with box plots
ax9 = plt.subplot(3, 3, 9)

# Create flow data for Sankey-style visualization
top_industries_flow = df['primary_industry'].value_counts().head(4).index
flow_data = []

# Calculate flows from industries to sizes to hiring levels
for industry in top_industries_flow:
    industry_data = df[df['primary_industry'] == industry]
    for size_cat in ['Small', 'Medium', 'Large']:
        size_data = industry_data[industry_data['size_category'] == size_cat]
        if len(size_data) > 0:
            avg_hiring = size_data['job_count'].mean()
            flow_data.append({
                'industry': industry[:8],
                'size': size_cat,
                'hiring_level': 'High' if avg_hiring > 50 else 'Low',
                'count': len(size_data),
                'job_openings': size_data['job_count'].tolist()
            })

# Create simplified Sankey-style flow using stacked bars
if flow_data:
    industries = list(set([f['industry'] for f in flow_data]))
    x_pos = np.arange(len(industries))
    
    # Create stacked bars showing flow
    bottom_high = np.zeros(len(industries))
    bottom_low = np.zeros(len(industries))
    
    for i, industry in enumerate(industries):
        industry_flows = [f for f in flow_data if f['industry'] == industry]
        high_count = sum([f['count'] for f in industry_flows if f['hiring_level'] == 'High'])
        low_count = sum([f['count'] for f in industry_flows if f['hiring_level'] == 'Low'])
        
        ax9.bar(i, high_count, bottom=bottom_high[i], color='darkgreen', alpha=0.7, label='High Hiring' if i == 0 else "")
        ax9.bar(i, low_count, bottom=bottom_high[i] + high_count, color='lightgreen', alpha=0.7, label='Low Hiring' if i == 0 else "")
        
        # Add box plots for job openings distribution
        industry_jobs = []
        for f in industry_flows:
            industry_jobs.extend(f['job_openings'])
        
        if industry_jobs:
            # Create mini box plot
            q1, median, q3 = np.percentile(industry_jobs, [25, 50, 75])
            ax9.plot([i-0.2, i+0.2], [median, median], 'r-', linewidth=2)
            ax9.plot([i, i], [q1, q3], 'r-', linewidth=1)

    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(industries, rotation=30, ha='right', fontweight='bold')
    ax9.set_xlabel('Industry', fontweight='bold')
    ax9.set_ylabel('Company Count', fontweight='bold')
    ax9.set_title('Industry-Size-Hiring Flow with Job Distribution', fontweight='bold', fontsize=12)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

# Adjust layout with more spacing
plt.tight_layout(pad=3.0)
plt.savefig('startup_clustering_analysis_refined.png', dpi=300, bbox_inches='tight')
plt.show()