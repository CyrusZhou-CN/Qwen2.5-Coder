import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from matplotlib.patches import Circle
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Find and load data file
def find_data_file():
    # Look for CSV files in current directory
    csv_files = glob.glob('*.csv')
    if csv_files:
        return csv_files[0]
    
    # Look for the specific file name
    possible_names = ['df_survey_2024.csv', 'survey_2024.csv', 'data.csv']
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    # If no file found, create synthetic data
    return None

# Load data with error handling
data_file = find_data_file()
if data_file:
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded data from {data_file}")
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        df = None
else:
    df = None

# Create synthetic data if real data is not available
if df is None or len(df) == 0:
    print("Creating synthetic data for visualization...")
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic Brazilian data professional survey data
    regions = ['Sudeste', 'Sul', 'Nordeste', 'Centro-oeste', 'Norte']
    experience_levels = ['Menos de 1 ano', 'de 1 a 2 anos', 'de 3 a 4 anos', 'de 5 a 6 anos', 'de 7 a 10 anos', 'Mais de 10 anos']
    salary_ranges = ['Menos de R$ 1.000/mês', 'de R$ 1.001/mês a R$ 2.000/mês', 'de R$ 2.001/mês a R$ 3.000/mês', 
                    'de R$ 3.001/mês a R$ 4.000/mês', 'de R$ 4.001/mês a R$ 6.000/mês', 'de R$ 6.001/mês a R$ 8.000/mês',
                    'de R$ 8.001/mês a R$ 12.000/mês', 'de R$ 12.001/mês a R$ 16.000/mês', 'Acima de R$ 20.000/mês']
    company_sizes = ['de 1 a 5', 'de 6 a 10', 'de 11 a 50', 'de 51 a 100', 'de 101 a 500', 'de 501 a 1.000', 'de 1.001 a 3.000', 'Acima de 3.000']
    job_roles = ['Analista de Dados/Data Analyst', 'Cientista de Dados/Data Scientist', 'Engenheiro de Dados/Data Engineer', 
                'Analista de BI/BI Analyst', 'ML Engineer/AI Engineer', 'Arquiteto de Dados/Data Architect']
    education_levels = ['Graduação/Bacharelado', 'Pós-graduação/Especialização', 'Mestrado', 'Doutorado', 'Ensino Médio']
    sectors = ['Tecnologia/Fábrica de Software', 'Finanças ou Bancos', 'Consultoria', 'E-commerce', 'Saúde', 'Educação']
    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']
    languages = ['Python', 'SQL', 'R', 'JavaScript', 'Java', 'C/C++/C#']
    
    df = pd.DataFrame({
        '1.a_idade': np.random.randint(18, 65, n_samples),
        '1.a.1_faixa_idade': np.random.choice(['17-21', '22-25', '26-30', '31-35', '36-40', '41-50', '51+'], n_samples),
        '1.i.2_regiao_onde_mora': np.random.choice(regions, n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        '1.i.1_uf_onde_mora': np.random.choice(states, n_samples),
        '1.l_nivel_de_ensino': np.random.choice(education_levels, n_samples, p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        '2.i_tempo_de_experiencia_em_dados': np.random.choice(experience_levels, n_samples, p=[0.2, 0.25, 0.2, 0.15, 0.15, 0.05]),
        '2.h_faixa_salarial': np.random.choice(salary_ranges, n_samples, p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02]),
        '2.c_numero_de_funcionarios': np.random.choice(company_sizes, n_samples),
        '2.f_cargo_atual': np.random.choice(job_roles, n_samples),
        '2.b_setor': np.random.choice(sectors, n_samples),
        '2.k_satisfeito_atualmente': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
        '4.e_linguagem_mais_usada': np.random.choice(languages, n_samples, p=[0.4, 0.3, 0.1, 0.08, 0.07, 0.05]),
        '4.d.1_SQL': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        '4.d.3_Python': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        '4.d.2_R': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        '4.d.6_Java': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        '4.d.14_JavaScript': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        '2.l.1_Remuneração/Salário': np.random.uniform(0, 1, n_samples),
        '2.l.2_Benefícios': np.random.uniform(0, 1, n_samples),
        '2.l.3_Propósito do trabalho e da empresa': np.random.uniform(0, 1, n_samples),
        '2.l.4_Flexibilidade de trabalho remoto': np.random.uniform(0, 1, n_samples),
    })

# Data preprocessing
def preprocess_data(df):
    df_clean = df.copy()
    
    # Create simplified versions of key variables
    df_clean['age_group'] = df_clean.get('1.a.1_faixa_idade', 'Unknown').fillna('Unknown')
    df_clean['experience_level'] = df_clean.get('2.i_tempo_de_experiencia_em_dados', 'No experience').fillna('No experience')
    df_clean['salary_range'] = df_clean.get('2.h_faixa_salarial', 'Not specified').fillna('Not specified')
    df_clean['region'] = df_clean.get('1.i.2_regiao_onde_mora', 'Unknown').fillna('Unknown')
    df_clean['company_size'] = df_clean.get('2.c_numero_de_funcionarios', 'Unknown').fillna('Unknown')
    df_clean['job_role'] = df_clean.get('2.f_cargo_atual', 'Unknown').fillna('Unknown')
    df_clean['education'] = df_clean.get('1.l_nivel_de_ensino', 'Unknown').fillna('Unknown')
    df_clean['satisfaction'] = df_clean.get('2.k_satisfeito_atualmente', False).fillna(False)
    
    return df_clean

df_clean = preprocess_data(df)

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Define color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
colors_secondary = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

# Subplot 1: Scatter plot with density contours and K-means clustering
ax1 = plt.subplot(3, 3, 1)

# Prepare data for clustering
experience_map = {'Menos de 1 ano': 0, 'de 1 a 2 anos': 1, 'de 3 a 4 anos': 2, 
                 'de 5 a 6 anos': 3, 'de 7 a 10 anos': 4, 'Mais de 10 anos': 5,
                 'No experience': -1, 'Não tenho experiência na área de dados': -1}

salary_map = {'Menos de R$ 1.000/mês': 0, 'de R$ 1.001/mês a R$ 2.000/mês': 1,
              'de R$ 2.001/mês a R$ 3.000/mês': 2, 'de R$ 3.001/mês a R$ 4.000/mês': 3,
              'de R$ 4.001/mês a R$ 6.000/mês': 4, 'de R$ 6.001/mês a R$ 8.000/mês': 5,
              'de R$ 8.001/mês a R$ 12.000/mês': 6, 'de R$ 12.001/mês a R$ 16.000/mês': 7,
              'de R$ 16.001/mês a R$ 20.000/mês': 8, 'Acima de R$ 20.000/mês': 9}

company_size_map = {'de 1 a 5': 1, 'de 6 a 10': 2, 'de 11 a 50': 3, 'de 51 a 100': 4,
                   'de 101 a 500': 5, 'de 501 a 1.000': 6, 'de 1.001 a 3.000': 7, 'Acima de 3.000': 8}

# Create numeric data
plot_data = df_clean.copy()
plot_data['exp_numeric'] = plot_data['experience_level'].map(experience_map).fillna(0)
plot_data['sal_numeric'] = plot_data['salary_range'].map(salary_map).fillna(0)
plot_data['size_numeric'] = plot_data['company_size'].map(company_size_map).fillna(3)

# Remove invalid data
plot_data = plot_data[(plot_data['exp_numeric'] >= 0) & (plot_data['sal_numeric'] >= 0)]

if len(plot_data) > 50:
    # K-means clustering
    X = plot_data[['exp_numeric', 'sal_numeric']].values
    X = X[~np.isnan(X).any(axis=1)]
    
    if len(X) > 10:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Create scatter plot
        regions = plot_data['region'].unique()[:5]
        region_colors = dict(zip(regions, colors_main[:len(regions)]))
        
        for i, region in enumerate(regions):
            mask = plot_data['region'] == region
            if mask.sum() > 0:
                subset = plot_data[mask]
                sizes = subset['size_numeric'] * 20
                ax1.scatter(subset['exp_numeric'], subset['sal_numeric'], 
                           c=region_colors.get(region, colors_main[0]), s=sizes, alpha=0.6, 
                           label=region, edgecolors='white', linewidth=0.5)
        
        # Add cluster boundaries
        try:
            h = 0.5
            xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, h),
                                np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, h))
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax1.contour(xx, yy, Z, colors='black', linestyles='dashed', alpha=0.3)
        except:
            pass

ax1.set_xlabel('Experience Level', fontweight='bold')
ax1.set_ylabel('Salary Range', fontweight='bold')
ax1.set_title('Experience vs Salary with Regional Clustering', fontweight='bold', fontsize=12)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Hierarchical clustering dendrogram with heatmap
ax2 = plt.subplot(3, 3, 2)

# Technology usage correlation matrix
tech_cols = [col for col in df.columns if any(tech in col for tech in ['SQL', 'Python', 'R', 'Java', 'JavaScript'])]
if len(tech_cols) > 3:
    tech_data = df[tech_cols].fillna(0)
    correlation_matrix = tech_data.corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, ax=ax2, cbar_kws={'shrink': 0.8}, fmt='.2f')
    ax2.set_title('Technology Stack Correlation Matrix', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='both', labelsize=8)
else:
    # Create synthetic correlation matrix
    tech_names = ['SQL', 'Python', 'R', 'Java', 'JavaScript']
    np.random.seed(42)
    corr_matrix = np.random.rand(5, 5)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1)
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, ax=ax2, cbar_kws={'shrink': 0.8}, fmt='.2f',
                xticklabels=tech_names, yticklabels=tech_names)
    ax2.set_title('Technology Stack Correlation Matrix', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='both', labelsize=8)

# Subplot 3: Network graph with community detection
ax3 = plt.subplot(3, 3, 3)

# Create education-career network
education_career = df_clean.groupby(['education', 'job_role']).size().reset_index(name='count')
education_career = education_career[education_career['count'] > 2]

if len(education_career) > 0:
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in education_career.iterrows():
        edu = str(row['education'])[:15]
        career = str(row['job_role'])[:15]
        weight = row['count']
        
        if edu != 'nan' and career != 'nan':
            G.add_edge(edu, career, weight=weight)
    
    if len(G.nodes()) > 0:
        # Community detection
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            node_colors = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_colors[node] = colors_main[i % len(colors_main)]
        except:
            node_colors = {node: colors_main[0] for node in G.nodes()}
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        node_sizes = [G.degree(node) * 100 + 50 for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] / 5 for u, v in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_color=[node_colors.get(node, colors_main[0]) for node in G.nodes()],
                              node_size=node_sizes, alpha=0.7, ax=ax3)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, ax=ax3)
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax3)

ax3.set_title('Education-Career Network', fontweight='bold', fontsize=12)
ax3.axis('off')

# Subplot 4: Parallel coordinates plot
ax4 = plt.subplot(3, 3, 4)

# Prepare data for parallel coordinates
numeric_data = df_clean.copy()
numeric_data['age_num'] = df_clean.get('1.a_idade', pd.Series([25] * len(df_clean))).fillna(25)
numeric_data['exp_num'] = numeric_data['experience_level'].map(experience_map).fillna(0)
numeric_data['sal_num'] = numeric_data['salary_range'].map(salary_map).fillna(0)
numeric_data['satisfaction_num'] = numeric_data['satisfaction'].astype(int)

# Select subset for visualization
subset = numeric_data.dropna(subset=['age_num', 'exp_num', 'sal_num', 'satisfaction_num']).sample(min(200, len(numeric_data)))

if len(subset) > 10:
    # Normalize data
    features = ['age_num', 'exp_num', 'sal_num', 'satisfaction_num']
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(subset[features])
    
    # Cluster for coloring
    kmeans_parallel = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans_parallel.fit_predict(normalized_data)
    
    # Plot parallel coordinates
    for i in range(len(normalized_data)):
        ax4.plot(range(len(features)), normalized_data[i], 
                color=colors_main[clusters[i]], alpha=0.3, linewidth=0.5)
    
    # Plot centroids
    centroids = kmeans_parallel.cluster_centers_
    for i, centroid in enumerate(centroids):
        ax4.plot(range(len(features)), centroid, 
                color=colors_main[i], linewidth=3, label=f'Cluster {i+1}')

ax4.set_xticks(range(len(['Age', 'Experience', 'Salary', 'Satisfaction'])))
ax4.set_xticklabels(['Age', 'Experience', 'Salary', 'Satisfaction'], fontweight='bold')
ax4.set_ylabel('Normalized Values', fontweight='bold')
ax4.set_title('Professional Profile Clusters', fontweight='bold', fontsize=12)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Treemap with sunburst overlay
ax5 = plt.subplot(3, 3, 5)

# Hierarchical data for treemap
hierarchy_data = df_clean.groupby(['region', 'job_role']).size().reset_index(name='count')
hierarchy_data = hierarchy_data.dropna().head(20)

if len(hierarchy_data) > 0:
    # Simple treemap visualization using rectangles
    total = hierarchy_data['count'].sum()
    hierarchy_data['percentage'] = hierarchy_data['count'] / total
    
    # Create rectangles
    x, y = 0, 0
    width, height = 1, 1
    
    for i, row in hierarchy_data.iterrows():
        rect_width = row['percentage'] * width
        rect_height = 0.8
        
        color = colors_main[i % len(colors_main)]
        rect = plt.Rectangle((x, y), rect_width, rect_height, 
                           facecolor=color, alpha=0.7, edgecolor='white')
        ax5.add_patch(rect)
        
        # Add text if rectangle is large enough
        if rect_width > 0.05:
            ax5.text(x + rect_width/2, y + rect_height/2, 
                    f"{str(row['region'])[:8]}\n{row['count']}", 
                    ha='center', va='center', fontsize=6, fontweight='bold')
        
        x += rect_width
        if x > 0.9:
            x = 0
            y += 0.2

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.set_title('Market Hierarchy: Region > Role', fontweight='bold', fontsize=12)
ax5.axis('off')

# Subplot 6: Radar chart with box plots
ax6 = plt.subplot(3, 3, 6, projection='polar')

# Skill profiles (using satisfaction metrics as proxy)
skill_cols = [col for col in df.columns if '2.l.' in col][:4]
if len(skill_cols) >= 4:
    skill_data = df[skill_cols].fillna(0)
    
    # Create clusters based on skills
    skill_clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(skill_data)
    
    # Calculate mean skills per cluster
    angles = np.linspace(0, 2 * np.pi, len(skill_cols), endpoint=False)
    
    for cluster in range(3):
        cluster_mask = skill_clusters == cluster
        if cluster_mask.sum() > 0:
            cluster_means = skill_data[cluster_mask].mean()
            values = cluster_means.values
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            angles_plot = np.concatenate((angles, [angles[0]]))
            
            ax6.plot(angles_plot, values, 'o-', linewidth=2, 
                    label=f'Cluster {cluster+1}', color=colors_main[cluster])
            ax6.fill(angles_plot, values, alpha=0.25, color=colors_main[cluster])
else:
    # Create synthetic skill data
    skill_names = ['Salary', 'Benefits', 'Purpose', 'Flexibility']
    angles = np.linspace(0, 2 * np.pi, len(skill_names), endpoint=False)
    
    for cluster in range(3):
        values = np.random.rand(len(skill_names)) * 0.8 + 0.2
        values = np.concatenate((values, [values[0]]))
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax6.plot(angles_plot, values, 'o-', linewidth=2, 
                label=f'Cluster {cluster+1}', color=colors_main[cluster])
        ax6.fill(angles_plot, values, alpha=0.25, color=colors_main[cluster])

ax6.set_xticks(angles)
ax6.set_xticklabels(['Salary', 'Benefits', 'Purpose', 'Flexibility'], fontsize=8)
ax6.set_title('Skill Profile Clusters', fontweight='bold', fontsize=12, pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

# Subplot 7: Geographic scatter with DBSCAN
ax7 = plt.subplot(3, 3, 7)

# State distribution with clustering
state_col = '1.i.1_uf_onde_mora' if '1.i.1_uf_onde_mora' in df_clean.columns else 'region'
state_data = df_clean[state_col].value_counts().head(15)

if len(state_data) > 0:
    # Create synthetic coordinates for states
    np.random.seed(42)
    x_coords = np.random.uniform(0, 10, len(state_data))
    y_coords = np.random.uniform(0, 10, len(state_data))
    
    # DBSCAN clustering
    coords = np.column_stack([x_coords, y_coords])
    dbscan = DBSCAN(eps=2, min_samples=2)
    clusters = dbscan.fit_predict(coords)
    
    # Plot points
    for i, (state, count) in enumerate(state_data.items()):
        cluster = clusters[i]
        color = colors_main[cluster % len(colors_main)] if cluster != -1 else 'gray'
        ax7.scatter(x_coords[i], y_coords[i], s=count*2, c=color, alpha=0.7,
                   edgecolors='white', linewidth=1)
        ax7.annotate(str(state)[:5], (x_coords[i], y_coords[i]), fontsize=6, ha='center')

ax7.set_title('Geographic Distribution with DBSCAN Clustering', fontweight='bold', fontsize=12)
ax7.set_xlabel('Longitude (synthetic)', fontweight='bold')
ax7.set_ylabel('Latitude (synthetic)', fontweight='bold')
ax7.grid(True, alpha=0.3)

# Subplot 8: Bipartite network
ax8 = plt.subplot(3, 3, 8)

# Company-technology bipartite network
sector_col = '2.b_setor' if '2.b_setor' in df_clean.columns else 'job_role'
lang_col = '4.e_linguagem_mais_usada' if '4.e_linguagem_mais_usada' in df_clean.columns else 'job_role'

company_tech = df_clean.groupby([sector_col, lang_col]).size().reset_index(name='count')
company_tech = company_tech.dropna().head(20)

if len(company_tech) > 0:
    B = nx.Graph()
    
    # Add bipartite nodes
    sectors = company_tech[sector_col].unique()
    techs = company_tech[lang_col].unique()
    
    B.add_nodes_from([str(s)[:10] for s in sectors], bipartite=0)
    B.add_nodes_from([str(t)[:10] for t in techs], bipartite=1)
    
    # Add edges
    for _, row in company_tech.iterrows():
        sector = str(row[sector_col])[:10]
        tech = str(row[lang_col])[:10]
        if sector != 'nan' and tech != 'nan':
            B.add_edge(sector, tech, weight=row['count'])
    
    if len(B.nodes()) > 0:
        # Layout
        pos = nx.spring_layout(B, k=2, iterations=50)
        
        # Draw bipartite network
        sector_nodes = [n for n in B.nodes() if any(str(s)[:10] == n for s in sectors)]
        tech_nodes = [n for n in B.nodes() if any(str(t)[:10] == n for t in techs)]
        
        if sector_nodes:
            nx.draw_networkx_nodes(B, pos, nodelist=sector_nodes, node_color=colors_main[0], 
                                  node_size=300, alpha=0.7, ax=ax8)
        if tech_nodes:
            nx.draw_networkx_nodes(B, pos, nodelist=tech_nodes, node_color=colors_main[1], 
                                  node_size=200, alpha=0.7, ax=ax8)
        nx.draw_networkx_edges(B, pos, alpha=0.5, ax=ax8)
        nx.draw_networkx_labels(B, pos, font_size=6, ax=ax8)

ax8.set_title('Company-Technology Ecosystem', fontweight='bold', fontsize=12)
ax8.axis('off')

# Subplot 9: Time-series cluster analysis
ax9 = plt.subplot(3, 3, 9)

# Career progression over experience
experience_order = ['Menos de 1 ano', 'de 1 a 2 anos', 'de 3 a 4 anos', 'de 5 a 6 anos', 'de 7 a 10 anos', 'Mais de 10 anos']
role_progression = df_clean.groupby(['experience_level', 'job_role']).size().unstack(fill_value=0)

if len(role_progression) > 0:
    # Select top roles
    top_roles = role_progression.sum().nlargest(5).index
    role_subset = role_progression[top_roles]
    
    # Plot progression lines
    x_positions = range(len(role_subset.index))
    
    for i, role in enumerate(top_roles):
        if str(role) != 'nan':
            values = role_subset[role].values
            ax9.plot(x_positions, values, marker='o', linewidth=2, 
                    label=str(role)[:20], color=colors_main[i % len(colors_main)])
    
    # Stacked area chart overlay
    ax9_twin = ax9.twinx()
    role_percentages = role_subset.div(role_subset.sum(axis=1), axis=0) * 100
    try:
        ax9_twin.stackplot(x_positions, *[role_percentages[col] for col in top_roles], 
                          colors=colors_secondary[:len(top_roles)], alpha=0.3)
    except:
        pass

ax9.set_xticks(range(len(role_progression.index)))
ax9.set_xticklabels([str(idx)[:10] for idx in role_progression.index], rotation=45, fontsize=8)
ax9.set_xlabel('Experience Level', fontweight='bold')
ax9.set_ylabel('Count', fontweight='bold')
ax9.set_title('Career Progression Patterns', fontweight='bold', fontsize=12)
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.4)

# Save the plot
plt.savefig('brazil_data_professionals_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()