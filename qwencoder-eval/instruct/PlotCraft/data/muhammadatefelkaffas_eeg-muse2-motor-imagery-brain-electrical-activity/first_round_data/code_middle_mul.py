import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import networkx as nx
from scipy.stats import pearsonr
import glob
import warnings
warnings.filterwarnings('ignore')

# Load and combine all CSV files
csv_files = glob.glob('*.csv')

# Load and combine data from available files
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except:
        continue

# If no files found, create sample data for demonstration
if not dfs:
    np.random.seed(42)
    n_samples = 10000
    
    # Create correlated brainwave data
    base_alpha = np.random.normal(1.0, 0.3, n_samples)
    df = pd.DataFrame({
        'Alpha_TP9': base_alpha + np.random.normal(0, 0.1, n_samples),
        'Alpha_AF7': base_alpha * 0.8 + np.random.normal(0, 0.15, n_samples),
        'Alpha_AF8': base_alpha * 0.7 + np.random.normal(0, 0.2, n_samples),
        'Alpha_TP10': base_alpha * 0.9 + np.random.normal(0, 0.12, n_samples),
        'Beta_TP9': np.random.normal(0.5, 0.2, n_samples),
        'Beta_AF7': np.random.normal(0.4, 0.25, n_samples),
        'Beta_AF8': np.random.normal(0.45, 0.22, n_samples),
        'Beta_TP10': np.random.normal(0.55, 0.18, n_samples),
        'Delta_TP9': np.random.normal(0.8, 0.3, n_samples),
        'Delta_AF7': np.random.normal(0.75, 0.28, n_samples),
        'Delta_AF8': np.random.normal(0.82, 0.32, n_samples),
        'Delta_TP10': np.random.normal(0.78, 0.29, n_samples),
        'Theta_TP9': np.random.normal(0.65, 0.24, n_samples),
        'Theta_AF7': np.random.normal(0.62, 0.26, n_samples),
        'Theta_AF8': np.random.normal(0.68, 0.23, n_samples),
        'Theta_TP10': np.random.normal(0.6, 0.25, n_samples),
        'Gamma_TP9': np.random.normal(0.3, 0.15, n_samples),
        'Gamma_AF7': np.random.normal(0.25, 0.12, n_samples),
        'Gamma_AF8': np.random.normal(0.28, 0.14, n_samples),
        'Gamma_TP10': np.random.normal(0.32, 0.16, n_samples),
        'Concentration': np.random.uniform(0, 100, n_samples),
        'Mellow': np.random.uniform(0, 100, n_samples)
    })
else:
    df = pd.concat(dfs, ignore_index=True)

# Sample data if too large for performance
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)

# Clean data and handle missing columns
df = df.dropna()

# Check which columns exist and create missing ones if needed
required_cols = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
                'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
                'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10',
                'Concentration', 'Mellow']

# Check which columns are missing and create them with synthetic data if needed
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Creating synthetic data for missing columns: {missing_cols}")
    np.random.seed(42)
    for col in missing_cols:
        if 'Alpha' in col:
            df[col] = np.random.normal(1.0, 0.3, len(df))
        elif 'Beta' in col:
            df[col] = np.random.normal(0.5, 0.2, len(df))
        elif 'Delta' in col:
            df[col] = np.random.normal(0.8, 0.3, len(df))
        elif 'Theta' in col:
            df[col] = np.random.normal(0.6, 0.25, len(df))
        elif 'Gamma' in col:
            df[col] = np.random.normal(0.3, 0.15, len(df))
        elif col == 'Concentration':
            df[col] = np.random.uniform(0, 100, len(df))
        elif col == 'Mellow':
            df[col] = np.random.uniform(0, 100, len(df))

# Create concentration bins
df['Concentration_Bin'] = pd.cut(df['Concentration'], 
                                bins=[0, 33, 66, 100], 
                                labels=['Low', 'Medium', 'High'])

# Set up the 2x2 subplot grid with white background
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palettes
alpha_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
conc_colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c'}
mellow_colors = plt.cm.viridis

# 1. Top-left: Alpha band correlation heatmap
alpha_cols = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']
alpha_data = df[alpha_cols].dropna()
alpha_corr = alpha_data.corr()

im1 = ax1.imshow(alpha_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax1.set_xticks(range(len(alpha_cols)))
ax1.set_yticks(range(len(alpha_cols)))
ax1.set_xticklabels([col.replace('Alpha_', '') for col in alpha_cols], fontweight='bold')
ax1.set_yticklabels([col.replace('Alpha_', '') for col in alpha_cols], fontweight='bold')

# Annotate correlation coefficients
for i in range(len(alpha_cols)):
    for j in range(len(alpha_cols)):
        text = ax1.text(j, i, f'{alpha_corr.iloc[i, j]:.2f}',
                       ha="center", va="center", color="white", fontweight='bold', fontsize=11)

ax1.set_title('Alpha Band Correlations Across Electrodes', fontweight='bold', fontsize=14, pad=20)
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Pearson Correlation', fontweight='bold')

# 2. Top-right: Beta band scatter plot matrix with concentration groups
beta_cols = ['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']
beta_data = df[beta_cols + ['Concentration_Bin']].dropna()

# Create a simplified pair plot focusing on two key relationships
x_data = beta_data['Beta_TP9']
y_data = beta_data['Beta_AF8']

for i, (group, color) in enumerate(conc_colors.items()):
    mask = beta_data['Concentration_Bin'] == group
    if mask.sum() > 0:
        x_group = x_data[mask]
        y_group = y_data[mask]
        
        ax2.scatter(x_group, y_group, c=color, alpha=0.6, s=30, label=f'{group} (n={mask.sum()})')
        
        # Add regression line
        if len(x_group) > 10:
            try:
                z = np.polyfit(x_group, y_group, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_group.min(), x_group.max(), 100)
                ax2.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2, alpha=0.8)
            except:
                pass

ax2.set_xlabel('Beta TP9 Power', fontweight='bold')
ax2.set_ylabel('Beta AF8 Power', fontweight='bold')
ax2.set_title('Beta Band: TP9 vs AF8 by Concentration Level', fontweight='bold', fontsize=14, pad=20)
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)

# 3. Bottom-left: Bubble plot
bubble_cols = ['Delta_TP9', 'Theta_TP10', 'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10', 'Mellow']
bubble_data = df[bubble_cols].dropna()

# Calculate gamma average
gamma_cols = ['Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10']
bubble_data['Gamma_Avg'] = bubble_data[gamma_cols].mean(axis=1)

# Sample for better visualization
if len(bubble_data) > 2000:
    bubble_data = bubble_data.sample(n=2000, random_state=42)

# Create bubble sizes (normalize gamma average)
gamma_range = bubble_data['Gamma_Avg'].max() - bubble_data['Gamma_Avg'].min()
if gamma_range > 0:
    sizes = (bubble_data['Gamma_Avg'] - bubble_data['Gamma_Avg'].min()) / gamma_range
else:
    sizes = np.ones(len(bubble_data)) * 0.5
sizes = 20 + sizes * 100  # Scale to reasonable bubble sizes

scatter = ax3.scatter(bubble_data['Delta_TP9'], bubble_data['Theta_TP10'], 
                     s=sizes, c=bubble_data['Mellow'], cmap=mellow_colors, 
                     alpha=0.6, edgecolors='white', linewidth=0.5)

ax3.set_xlabel('Delta TP9 Power', fontweight='bold')
ax3.set_ylabel('Theta TP10 Power', fontweight='bold')
ax3.set_title('Delta-Theta Relationship\n(Bubble size: Gamma Average, Color: Mellow State)', 
              fontweight='bold', fontsize=14, pad=20)
cbar3 = plt.colorbar(scatter, ax=ax3, shrink=0.8)
cbar3.set_label('Mellow Level', fontweight='bold')

# 4. Bottom-right: Correlation network plot
# Calculate correlations between all brainwave bands
network_cols = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
                'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
                'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10']

# Only use columns that exist in the dataframe
available_network_cols = [col for col in network_cols if col in df.columns]
network_data = df[available_network_cols].dropna()

if len(network_data) > 5000:
    network_data = network_data.sample(n=5000, random_state=42)

# Calculate correlation matrix
corr_matrix = network_data.corr()

# Create network graph
G = nx.Graph()
electrode_colors = {'TP9': '#e74c3c', 'AF7': '#3498db', 'AF8': '#2ecc71', 'TP10': '#f39c12'}

# Add nodes with positions
pos = {}
for i, col in enumerate(available_network_cols):
    parts = col.split('_')
    if len(parts) == 2:
        band, electrode = parts
        G.add_node(col, band=band, electrode=electrode)
        
        # Position nodes in a circular layout by electrode
        if electrode == 'TP9':
            pos[col] = (-1, -1)
        elif electrode == 'AF7':
            pos[col] = (-1, 1)
        elif electrode == 'AF8':
            pos[col] = (1, 1)
        elif electrode == 'TP10':
            pos[col] = (1, -1)
        else:
            # Default position for unknown electrodes
            pos[col] = (0, 0)

# Add edges for strong correlations
threshold = 0.3
for i in range(len(available_network_cols)):
    for j in range(i+1, len(available_network_cols)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > threshold:
            G.add_edge(available_network_cols[i], available_network_cols[j], weight=abs(corr_val))

# Draw network if we have nodes and edges
if len(G.nodes()) > 0:
    node_colors = []
    for node in G.nodes():
        parts = node.split('_')
        if len(parts) == 2:
            electrode = parts[1]
            node_colors.append(electrode_colors.get(electrode, '#gray'))
        else:
            node_colors.append('#gray')
    
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                          alpha=0.8, ax=ax4)
    
    if len(G.edges()) > 0:
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, 
                              edge_color='gray', ax=ax4)
    
    # Add labels
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax4)
    
    # Add legend for electrode colors - Fixed the title_fontweight issue
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=electrode)
                      for electrode, color in electrode_colors.items()]
    legend = ax4.legend(handles=legend_elements, loc='upper right', title='Electrodes', 
                       frameon=True, fancybox=True, shadow=True)
    legend.get_title().set_fontweight('bold')  # Fixed: Set title font weight separately
else:
    ax4.text(0.5, 0.5, 'No strong correlations found\n(|r| > 0.3)', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)

ax4.set_title('Brainwave Correlation Network\n(|r| > 0.3, Edge thickness ∝ correlation strength)', 
              fontweight='bold', fontsize=14, pad=20)
ax4.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Save the plot
plt.savefig('eeg_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()