import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# List of all CSV files from the data description
csv_files = [
    'museMonitor_2024-06-05--17-33-40_3002428320981162812.csv',
    'museMonitor_2024-06-22--18-41-35_2861114036213037750.csv',
    'museMonitor_2024-06-22--18-29-43_644927611320945296.csv',
    'museMonitor_2024-06-22--00-04-51_5653597502572991858.csv',
    'museMonitor_2024-06-06--12-56-03_1872932237480975429.csv',
    'museMonitor_2024-06-05--17-23-37_9208698270315378717.csv',
    'museMonitor_2024-06-19--00-37-24_4635291951372140676.csv',
    'museMonitor_2024-06-21--21-59-52_6156926087727383308.csv',
    'museMonitor_2024-06-23--03-40-25_4163822837414138498.csv',
    'museMonitor_2024-06-12--14-22-03_4954767773620636657.csv',
    'museMonitor_2024-06-20--22-56-06_1640398272947211195.csv',
    'museMonitor_2024-06-02--09-47-17_5812437961079996628.csv',
    'museMonitor_2024-06-19--10-40-52_4943173446412590144.csv',
    'museMonitor_2024-06-13--14-10-20_4505350993163227406.csv',
    'museMonitor_2024-06-22--00-32-25_561002230933821771.csv',
    'museMonitor_2024-06-06--13-58-43_4711803696841563144.csv',
    'museMonitor_2024-06-22--18-57-07_8732425103180053829.csv',
    'museMonitor_2024-06-21--22-22-27_335105096489991602.csv',
    'museMonitor_2024-06-21--00-02-00_3021179978499686494.csv',
    'museMonitor_2024-06-22--18-50-07_1277522147074047198.csv',
    'museMonitor_2024-06-12--13-38-16_3222374500720036221.csv',
    'museMonitor_2024-06-13--14-48-13_779422176232281909.csv',
    'museMonitor_2024-06-21--18-29-25_6241035167039508312.csv',
    'museMonitor_2024-06-21--23-10-14_7413883026386137496.csv',
    'museMonitor_2024-06-17--21-42-37_7129358710896752933.csv',
    'museMonitor_2024-06-21--20-00-24_6554922359716231338.csv',
    'museMonitor_2024-06-20--16-28-41_7279045766420892505.csv',
    'museMonitor_2024-06-03--05-38-00_6365449446820111413.csv',
    'museMonitor_2024-06-21--01-47-40_2765725268234193889.csv',
    'museMonitor_2024-06-18--01-31-42_5718445944978450870.csv'
]

# Load and combine datasets
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"Successfully loaded {file}: {df.shape}")
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {file}")
        continue
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

# Check if we have any data
if not dfs:
    print("No CSV files found. Using sample data for demonstration.")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample brainwave data
    sample_data = {}
    electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    for band in bands:
        for electrode in electrodes:
            sample_data[f'{band}_{electrode}'] = np.random.normal(0, 1, n_samples)
    
    sample_data['Mellow'] = np.random.uniform(0, 100, n_samples)
    sample_data['Concentration'] = np.random.uniform(0, 100, n_samples)
    
    df = pd.DataFrame(sample_data)
else:
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

# Remove rows with missing values in key columns
brainwave_cols = [col for col in df.columns if any(band in col for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']) and any(pos in col for pos in ['TP9', 'AF7', 'AF8', 'TP10'])]
cognitive_cols = ['Mellow', 'Concentration']
key_cols = brainwave_cols + cognitive_cols

# Filter columns that exist in the dataframe
existing_cols = [col for col in key_cols if col in df.columns]
df_clean = df[existing_cols].dropna()

# Sample data for performance (use every 50th row for large datasets)
sample_rate = max(1, len(df_clean) // 5000)  # Limit to ~5000 samples
df_sample = df_clean.iloc[::sample_rate].copy()

print(f"Working with {len(df_sample)} samples after cleaning and sampling")

# Create the 2x2 subplot layout
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('white')

# Define electrode positions and brainwave bands
electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Create correlation matrix for all brainwave features
brainwave_features = []
for band in bands:
    for electrode in electrodes:
        col_name = f'{band}_{electrode}'
        if col_name in df_sample.columns:
            brainwave_features.append(col_name)

if len(brainwave_features) == 0:
    print("No brainwave features found. Check column names.")
    print("Available columns:", df_sample.columns.tolist())

correlation_matrix = df_sample[brainwave_features].corr()

# Subplot 1: Correlation heatmap of all brainwave bands across electrodes
ax1 = plt.subplot(2, 2, 1)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('Brainwave Correlation Heatmap\nAcross All Bands and Electrodes', 
              fontweight='bold', fontsize=12, pad=15)
ax1.set_xlabel('Brainwave Features', fontweight='bold')
ax1.set_ylabel('Brainwave Features', fontweight='bold')
ax1.tick_params(axis='both', labelsize=8)

# Subplot 2: Scatter plot matrix for Alpha and Beta bands with concentration coloring
ax2 = plt.subplot(2, 2, 2)

# Get Alpha and Beta features
alpha_features = [col for col in brainwave_features if 'Alpha' in col]
beta_features = [col for col in brainwave_features if 'Beta' in col]

# Create scatter plot comparing Alpha vs Beta for each electrode
if 'Concentration' in df_sample.columns and len(alpha_features) > 0 and len(beta_features) > 0:
    # Use first available Alpha and Beta features for demonstration
    alpha_col = alpha_features[0]
    beta_col = beta_features[0]
    
    scatter = ax2.scatter(df_sample[alpha_col], df_sample[beta_col], 
                         c=df_sample['Concentration'], cmap='viridis', 
                         alpha=0.6, s=20)
    
    ax2.set_title('Alpha vs Beta Bands\nColored by Concentration Level', 
                  fontweight='bold', fontsize=12, pad=15)
    ax2.set_xlabel(f'{alpha_col} Activity', fontweight='bold')
    ax2.set_ylabel(f'{beta_col} Activity', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for concentration
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Concentration Level', fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Alpha/Beta data or\nConcentration not available', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Alpha vs Beta Analysis', fontweight='bold', fontsize=12, pad=15)

# Subplot 3: Bubble plot of Mellow vs Concentration with Alpha activity as bubble size
ax3 = plt.subplot(2, 2, 3)

if 'Mellow' in df_sample.columns and 'Concentration' in df_sample.columns and len(alpha_features) > 0:
    # Calculate average Alpha activity across all electrodes
    df_sample['Alpha_avg'] = df_sample[alpha_features].mean(axis=1)
    
    # Normalize bubble sizes
    alpha_range = df_sample['Alpha_avg'].max() - df_sample['Alpha_avg'].min()
    if alpha_range > 0:
        bubble_sizes = (df_sample['Alpha_avg'] - df_sample['Alpha_avg'].min()) / alpha_range
        bubble_sizes = 20 + bubble_sizes * 100  # Scale to reasonable size range
    else:
        bubble_sizes = np.full(len(df_sample), 50)
    
    scatter3 = ax3.scatter(df_sample['Mellow'], df_sample['Concentration'], 
                          s=bubble_sizes, alpha=0.6, c=df_sample['Alpha_avg'], 
                          cmap='plasma', edgecolors='black', linewidth=0.5)
    
    ax3.set_title('Mellow vs Concentration\nBubble Size = Average Alpha Activity', 
                  fontweight='bold', fontsize=12, pad=15)
    ax3.set_xlabel('Mellow Score', fontweight='bold')
    ax3.set_ylabel('Concentration Score', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for Alpha activity
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('Average Alpha Activity', fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'Mellow/Concentration data\nor Alpha features not available', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Mellow vs Concentration Analysis', fontweight='bold', fontsize=12, pad=15)

# Subplot 4: Network-style correlation plot for strong correlations
ax4 = plt.subplot(2, 2, 4)

if len(brainwave_features) > 1:
    # Create network graph for strong correlations (|r| > 0.5)
    G = nx.Graph()
    
    # Add nodes for each brainwave feature
    for feature in brainwave_features:
        G.add_node(feature)
    
    # Add edges for strong correlations
    strong_correlations = []
    for i, feature1 in enumerate(brainwave_features):
        for j, feature2 in enumerate(brainwave_features):
            if i < j:  # Avoid duplicates
                corr_val = correlation_matrix.loc[feature1, feature2]
                if abs(corr_val) > 0.5:
                    G.add_edge(feature1, feature2, weight=abs(corr_val), correlation=corr_val)
                    strong_correlations.append((feature1, feature2, corr_val))
    
    if len(G.nodes()) > 0:
        # Create layout for network
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if 'Alpha' in node:
                node_colors.append('#FF6B6B')
            elif 'Beta' in node:
                node_colors.append('#4ECDC4')
            elif 'Theta' in node:
                node_colors.append('#45B7D1')
            elif 'Delta' in node:
                node_colors.append('#96CEB4')
            else:  # Gamma
                node_colors.append('#FFEAA7')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax4)
        
        # Draw edges with thickness based on correlation strength
        edges = G.edges(data=True)
        for edge in edges:
            weight = edge[2]['weight']
            correlation = edge[2]['correlation']
            color = 'red' if correlation > 0 else 'blue'
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], 
                                  width=weight*3, alpha=0.7, edge_color=color, ax=ax4)
        
        # Add labels for nodes
        labels = {node: node.replace('_', '\n') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax4)
        
        # Add legend for node colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=10, label='Alpha'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=10, label='Beta'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', markersize=10, label='Theta'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#96CEB4', markersize=10, label='Delta'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFEAA7', markersize=10, label='Gamma')
        ]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No strong correlations\n(|r| > 0.5) found', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    ax4.set_title('Strong Correlation Network\n(|r| > 0.5, Red=Positive, Blue=Negative)', 
                  fontweight='bold', fontsize=12, pad=15)
    ax4.axis('off')
else:
    ax4.text(0.5, 0.5, 'Insufficient brainwave\nfeatures for network analysis', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Correlation Network Analysis', fontweight='bold', fontsize=12, pad=15)
    ax4.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)

# Add overall title
fig.suptitle('EEG Brainwave Correlation Analysis in Motor Imagery Tasks', 
             fontsize=16, fontweight='bold', y=0.98)

# Print summary statistics
print("\nDataset Summary:")
print(f"Total samples analyzed: {len(df_sample):,}")
if len(brainwave_features) > 1:
    strong_correlations_count = len([1 for i in range(len(brainwave_features)) 
                                   for j in range(i+1, len(brainwave_features)) 
                                   if abs(correlation_matrix.iloc[i,j]) > 0.5])
    print(f"Number of strong correlations (|r| > 0.5): {strong_correlations_count}")

if 'Alpha_avg' in df_sample.columns:
    print(f"Average Alpha activity: {df_sample['Alpha_avg'].mean():.3f} ± {df_sample['Alpha_avg'].std():.3f}")
if 'Concentration' in df_sample.columns:
    print(f"Average Concentration: {df_sample['Concentration'].mean():.2f} ± {df_sample['Concentration'].std():.2f}")
if 'Mellow' in df_sample.columns:
    print(f"Average Mellow score: {df_sample['Mellow'].mean():.2f} ± {df_sample['Mellow'].std():.2f}")

plt.savefig('eeg_brainwave_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()