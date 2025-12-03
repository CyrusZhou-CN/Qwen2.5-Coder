import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
accounts = pd.read_csv('accounts.csv')
transactions = pd.read_csv('transactions.csv')
alerts = pd.read_csv('alerts.csv')

# Data preprocessing
# Sample data for performance (using representative samples)
np.random.seed(42)
accounts_sample = accounts.sample(n=min(1000, len(accounts)))
transactions_sample = transactions.sample(n=min(50000, len(transactions)))
alerts_sample = alerts.sample(n=min(500, len(alerts)))

# Merge datasets for analysis
tx_with_accounts = transactions_sample.merge(
    accounts_sample[['ACCOUNT_ID', 'IS_FRAUD']], 
    left_on='SENDER_ACCOUNT_ID', 
    right_on='ACCOUNT_ID', 
    how='inner'
).rename(columns={'IS_FRAUD': 'SENDER_FRAUD'})

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Row 1: Account Analysis

# Subplot 1: Scatter plot with box plots overlay
ax1 = plt.subplot(3, 3, 1)
fraud_accounts = accounts_sample[accounts_sample['IS_FRAUD'] == True]
normal_accounts = accounts_sample[accounts_sample['IS_FRAUD'] == False]

# Scatter plot
ax1.scatter(normal_accounts.index, normal_accounts['INIT_BALANCE'], 
           alpha=0.6, c='lightblue', s=30, label='Normal')
ax1.scatter(fraud_accounts.index, fraud_accounts['INIT_BALANCE'], 
           alpha=0.8, c='red', s=30, label='Fraud')

# Add box plot overlay - Fixed: removed alpha parameter
box_data = [normal_accounts['INIT_BALANCE'].dropna(), fraud_accounts['INIT_BALANCE'].dropna()]
box_positions = [len(accounts_sample)*0.2, len(accounts_sample)*0.8]
bp = ax1.boxplot(box_data, positions=box_positions, widths=len(accounts_sample)*0.1, 
                patch_artist=True)
# Set transparency manually after creation
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('red')
bp['boxes'][1].set_alpha(0.7)

ax1.set_title('Account Balance vs Fraud Status with Distribution', fontweight='bold', fontsize=12)
ax1.set_xlabel('Account Index')
ax1.set_ylabel('Initial Balance')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Network graph
ax2 = plt.subplot(3, 3, 2)
# Create network from transaction sample
G = nx.Graph()
tx_subset = transactions_sample.head(200)  # Limit for visualization

for _, row in tx_subset.iterrows():
    sender = row['SENDER_ACCOUNT_ID']
    receiver = row['RECEIVER_ACCOUNT_ID']
    amount = row['TX_AMOUNT']
    
    if G.has_edge(sender, receiver):
        G[sender][receiver]['weight'] += amount
    else:
        G.add_edge(sender, receiver, weight=amount)

# Add node attributes
for node in G.nodes():
    account_info = accounts_sample[accounts_sample['ACCOUNT_ID'] == node]
    if not account_info.empty:
        G.nodes[node]['fraud'] = account_info.iloc[0]['IS_FRAUD']
        G.nodes[node]['balance'] = account_info.iloc[0]['INIT_BALANCE']
    else:
        G.nodes[node]['fraud'] = False
        G.nodes[node]['balance'] = 100

# Layout and visualization
if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1, iterations=50)
    node_colors = ['red' if G.nodes[node].get('fraud', False) else 'lightblue' for node in G.nodes()]
    node_sizes = [max(G.nodes[node].get('balance', 100)/5, 10) for node in G.nodes()]

    nx.draw(G, pos, ax=ax2, node_color=node_colors, node_size=node_sizes, 
            alpha=0.7, edge_color='gray', width=0.5)

ax2.set_title('Account Transaction Network', fontweight='bold', fontsize=12)

# Subplot 3: Histogram with KDE overlay
ax3 = plt.subplot(3, 3, 3)
normal_balances = normal_accounts['INIT_BALANCE'].dropna()
fraud_balances = fraud_accounts['INIT_BALANCE'].dropna()

# Histogram
ax3.hist(normal_balances, bins=30, alpha=0.6, color='lightblue', 
         label='Normal', density=True)
ax3.hist(fraud_balances, bins=30, alpha=0.6, color='red', 
         label='Fraud', density=True)

# KDE overlay
if len(normal_balances) > 1:
    x_normal = np.linspace(normal_balances.min(), normal_balances.max(), 100)
    kde_normal = stats.gaussian_kde(normal_balances)
    ax3.plot(x_normal, kde_normal(x_normal), color='blue', linewidth=2, label='Normal KDE')

if len(fraud_balances) > 1:
    x_fraud = np.linspace(fraud_balances.min(), fraud_balances.max(), 100)
    kde_fraud = stats.gaussian_kde(fraud_balances)
    ax3.plot(x_fraud, kde_fraud(x_fraud), color='darkred', linewidth=2, label='Fraud KDE')

ax3.set_title('Balance Distribution with KDE', fontweight='bold', fontsize=12)
ax3.set_xlabel('Initial Balance')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Row 2: Transaction Flow Analysis

# Subplot 4: Bubble chart with heatmap
ax4 = plt.subplot(3, 3, 4)
tx_subset = transactions_sample.head(1000)

if len(tx_subset) > 0:
    # Create heatmap data
    sender_ids = tx_subset['SENDER_ACCOUNT_ID'].values
    receiver_ids = tx_subset['RECEIVER_ACCOUNT_ID'].values
    amounts = tx_subset['TX_AMOUNT'].values

    # Normalize for visualization
    if len(sender_ids) > 0 and sender_ids.max() != sender_ids.min():
        sender_norm = (sender_ids - sender_ids.min()) / (sender_ids.max() - sender_ids.min()) * 100
        receiver_norm = (receiver_ids - receiver_ids.min()) / (receiver_ids.max() - receiver_ids.min()) * 100

        # Create 2D histogram for heatmap
        hist, xedges, yedges = np.histogram2d(sender_norm, receiver_norm, bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Plot heatmap
        im = ax4.imshow(hist.T, extent=extent, origin='lower', alpha=0.6, cmap='Blues')

        # Overlay bubble chart
        bubble_sizes = (amounts / amounts.max()) * 100
        ax4.scatter(sender_norm, receiver_norm, s=bubble_sizes, alpha=0.7, c='red', edgecolors='black')

ax4.set_title('Transaction Flow: Bubble Chart with Density Heatmap', fontweight='bold', fontsize=12)
ax4.set_xlabel('Sender Account (Normalized)')
ax4.set_ylabel('Receiver Account (Normalized)')

# Subplot 5: Parallel coordinates plot
ax5 = plt.subplot(3, 3, 5)
# Prepare data for parallel coordinates
pc_data = tx_subset[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'IS_FRAUD']].head(100)

if len(pc_data) > 0:
    # Normalize data for parallel coordinates
    pc_normalized = pc_data.copy()
    for col in ['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT']:
        col_min, col_max = pc_data[col].min(), pc_data[col].max()
        if col_max != col_min:
            pc_normalized[col] = (pc_data[col] - col_min) / (col_max - col_min)
        else:
            pc_normalized[col] = 0.5

    # Plot parallel coordinates
    x_pos = np.arange(3)
    for i, row in pc_normalized.iterrows():
        color = 'red' if row['IS_FRAUD'] else 'lightblue'
        alpha = 0.8 if row['IS_FRAUD'] else 0.3
        ax5.plot(x_pos, [row['SENDER_ACCOUNT_ID'], row['RECEIVER_ACCOUNT_ID'], row['TX_AMOUNT']], 
                 color=color, alpha=alpha, linewidth=1)

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['Sender', 'Receiver', 'Amount'])

ax5.set_title('Parallel Coordinates: Transaction Patterns', fontweight='bold', fontsize=12)
ax5.set_ylabel('Normalized Values')
ax5.grid(True, alpha=0.3)

# Subplot 6: Violin plots with strip plots
ax6 = plt.subplot(3, 3, 6)
alert_types = alerts_sample['ALERT_TYPE'].unique()
alert_amounts = []
alert_labels = []

for alert_type in alert_types:
    amounts = alerts_sample[alerts_sample['ALERT_TYPE'] == alert_type]['TX_AMOUNT']
    if len(amounts) > 0:
        alert_amounts.append(amounts)
        alert_labels.append(alert_type)

if alert_amounts:
    # Violin plot
    parts = ax6.violinplot(alert_amounts, positions=range(len(alert_amounts)), 
                          showmeans=True, showmedians=True)
    
    # Strip plot overlay
    for i, amounts in enumerate(alert_amounts):
        y_jitter = np.random.normal(i, 0.04, len(amounts))
        ax6.scatter(y_jitter, amounts, alpha=0.6, s=20)

    ax6.set_xticks(range(len(alert_labels)))
    ax6.set_xticklabels(alert_labels, rotation=45)

ax6.set_title('Transaction Amounts by Alert Type', fontweight='bold', fontsize=12)
ax6.set_ylabel('Transaction Amount')
ax6.grid(True, alpha=0.3)

# Row 3: Alert Pattern Investigation

# Subplot 7: Treemap with embedded bar charts
ax7 = plt.subplot(3, 3, 7)
alert_counts = alerts_sample['ALERT_TYPE'].value_counts()
total_alerts = len(alerts_sample)

if len(alert_counts) > 0:
    # Simple treemap representation using rectangles
    colors = plt.cm.Set3(np.linspace(0, 1, len(alert_counts)))
    y_pos = 0
    for i, (alert_type, count) in enumerate(alert_counts.items()):
        height = count / total_alerts
        rect = Rectangle((0, y_pos), 1, height, facecolor=colors[i], 
                        edgecolor='black', alpha=0.7)
        ax7.add_patch(rect)
        
        # Add text
        ax7.text(0.5, y_pos + height/2, f'{alert_type}\n{count}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        y_pos += height

ax7.set_xlim(0, 1)
ax7.set_ylim(0, 1)
ax7.set_title('Alert Type Distribution (Treemap)', fontweight='bold', fontsize=12)
ax7.set_xticks([])
ax7.set_yticks([])

# Subplot 8: Cluster analysis
ax8 = plt.subplot(3, 3, 8)
cluster_data = alerts_sample[['TX_AMOUNT', 'TIMESTAMP']].dropna()

if len(cluster_data) > 0:
    # Normalize data
    amount_min, amount_max = cluster_data['TX_AMOUNT'].min(), cluster_data['TX_AMOUNT'].max()
    time_min, time_max = cluster_data['TIMESTAMP'].min(), cluster_data['TIMESTAMP'].max()
    
    if amount_max != amount_min:
        amounts_norm = (cluster_data['TX_AMOUNT'] - amount_min) / (amount_max - amount_min)
    else:
        amounts_norm = np.full(len(cluster_data), 0.5)
        
    if time_max != time_min:
        timestamps_norm = (cluster_data['TIMESTAMP'] - time_min) / (time_max - time_min)
    else:
        timestamps_norm = np.full(len(cluster_data), 0.5)
    
    # Color by alert type
    alert_types_unique = alerts_sample['ALERT_TYPE'].unique()
    color_map = {alert_type: plt.cm.Set1(i/len(alert_types_unique)) 
                for i, alert_type in enumerate(alert_types_unique)}
    
    colors = [color_map.get(alert_type, 'gray') for alert_type in alerts_sample['ALERT_TYPE']]
    markers = ['o' if fraud else '^' for fraud in alerts_sample['IS_FRAUD']]
    
    for i, (x, y) in enumerate(zip(timestamps_norm, amounts_norm)):
        if i < len(colors) and i < len(markers) and i < len(alerts_sample):
            ax8.scatter(x, y, c=[colors[i]], marker=markers[i], s=50, alpha=0.7)

ax8.set_title('Cluster Analysis: Amount vs Time', fontweight='bold', fontsize=12)
ax8.set_xlabel('Normalized Timestamp')
ax8.set_ylabel('Normalized Amount')
ax8.grid(True, alpha=0.3)

# Subplot 9: Correlation heatmap with dendrogram
ax9 = plt.subplot(3, 3, 9)

# Prepare numerical data for correlation
numerical_cols = ['INIT_BALANCE', 'TX_BEHAVIOR_ID']
if len(accounts_sample) > 0:
    corr_data = accounts_sample[numerical_cols].corr()
    
    # Create heatmap
    im = ax9.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(len(corr_data)):
        for j in range(len(corr_data.columns)):
            text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax9.set_xticks(range(len(corr_data.columns)))
    ax9.set_yticks(range(len(corr_data)))
    ax9.set_xticklabels(corr_data.columns, rotation=45)
    ax9.set_yticklabels(corr_data.index)
    
    # Add colorbar
    plt.colorbar(im, ax=ax9, shrink=0.8)

ax9.set_title('Correlation Heatmap', fontweight='bold', fontsize=12)

# Overall layout adjustments
plt.suptitle('Comprehensive AML Analysis: Money Laundering Patterns and Account Clustering', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('aml_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()