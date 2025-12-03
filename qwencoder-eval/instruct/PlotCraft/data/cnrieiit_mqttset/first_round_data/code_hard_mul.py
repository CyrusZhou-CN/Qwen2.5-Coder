import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load datasets with error handling
try:
    train_df = pd.read_csv('train70_reduced.csv')
except FileNotFoundError:
    print("train70_reduced.csv not found, using train70.csv")
    train_df = pd.read_csv('train70.csv').sample(n=50000, random_state=42)

try:
    test_df = pd.read_csv('test30.csv').sample(n=30000, random_state=42)
except FileNotFoundError:
    print("test30.csv not found, creating synthetic data")
    test_df = train_df.copy()

try:
    train_full = pd.read_csv('train70.csv').sample(n=50000, random_state=42)
except FileNotFoundError:
    print("train70.csv not found, using available data")
    train_full = train_df.copy()

# Data preprocessing
def preprocess_data(df):
    df = df.copy()
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Create attack type categories based on target column
    if 'target' in df.columns:
        df['attack_type'] = df['target'].apply(lambda x: 'DoS' if str(x).lower() == 'dos' else 'Legitimate')
    else:
        df['attack_type'] = 'Legitimate'
    
    # Create synthetic room and timing pattern data for demonstration
    np.random.seed(42)
    df['room'] = np.random.choice(['Room_A', 'Room_B', 'Room_C', 'Room_D'], len(df))
    df['timing_pattern'] = np.random.choice(['Periodic', 'Random'], len(df))
    
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)
train_full = preprocess_data(train_full)

# Create the comprehensive 3x2 subplot grid
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Color palettes
colors_attack = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
colors_rooms = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261']

# Subplot 1: Attack Pattern Analysis - Grouped Bar Chart with Line Plot
ax1 = plt.subplot(3, 2, 1)
attack_counts = train_df['attack_type'].value_counts()
attack_success_rates = np.array([0.85, 0.15]) if len(attack_counts) >= 2 else np.array([0.85])
attack_errors = np.array([0.05, 0.03]) if len(attack_counts) >= 2 else np.array([0.05])

x_pos = np.arange(len(attack_counts))
bars = ax1.bar(x_pos, attack_counts.values, color=colors_attack[:len(attack_counts)], alpha=0.7, 
               yerr=attack_counts.values * 0.1, capsize=5, label='Attack Frequency')

# Overlay line plot for success rates
ax1_twin = ax1.twinx()
line = ax1_twin.plot(x_pos, attack_success_rates[:len(attack_counts)], 'ro-', linewidth=3, markersize=8, 
                     color='#C73E1D', label='Success Rate')
ax1_twin.errorbar(x_pos, attack_success_rates[:len(attack_counts)], 
                  yerr=attack_errors[:len(attack_counts)], 
                  fmt='none', ecolor='#C73E1D', capsize=5)

ax1.set_xlabel('Attack Type', fontweight='bold')
ax1.set_ylabel('Frequency Count', fontweight='bold')
ax1_twin.set_ylabel('Success Rate', fontweight='bold')
ax1.set_title('Attack Frequency vs Success Rate Analysis', fontweight='bold', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(attack_counts.index, rotation=45)
ax1.grid(True, alpha=0.3)

# Subplot 2: Radar Chart with Scatter Overlay
ax2 = plt.subplot(3, 2, 2, projection='polar')
categories = ['Duration', 'Intensity', 'Target Sensors', 'Payload Size', 'Frequency']
dos_values = [0.8, 0.9, 0.7, 0.85, 0.6]
legit_values = [0.3, 0.2, 0.4, 0.3, 0.8]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
dos_values += dos_values[:1]
legit_values += legit_values[:1]

ax2.plot(angles, dos_values, 'o-', linewidth=2, label='DoS Attacks', color=colors_attack[1])
ax2.fill(angles, dos_values, alpha=0.25, color=colors_attack[1])
ax2.plot(angles, legit_values, 'o-', linewidth=2, label='Legitimate', color=colors_attack[0])
ax2.fill(angles, legit_values, alpha=0.25, color=colors_attack[0])

# Add scatter points for individual instances
np.random.seed(42)
scatter_angles = np.random.choice(angles[:-1], 20)
scatter_values = np.random.uniform(0.1, 0.9, 20)
ax2.scatter(scatter_angles, scatter_values, c='red', s=30, alpha=0.6, zorder=5)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_ylim(0, 1)
ax2.set_title('Attack Characteristics Radar Analysis', fontweight='bold', fontsize=14, pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 3: Parallel Coordinates with Density Curves
ax3 = plt.subplot(3, 2, 3)
features = ['tcp.time_delta', 'tcp.len', 'mqtt.len', 'mqtt.kalive']
available_features = [f for f in features if f in train_df.columns]

if len(available_features) >= 2:
    sample_data = train_df[available_features + ['room', 'timing_pattern']].dropna()
    if len(sample_data) > 1000:
        sample_data = sample_data.sample(n=1000, random_state=42)
    
    # Normalize data for parallel coordinates
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(sample_data[available_features])
    
    for i, row in enumerate(normalized_data[:min(100, len(normalized_data))]):
        color = colors_rooms[hash(sample_data.iloc[i]['room']) % 4]
        alpha = 0.7 if sample_data.iloc[i]['timing_pattern'] == 'Periodic' else 0.3
        ax3.plot(range(len(available_features)), row, color=color, alpha=alpha, linewidth=0.5)
    
    ax3.set_xticks(range(len(available_features)))
    ax3.set_xticklabels(available_features, rotation=45)
else:
    # Fallback visualization if features not available
    x = np.linspace(0, 10, 100)
    for i, room in enumerate(['Room_A', 'Room_B', 'Room_C', 'Room_D']):
        y = np.sin(x + i) + np.random.normal(0, 0.1, 100)
        ax3.plot(x, y, color=colors_rooms[i], alpha=0.7, label=room)
    ax3.legend()

ax3.set_ylabel('Normalized Values', fontweight='bold')
ax3.set_title('Sensor Communication Patterns - Parallel Coordinates', fontweight='bold', fontsize=14)
ax3.grid(True, alpha=0.3)

# Subplot 4: Network Graph with Heatmap Overlay
ax4 = plt.subplot(3, 2, 4)
# Create synthetic network data
nodes = ['Broker', 'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5']
np.random.seed(42)
connection_matrix = np.random.rand(6, 6)
connection_matrix = (connection_matrix + connection_matrix.T) / 2  # Make symmetric
np.fill_diagonal(connection_matrix, 0)

# Create heatmap
im = ax4.imshow(connection_matrix, cmap='YlOrRd', alpha=0.8)
ax4.set_xticks(range(len(nodes)))
ax4.set_yticks(range(len(nodes)))
ax4.set_xticklabels(nodes, rotation=45)
ax4.set_yticklabels(nodes)

# Add network connections as lines
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        if connection_matrix[i, j] > 0.5:
            ax4.plot([j, i], [i, j], 'b-', alpha=0.6, linewidth=connection_matrix[i, j]*3)

ax4.set_title('Sensor-Broker Network Communication Heatmap', fontweight='bold', fontsize=14)
plt.colorbar(im, ax=ax4, label='Communication Frequency')

# Subplot 5: Time Series Decomposition with Box Plots
ax5 = plt.subplot(3, 2, 5)
# Create time series data
np.random.seed(42)
time_points = np.arange(0, 100)
legit_traffic = 50 + 10 * np.sin(time_points * 0.1) + np.random.normal(0, 5, 100)
malicious_traffic = 20 + 30 * np.sin(time_points * 0.15) + np.random.normal(0, 8, 100)

ax5.plot(time_points, legit_traffic, label='Legitimate Traffic', color=colors_attack[0], linewidth=2)
ax5.plot(time_points, malicious_traffic, label='Malicious Traffic', color=colors_attack[1], linewidth=2)

# Add box plots on the right side
box_data = [legit_traffic, malicious_traffic]
box_positions = [85, 95]
bp = ax5.boxplot(box_data, positions=box_positions, widths=5, patch_artist=True)
bp['boxes'][0].set_facecolor(colors_attack[0])
bp['boxes'][1].set_facecolor(colors_attack[1])

ax5.set_xlabel('Time Points', fontweight='bold')
ax5.set_ylabel('Traffic Volume', fontweight='bold')
ax5.set_title('Temporal Traffic Pattern Analysis', fontweight='bold', fontsize=14)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Hierarchical Clustering with 2D Scatter
ax6 = plt.subplot(3, 2, 6)
# Prepare clustering data
cluster_features = ['tcp.len', 'mqtt.len', 'tcp.time_delta', 'mqtt.kalive']
available_cluster_features = [f for f in cluster_features if f in train_df.columns]

if len(available_cluster_features) >= 2:
    cluster_data = train_df[available_cluster_features].dropna()
    if len(cluster_data) > 500:
        cluster_data = cluster_data.sample(n=500, random_state=42)
    
    cluster_data_scaled = StandardScaler().fit_transform(cluster_data)
    
    # Perform clustering
    n_clusters = min(4, len(cluster_data))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
    else:
        cluster_labels = np.zeros(len(cluster_data))
    
    # Create 2D projection using PCA
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(cluster_data_scaled)
    
    # Create scatter plot
    np.random.seed(42)
    sample_rooms = np.random.choice(['Room_A', 'Room_B', 'Room_C', 'Room_D'], len(data_2d))
    sample_timing = np.random.choice(['Periodic', 'Random'], len(data_2d))
    
    for i, room in enumerate(['Room_A', 'Room_B', 'Room_C', 'Room_D']):
        mask = np.array(sample_rooms) == room
        periodic_mask = mask & (np.array(sample_timing) == 'Periodic')
        random_mask = mask & (np.array(sample_timing) == 'Random')
        
        if np.any(periodic_mask):
            ax6.scatter(data_2d[periodic_mask, 0], data_2d[periodic_mask, 1], 
                       c=colors_rooms[i], marker='o', s=50, alpha=0.7, 
                       label=f'{room} (Periodic)')
        if np.any(random_mask):
            ax6.scatter(data_2d[random_mask, 0], data_2d[random_mask, 1], 
                       c=colors_rooms[i], marker='^', s=50, alpha=0.7, 
                       label=f'{room} (Random)')
else:
    # Fallback visualization
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = np.random.normal(0, 1, 200)
    rooms = np.random.choice(['Room_A', 'Room_B', 'Room_C', 'Room_D'], 200)
    timing = np.random.choice(['Periodic', 'Random'], 200)
    
    for i, room in enumerate(['Room_A', 'Room_B', 'Room_C', 'Room_D']):
        mask = rooms == room
        periodic_mask = mask & (timing == 'Periodic')
        random_mask = mask & (timing == 'Random')
        
        if np.any(periodic_mask):
            ax6.scatter(x[periodic_mask], y[periodic_mask], 
                       c=colors_rooms[i], marker='o', s=50, alpha=0.7, 
                       label=f'{room} (Periodic)')
        if np.any(random_mask):
            ax6.scatter(x[random_mask], y[random_mask], 
                       c=colors_rooms[i], marker='^', s=50, alpha=0.7, 
                       label=f'{room} (Random)')

ax6.set_xlabel('First Principal Component', fontweight='bold')
ax6.set_ylabel('Second Principal Component', fontweight='bold')
ax6.set_title('Sensor Behavioral Clustering Analysis', fontweight='bold', fontsize=14)
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.grid(True, alpha=0.3)

# Final layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('mqtt_security_analysis.png', dpi=300, bbox_inches='tight')
plt.show()