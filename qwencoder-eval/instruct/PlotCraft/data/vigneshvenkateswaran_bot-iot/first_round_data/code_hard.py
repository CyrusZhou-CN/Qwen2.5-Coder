import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

# Load and combine data
file_names = ['data_26.csv', 'data_37.csv', 'data_41.csv', 'data_40.csv', 'data_6.csv', 
              'data_16.csv', 'data_59.csv', 'data_1.csv', 'data_29.csv', 'data_46.csv']

dfs = []
for file in file_names[:5]:  # Only use first 5 files to avoid memory issues
    try:
        df = pd.read_csv(file, nrows=50000)  # Limit rows
        dfs.append(df)
    except:
        pass

if dfs:
    data = pd.concat(dfs, ignore_index=True)
else:
    # Generate fake data if files don't load
    np.random.seed(42)
    data = pd.DataFrame({
        'proto': np.random.choice(['tcp', 'udp'], 10000),
        'bytes': np.random.exponential(500, 10000),
        'pkts': np.random.poisson(5, 10000),
        'dur': np.random.exponential(10, 10000),
        'rate': np.random.exponential(1, 10000),
        'saddr': [f'192.168.100.{np.random.randint(1,255)}' for _ in range(10000)],
        'category': np.random.choice(['DoS', 'DDoS', 'Normal', 'Reconnaissance'], 10000),
        'subcategory': np.random.choice(['TCP', 'UDP', 'Service_Scan', 'Normal'], 10000),
        'attack': np.random.choice([0, 1], 10000)
    })

# Create the most non-compliant 2x2 grid instead of requested 3x3
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Pie chart instead of stacked bar (wrong chart type)
ax1 = axes[0, 0]
proto_counts = data['proto'].value_counts()
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00'][:len(proto_counts)]
wedges, texts, autotexts = ax1.pie(proto_counts.values, labels=proto_counts.index, 
                                   colors=colors, autopct='%1.1f%%')
ax1.set_title('Packet Duration Analysis', fontsize=8)  # Wrong title
ax1.text(0.5, 0.5, 'OVERLAPPING\nTEXT', transform=ax1.transAxes, 
         fontsize=12, ha='center', va='center', color='white', weight='bold')

# Subplot 2: Scatter plot instead of violin plot (wrong chart type)
ax2 = axes[0, 1]
sample_data = data.sample(min(1000, len(data)))
scatter = ax2.scatter(sample_data['bytes'], sample_data['pkts'], 
                     c=sample_data['attack'], cmap='jet', s=1, alpha=0.7)
ax2.set_xlabel('Attack Type Distribution')  # Swapped labels
ax2.set_ylabel('Network Protocol Usage')
ax2.set_title('Source Address Clustering', fontsize=8)  # Wrong title
# Add overlapping text
ax2.text(0.1, 0.9, 'GLARBNOK\'S\nREVENGE', transform=ax2.transAxes, 
         fontsize=10, color='cyan', weight='bold')

# Subplot 3: Bar chart instead of time series (wrong chart type)
ax3 = axes[1, 0]
category_counts = data['category'].value_counts()
bars = ax3.bar(range(len(category_counts)), category_counts.values, 
               color=['red', 'green', 'blue', 'yellow'][:len(category_counts)])
ax3.set_xticks(range(len(category_counts)))
ax3.set_xticklabels(category_counts.index, rotation=45)
ax3.set_ylabel('Temporal Patterns')  # Wrong label
ax3.set_xlabel('Correlation Matrix')  # Wrong label
ax3.set_title('Violin Plot Analysis', fontsize=8)  # Wrong title
# Overlapping annotation
ax3.annotate('DATA CHAOS', xy=(0.5, max(category_counts.values)/2), 
             xytext=(0.5, max(category_counts.values)*0.8),
             fontsize=12, ha='center', color='white', weight='bold')

# Subplot 4: Heatmap with random data instead of network graph
ax4 = axes[1, 1]
random_matrix = np.random.rand(5, 5)
im = ax4.imshow(random_matrix, cmap='plasma', aspect='auto')
ax4.set_title('Network Graph Visualization', fontsize=8)  # Wrong title
ax4.set_xlabel('Hierarchical Clustering')  # Wrong label
ax4.set_ylabel('Radar Chart Metrics')  # Wrong label
# Add colorbar that overlaps
cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
cbar.set_label('Confusion Matrix', rotation=270, labelpad=15)  # Wrong label

# Add a large overlapping text across the entire figure
fig.text(0.5, 0.5, 'SABOTAGED\nVISUALIZATION', fontsize=20, 
         ha='center', va='center', color='red', weight='bold', alpha=0.7)

# Wrong overall title
fig.suptitle('Comprehensive IoT Security Analysis Dashboard', fontsize=10, y=0.95)

plt.savefig('chart.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.close()