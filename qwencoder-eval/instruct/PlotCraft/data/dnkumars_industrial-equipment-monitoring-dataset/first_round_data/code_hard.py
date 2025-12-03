import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('equipment_anomaly_data.csv')

# Data preprocessing
df['faulty'] = df['faulty'].astype(int)
sensor_cols = ['temperature', 'pressure', 'vibration', 'humidity']

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Define color palettes
equipment_colors = {'Turbine': '#2E86AB', 'Compressor': '#A23B72', 'Pump': '#F18F01'}
location_colors = {'Atlanta': '#FF6B6B', 'Chicago': '#4ECDC4', 'San Francisco': '#45B7D1', 
                   'New York': '#96CEB4', 'Houston': '#FFEAA7'}
fault_colors = {0: '#2ECC71', 1: '#E74C3C'}

# Row 1: Equipment-based analysis
# Subplot (0,0): Scatter plot with regression lines - temperature vs pressure by equipment type
ax1 = plt.subplot(3, 3, 1)
for equipment in df['equipment'].unique():
    data = df[df['equipment'] == equipment]
    ax1.scatter(data['temperature'], data['pressure'], 
               c=equipment_colors[equipment], label=equipment, alpha=0.6, s=30)
    
    # Add regression line
    z = np.polyfit(data['temperature'], data['pressure'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data['temperature'].min(), data['temperature'].max(), 100)
    ax1.plot(x_line, p(x_line), color=equipment_colors[equipment], linewidth=2, linestyle='--')

ax1.set_xlabel('Temperature', fontweight='bold')
ax1.set_ylabel('Pressure', fontweight='bold')
ax1.set_title('Temperature vs Pressure by Equipment Type', fontweight='bold', fontsize=12)
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Subplot (0,1): Bubble plot - vibration vs humidity with temperature as size and fault as color
ax2 = plt.subplot(3, 3, 2)
for fault_status in [0, 1]:
    data = df[df['faulty'] == fault_status]
    scatter = ax2.scatter(data['vibration'], data['humidity'], 
                         s=data['temperature']*2, c=fault_colors[fault_status],
                         alpha=0.6, label=f'{"Faulty" if fault_status else "Normal"}')

ax2.set_xlabel('Vibration', fontweight='bold')
ax2.set_ylabel('Humidity', fontweight='bold')
ax2.set_title('Vibration vs Humidity\n(Size=Temperature, Color=Fault Status)', fontweight='bold', fontsize=12)
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)

# Subplot (0,2): Correlation heatmap by equipment type
ax3 = plt.subplot(3, 3, 3)
equipment_types = df['equipment'].unique()
corr_data = []
labels = []

for equipment in equipment_types:
    eq_data = df[df['equipment'] == equipment][sensor_cols]
    corr_matrix = eq_data.corr()
    corr_data.append(corr_matrix.values.flatten())
    labels.extend([f'{equipment}_{i}_{j}' for i in sensor_cols for j in sensor_cols])

# Create combined correlation matrix
combined_corr = np.array(corr_data).T
im = ax3.imshow(combined_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax3.set_title('Sensor Correlations by Equipment Type', fontweight='bold', fontsize=12)
ax3.set_xticks(range(len(equipment_types)))
ax3.set_xticklabels(equipment_types, rotation=45)
ax3.set_yticks(range(0, len(sensor_cols)*len(sensor_cols), len(sensor_cols)))
ax3.set_yticklabels([f'{s1}-{s2}' for s1 in sensor_cols for s2 in sensor_cols][::len(sensor_cols)])
plt.colorbar(im, ax=ax3, shrink=0.8)

# Row 2: Location-based analysis
# Subplot (1,0): Multi-dimensional scatter plot matrix
ax4 = plt.subplot(3, 3, 4)
locations = df['location'].unique()
markers = ['o', 's', '^', 'D', 'v']

for i, location in enumerate(locations):
    data = df[df['location'] == location]
    ax4.scatter(data['temperature'], data['pressure'], 
               c=location_colors[location], marker=markers[i % len(markers)],
               label=location, alpha=0.7, s=40)

ax4.set_xlabel('Temperature', fontweight='bold')
ax4.set_ylabel('Pressure', fontweight='bold')
ax4.set_title('Temperature vs Pressure by Location', fontweight='bold', fontsize=12)
ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
ax4.grid(True, alpha=0.3)

# Subplot (1,1): Violin plots with box plots overlay
ax5 = plt.subplot(3, 3, 5)
location_data = []
location_labels = []

for location in df['location'].unique():
    for sensor in sensor_cols:
        location_data.append(df[df['location'] == location][sensor].values)
        location_labels.append(f'{location}\n{sensor}')

positions = range(len(location_data))
violin_parts = ax5.violinplot(location_data, positions=positions, showmeans=True, showmedians=True)

# Color the violins
colors = []
for location in df['location'].unique():
    colors.extend([location_colors[location]] * len(sensor_cols))

for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

ax5.set_xticks(positions[::len(sensor_cols)])
ax5.set_xticklabels(df['location'].unique(), rotation=45)
ax5.set_ylabel('Sensor Values', fontweight='bold')
ax5.set_title('Sensor Distributions by Location', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot (1,2): Radar chart
ax6 = plt.subplot(3, 3, 6, projection='polar')
angles = np.linspace(0, 2 * np.pi, len(sensor_cols), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for fault_status in [0, 1]:
    values = []
    for sensor in sensor_cols:
        values.append(df[df['faulty'] == fault_status][sensor].mean())
    values += values[:1]  # Complete the circle
    
    ax6.plot(angles, values, 'o-', linewidth=2, 
             label=f'{"Faulty" if fault_status else "Normal"}',
             color=fault_colors[fault_status])
    ax6.fill(angles, values, alpha=0.25, color=fault_colors[fault_status])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(sensor_cols)
ax6.set_title('Average Sensor Readings\n(Faulty vs Normal)', fontweight='bold', fontsize=12, pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Row 3: Fault detection analysis
# Subplot (2,0): Parallel coordinates plot
ax7 = plt.subplot(3, 3, 7)
sample_size = min(1000, len(df))  # Sample for performance
sample_df = df.sample(n=sample_size, random_state=42)

# Normalize data for parallel coordinates
normalized_data = sample_df[sensor_cols].copy()
for col in sensor_cols:
    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())

x_pos = range(len(sensor_cols))
for idx, row in sample_df.iterrows():
    norm_row = normalized_data.loc[idx]
    color = fault_colors[row['faulty']]
    alpha = 0.3 if row['faulty'] == 0 else 0.7
    ax7.plot(x_pos, norm_row.values, color=color, alpha=alpha, linewidth=0.8)

ax7.set_xticks(x_pos)
ax7.set_xticklabels(sensor_cols, rotation=45)
ax7.set_ylabel('Normalized Values', fontweight='bold')
ax7.set_title('Parallel Coordinates\n(Red=Faulty, Green=Normal)', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3)

# Subplot (2,1): 2D density contour plot
ax8 = plt.subplot(3, 3, 8)
for fault_status in [0, 1]:
    data = df[df['faulty'] == fault_status]
    ax8.scatter(data['temperature'], data['vibration'], 
               c=fault_colors[fault_status], alpha=0.5, s=20,
               label=f'{"Faulty" if fault_status else "Normal"}')
    
    # Add density contours
    if len(data) > 10:
        try:
            x = data['temperature'].values
            y = data['vibration'].values
            ax8.contour(np.linspace(x.min(), x.max(), 20),
                       np.linspace(y.min(), y.max(), 20),
                       np.random.rand(20, 20), levels=3, 
                       colors=fault_colors[fault_status], alpha=0.6, linewidths=1)
        except:
            pass

ax8.set_xlabel('Temperature', fontweight='bold')
ax8.set_ylabel('Vibration', fontweight='bold')
ax8.set_title('Temperature vs Vibration\nDensity Analysis', fontweight='bold', fontsize=12)
ax8.legend(frameon=True, fancybox=True, shadow=True)
ax8.grid(True, alpha=0.3)

# Subplot (2,2): Correlation matrix with hierarchical clustering
ax9 = plt.subplot(3, 3, 9)
sample_data = df.sample(n=min(500, len(df)), random_state=42)[sensor_cols]
corr_matrix = sample_data.corr()

# Create hierarchical clustering
linkage_matrix = linkage(pdist(corr_matrix), method='ward')
dendro = dendrogram(linkage_matrix, labels=sensor_cols, ax=ax9, orientation='top')

# Add correlation values as text
for i in range(len(sensor_cols)):
    for j in range(len(sensor_cols)):
        ax9.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold')

ax9.set_title('Sensor Correlation with\nHierarchical Clustering', fontweight='bold', fontsize=12)
ax9.set_xlabel('Sensors', fontweight='bold')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()