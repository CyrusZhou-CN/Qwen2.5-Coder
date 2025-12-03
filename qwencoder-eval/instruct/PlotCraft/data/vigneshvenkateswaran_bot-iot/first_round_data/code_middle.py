import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Load and combine all datasets
file_names = ['data_26.csv', 'data_37.csv', 'data_41.csv', 'data_40.csv', 'data_6.csv', 
              'data_16.csv', 'data_59.csv', 'data_1.csv', 'data_29.csv', 'data_46.csv',
              'data_51.csv', 'data_32.csv', 'data_68.csv', 'data_54.csv', 'data_71.csv',
              'data_30.csv', 'data_42.csv', 'data_65.csv', 'data_19.csv', 'data_63.csv']

# Sample data from each file to create a manageable dataset
all_data = []
for file in file_names[:5]:  # Use only first 5 files to avoid memory issues
    try:
        df = pd.read_csv(file)
        sample_size = min(10000, len(df))
        all_data.append(df.sample(n=sample_size, random_state=42))
    except:
        pass

if all_data:
    data = pd.concat(all_data, ignore_index=True)
else:
    # Create fake data if files don't load
    np.random.seed(42)
    data = pd.DataFrame({
        'proto': np.random.choice(['tcp', 'udp', 'arp'], 50000),
        'category': np.random.choice(['Normal', 'DoS', 'DDoS', 'Reconnaissance'], 50000),
        'subcategory ': np.random.choice(['Normal', 'TCP', 'UDP', 'Service_Scan'], 50000),
        'attack': np.random.choice([0, 1], 50000),
        'saddr': [f'192.168.100.{np.random.randint(1,255)}' for _ in range(50000)],
        'bytes': np.random.randint(100, 2000, 50000),
        'dur': np.random.uniform(0.1, 100, 50000),
        'pkts': np.random.randint(1, 50, 50000)
    })

# Create the sabotaged 1x3 layout instead of requested 2x2
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.02, right=0.98, top=0.85, bottom=0.15)

# SUBPLOT 1: Scatter plot instead of stacked bar (completely wrong chart type)
ax1 = axes[0]
proto_counts = data['proto'].value_counts()
x_pos = np.arange(len(proto_counts))
y_vals = proto_counts.values

# Use scatter plot with terrible colors
colors = ['#FF00FF', '#00FFFF', '#FFFF00']  # Bright, clashing colors
ax1.scatter(x_pos, y_vals, c=colors[:len(x_pos)], s=500, alpha=0.8, edgecolors='red', linewidth=3)
ax1.set_title('Amplitude Distribution', fontsize=8)  # Wrong title
ax1.set_xlabel('Time', fontsize=8)  # Swapped labels
ax1.set_ylabel('Protocol', fontsize=8)
ax1.grid(True, color='white', linewidth=2)  # Heavy white grid on dark background

# SUBPLOT 2: Bar chart instead of pie chart
ax2 = axes[1]
subcat_counts = data['subcategory '].value_counts()
bars = ax2.bar(range(len(subcat_counts)), subcat_counts.values, 
               color=['red', 'green', 'blue', 'yellow'][:len(subcat_counts)])
ax2.set_title('Glarbnok\'s Revenge Analysis', fontsize=8)  # Nonsensical title
ax2.set_xlabel('Bytes Transferred', fontsize=8)  # Wrong labels
ax2.set_ylabel('Source IP Count', fontsize=8)
ax2.tick_params(axis='x', rotation=90, labelsize=6)

# Add overlapping text annotations
for i, bar in enumerate(bars):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
             f'Data Series {i}', ha='center', va='center', fontsize=12, 
             color='white', weight='bold')

# SUBPLOT 3: Line plot instead of treemap
ax3 = axes[2]
# Create random time series data
time_data = np.linspace(0, 10, 100)
random_data = np.random.random(100) * 100

ax3.plot(time_data, random_data, color='cyan', linewidth=4, marker='o', markersize=8)
ax3.fill_between(time_data, random_data, alpha=0.3, color='magenta')
ax3.set_title('Network Packet Hierarchical Structure', fontsize=8)  # Misleading title
ax3.set_xlabel('Attack Duration (seconds)', fontsize=8)
ax3.set_ylabel('Percentage Contribution', fontsize=8)
ax3.grid(True, color='yellow', linewidth=1, alpha=0.7)

# Add overlapping legend that covers data
legend_elements = [mpatches.Patch(color='red', label='Category A'),
                   mpatches.Patch(color='blue', label='Category B'),
                   mpatches.Patch(color='green', label='Category C')]
ax3.legend(handles=legend_elements, loc='center', fontsize=14, 
           bbox_to_anchor=(0.5, 0.5), framealpha=0.9)

# Add main title that overlaps with subplots
fig.suptitle('Comprehensive Multi-Dimensional Cyber Security Attack Pattern Visualization Dashboard', 
             fontsize=16, y=0.95, color='white', weight='bold')

# Add random text annotations that overlap everything
fig.text(0.3, 0.7, 'CRITICAL ALERT', fontsize=20, color='red', weight='bold', rotation=45)
fig.text(0.7, 0.3, 'SYSTEM BREACH DETECTED', fontsize=15, color='yellow', weight='bold', rotation=-30)

plt.savefig('chart.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.close()