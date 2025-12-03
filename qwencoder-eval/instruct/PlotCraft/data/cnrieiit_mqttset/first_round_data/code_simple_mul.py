import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Check if file exists and load data with proper error handling
file_path = 'train70.csv'
if not os.path.exists(file_path):
    # Try alternative file paths
    possible_paths = ['./train70.csv', '../train70.csv', 'data/train70.csv']
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    else:
        raise FileNotFoundError(f"Could not find train70.csv in current directory or common subdirectories")

# Load data
df = pd.read_csv(file_path)

# Calculate the distribution of attack types
target_counts = df['target'].value_counts()
total_records = len(df)

# Calculate percentages
target_percentages = (target_counts / total_records) * 100

# Prepare data for pie chart
labels = target_counts.index.tolist()
sizes = target_percentages.values.tolist()

# Create a color palette that distinguishes legitimate from malicious traffic
colors = []
for label in labels:
    if label.lower() == 'legitimate':
        colors.append('#2E8B57')  # Sea green for legitimate traffic
    else:
        # Different shades of red/orange for different attack types
        attack_colors = {
            'SlowITe': '#FF6B6B',
            'Bruteforce': '#FF4757',
            'Malformed data': '#FF3838',
            'Flooding': '#FF2D2D',
            'DoS attack': '#FF1E1E'
        }
        colors.append(attack_colors.get(label, '#FF0000'))

# Create the pie chart
plt.figure(figsize=(12, 10))

# Create pie chart with percentage labels
wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%',
                                  startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})

# Enhance the appearance of percentage labels
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# Enhance the appearance of category labels
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

# Add a title
plt.title('Distribution of Attack Types in MQTT Network Security Dataset', 
          fontsize=16, fontweight='bold', pad=20)

# Add a legend to better distinguish categories
legend_labels = [f'{label}: {size:.2f}%' for label, size in zip(labels, sizes)]
plt.legend(wedges, legend_labels, title="Traffic Categories", 
          loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10, title_fontsize=12)

# Ensure equal aspect ratio for circular pie chart
plt.axis('equal')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('mqtt_attack_distribution_pie_chart.png', dpi=300, bbox_inches='tight')

# Print summary statistics
print("\nMQTT Network Traffic Distribution Summary:")
print("=" * 50)
for label, percentage in zip(labels, sizes):
    count = target_counts[label]
    print(f"{label}: {count:,} records ({percentage:.2f}%)")
print(f"\nTotal Records: {total_records:,}")