import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import squarify

# Set font to support Bengali characters
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans Bengali']

# Load data
union_df = pd.read_csv('union.csv')
upozila_df = pd.read_csv('upozila.csv')
district_df = pd.read_csv('district.csv')

# Data preprocessing - merge all datasets
merged_df = union_df.merge(upozila_df, on=['district_id', 'upozila_id'])
merged_df = merged_df.merge(district_df, on='district_id')

# Calculate statistics for each district and upazila
district_stats = merged_df.groupby(['district_id', 'জেলা']).agg({
    'upozila_id': 'nunique',
    'ইউনিয়ন': 'count'
}).rename(columns={'upozila_id': 'num_upazilas', 'ইউনিয়ন': 'num_unions'}).reset_index()

upazila_stats = merged_df.groupby(['district_id', 'জেলা', 'upozila_id', 'উপজেলা']).agg({
    'ইউনিয়ন': 'count'
}).rename(columns={'ইউনিয়ন': 'num_unions'}).reset_index()

# Create figure with white background
fig = plt.figure(figsize=(16, 12), facecolor='white')

# Create treemap (top subplot)
ax1 = plt.subplot(2, 1, 1)

# Get top 10 districts by union count for consistency
top_10_districts = district_stats.nlargest(10, 'num_unions')

# Prepare hierarchical data for treemap
treemap_data = []
treemap_labels = []
treemap_colors = []

# Generate color palette for districts
district_colors = plt.cm.Set3(np.linspace(0, 1, len(top_10_districts)))

# Create nested structure: districts containing upazilas sized by union count
for i, (_, district_row) in enumerate(top_10_districts.iterrows()):
    district_id = district_row['district_id']
    district_name = district_row['জেলা']
    
    # Get upazilas for this district
    district_upazilas = upazila_stats[upazila_stats['district_id'] == district_id].copy()
    
    # Add district-level entry (parent rectangle)
    district_total_unions = district_upazilas['num_unions'].sum()
    treemap_data.append(district_total_unions)
    treemap_labels.append(f"{district_name}\n({district_total_unions} unions)")
    treemap_colors.append(district_colors[i])

# Create treemap with proper hierarchical sizing
squarify.plot(sizes=treemap_data, label=treemap_labels, color=treemap_colors, 
              alpha=0.7, text_kwargs={'fontsize': 10, 'weight': 'bold', 'color': 'black'})

ax1.set_title('Administrative Structure: Top 10 Districts by Union Count\n(Rectangle size proportional to total unions per district)', 
              fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

# Create horizontal bar chart (bottom subplot)
ax2 = plt.subplot(2, 1, 2)

# Create color map based on number of upazilas
upazila_counts = top_10_districts['num_upazilas']
norm = plt.Normalize(vmin=upazila_counts.min(), vmax=upazila_counts.max())
colors_bar = plt.cm.viridis(norm(upazila_counts))

# Create horizontal bar chart
bars = ax2.barh(range(len(top_10_districts)), top_10_districts['num_unions'], 
                color=colors_bar, alpha=0.8, edgecolor='white', linewidth=1.5)

# Customize the bar chart
ax2.set_yticks(range(len(top_10_districts)))
ax2.set_yticklabels(top_10_districts['জেলা'], fontsize=11)
ax2.set_xlabel('Number of Unions', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Districts by Union Count\n(Bar color indicates number of upazilas)', 
              fontsize=14, fontweight='bold', pad=15)

# Add value labels on bars with right alignment
for i, (bar, unions, upazilas) in enumerate(zip(bars, top_10_districts['num_unions'], 
                                                top_10_districts['num_upazilas'])):
    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
             f'{unions} unions ({upazilas} upazilas)', 
             va='center', ha='left', fontsize=9, fontweight='bold')

# Add colorbar for upazila count with improved thickness
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', shrink=0.6, pad=0.02, 
                   aspect=15)  # aspect parameter makes colorbar thicker
cbar.set_label('Number of Upazilas', fontsize=10, fontweight='bold')

# Style the axes
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)

# Invert y-axis to show highest values at top
ax2.invert_yaxis()

# Set x-axis limits to accommodate text labels
max_unions = top_10_districts['num_unions'].max()
ax2.set_xlim(0, max_unions * 1.25)

# Adjust layout
plt.tight_layout(pad=3.0)
plt.show()

# Create a second figure showing the detailed hierarchical treemap
fig2 = plt.figure(figsize=(18, 10), facecolor='white')

# Prepare data for detailed hierarchical treemap showing upazilas within districts
# Select top 6 districts for better readability in hierarchical view
top_6_districts = district_stats.nlargest(6, 'num_unions')

# Create nested treemap data
nested_data = []
nested_labels = []
nested_colors = []

district_color_map = plt.cm.Set3(np.linspace(0, 1, len(top_6_districts)))

for i, (_, district_row) in enumerate(top_6_districts.iterrows()):
    district_id = district_row['district_id']
    district_name = district_row['জেলা']
    base_color = district_color_map[i]
    
    # Get upazilas for this district
    district_upazilas = upazila_stats[upazila_stats['district_id'] == district_id].copy()
    
    # Create color variations for upazilas within the same district
    upazila_colors = plt.cm.get_cmap('Set3')(np.linspace(i/len(top_6_districts), 
                                                         (i+1)/len(top_6_districts), 
                                                         len(district_upazilas)))
    
    for j, (_, upazila_row) in enumerate(district_upazilas.iterrows()):
        # Size proportional to number of unions (key requirement)
        size = upazila_row['num_unions']
        label = f"{district_name}\n{upazila_row['উপজেলা']}\n{size} unions"
        
        nested_data.append(size)
        nested_labels.append(label)
        nested_colors.append(upazila_colors[j])

# Create detailed hierarchical treemap
squarify.plot(sizes=nested_data, label=nested_labels, color=nested_colors, 
              alpha=0.8, text_kwargs={'fontsize': 8, 'weight': 'bold', 'color': 'black'})

plt.title('Detailed Administrative Hierarchy: Districts → Upazilas\n(Rectangle size proportional to union count per upazila)', 
          fontsize=16, fontweight='bold', pad=20)
plt.axis('off')

plt.tight_layout()
plt.show()