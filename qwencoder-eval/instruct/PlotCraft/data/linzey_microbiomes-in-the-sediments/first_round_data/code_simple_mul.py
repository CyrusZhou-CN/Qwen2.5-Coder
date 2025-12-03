import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
taxonomy_df = pd.read_csv('Supplementary Fig.3.csv')
metadata_df = pd.read_csv('Fig.1bc.csv')

# Merge datasets on sample names
merged_df = pd.merge(taxonomy_df, metadata_df, left_on='Samples', right_on='Sample', how='inner')

# Extract depth from sample names (assuming format like "MC02(8-10cmbsf)")
def extract_depth(sample_name):
    if '(' in sample_name and 'cmbsf' in sample_name:
        depth_part = sample_name.split('(')[1].split('cmbsf')[0]
        if '-' in depth_part:
            # Take the middle of the range
            start, end = depth_part.split('-')
            return (float(start) + float(end)) / 2
        else:
            return float(depth_part)
    return 0

merged_df['Depth_cm'] = merged_df['Samples'].apply(extract_depth)

# Get top 8 most abundant phyla (excluding 'Samples' column)
phyla_columns = [col for col in taxonomy_df.columns if col != 'Samples']
mean_abundances = taxonomy_df[phyla_columns].mean().sort_values(ascending=False)
top_8_phyla = mean_abundances.head(8).index.tolist()

# Select 4 key phyla for line plots (top 4 most abundant)
key_phyla = top_8_phyla[:4]

# Define color palettes
area_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
line_colors = ['#000000', '#FF0000', '#0000FF', '#008000']

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Separate data by group
slope_data = merged_df[merged_df['Group'] == 'Slope'].copy()
bottom_data = merged_df[merged_df['Group'] == 'Bottom'].copy()

# Sort by depth for proper plotting
slope_data = slope_data.sort_values('Depth_cm')
bottom_data = bottom_data.sort_values('Depth_cm')

# Function to create combined stacked area and line plot
def create_combined_plot(ax, data, title, x_label):
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold', fontsize=14)
        return
    
    x_values = data['Depth_cm'].values
    
    # Create stacked area chart for top 8 phyla
    y_values = []
    for phylum in top_8_phyla:
        y_values.append(data[phylum].values)
    
    # Stack the areas
    ax.stackplot(x_values, *y_values, labels=top_8_phyla, colors=area_colors, alpha=0.7)
    
    # Overlay line plots for key phyla
    for i, phylum in enumerate(key_phyla):
        ax.plot(x_values, data[phylum].values, color=line_colors[i], 
                linewidth=3, marker='o', markersize=6, label=f'{phylum} (line)')
    
    # Styling
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Relative Abundance', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)

# Top row: Slope vs Bottom comparison
if len(slope_data) > 0:
    create_combined_plot(axes[0, 0], slope_data, 'Slope Position - Depth Profile', 'Sediment Depth (cm)')
    
    # Create legend for slope plot
    area_handles = [plt.Rectangle((0,0),1,1, facecolor=area_colors[i], alpha=0.7) for i in range(len(top_8_phyla))]
    line_handles = [plt.Line2D([0], [0], color=line_colors[i], linewidth=3, marker='o') for i in range(len(key_phyla))]
    
    area_labels = [f'{phylum} (area)' for phylum in top_8_phyla]
    line_labels = [f'{phylum} (line)' for phylum in key_phyla]
    
    axes[0, 0].legend(area_handles + line_handles, area_labels + line_labels, 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

if len(bottom_data) > 0:
    create_combined_plot(axes[0, 1], bottom_data, 'Bottom Position - Depth Profile', 'Sediment Depth (cm)')
else:
    # If no bottom data, show slope data with different grouping
    axes[0, 1].text(0.5, 0.5, 'Limited Bottom Position Data\nShowing Additional Slope Analysis', 
                    ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].set_title('Bottom Position - Depth Profile', fontweight='bold', fontsize=14)

# Bottom row: Temporal changes in deepest samples
# Get deepest samples (>20 cm depth)
deep_slope = slope_data[slope_data['Depth_cm'] > 20].copy()
deep_bottom = bottom_data[bottom_data['Depth_cm'] > 20].copy() if len(bottom_data) > 0 else pd.DataFrame()

# If insufficient deep bottom data, use different depth threshold for slope
if len(deep_bottom) < 3:
    deep_samples_alt = slope_data[slope_data['Depth_cm'] > 15].copy()
    create_combined_plot(axes[1, 0], deep_samples_alt, 'Deep Slope Samples (>15cm) - Temporal Changes', 'Sediment Depth (cm)')
else:
    create_combined_plot(axes[1, 0], deep_slope, 'Deep Slope Samples (>20cm) - Temporal Changes', 'Sediment Depth (cm)')

if len(deep_bottom) > 0:
    create_combined_plot(axes[1, 1], deep_bottom, 'Deep Bottom Samples (>20cm) - Temporal Changes', 'Sediment Depth (cm)')
else:
    # Show surface samples comparison instead
    surface_slope = slope_data[slope_data['Depth_cm'] <= 10].copy()
    create_combined_plot(axes[1, 1], surface_slope, 'Surface Slope Samples (≤10cm) - Depth Profile', 'Sediment Depth (cm)')

# Add depth markers on x-axes
for ax in axes.flat:
    if ax.get_xlim()[1] > 0:  # Only if there's actual data
        depth_markers = [0, 10, 20, 30, 40]
        for depth in depth_markers:
            if depth <= ax.get_xlim()[1]:
                ax.axvline(x=depth, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add main title
fig.suptitle('Microbial Community Composition Changes with Sediment Depth\nAcross Different Trench Positions', 
             fontweight='bold', fontsize=16, y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

# Add a comprehensive legend outside the plots
legend_elements = []
for i, phylum in enumerate(top_8_phyla):
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=area_colors[i], alpha=0.7, label=f'{phylum} (stacked area)'))

for i, phylum in enumerate(key_phyla):
    legend_elements.append(plt.Line2D([0], [0], color=line_colors[i], linewidth=3, marker='o', 
                                     label=f'{phylum} (trend line)'))

fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=11)

plt.show()