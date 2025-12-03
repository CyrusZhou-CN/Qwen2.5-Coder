import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df_pt = pd.read_csv('artigos_rna_pt.csv')
df_en = pd.read_csv('artigos_rna_ing.csv')

# Data preprocessing
# Calculate text lengths
df_pt['text_length'] = df_pt['Texto'].str.len()
df_en['text_length'] = df_en['Texto'].str.len()

# Count unique titles for frequency analysis
pt_title_counts = df_pt['Título'].value_counts()
en_title_counts = df_en['Título'].value_counts()

# Create 2x1 subplot layout with white background
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Top plot: Histogram with overlaid KDE curve for title frequency distribution
# Create frequency data for histogram
pt_freq_data = pt_title_counts.values
en_freq_data = en_title_counts.values

# Plot histograms with transparency and improved binning
bins = np.arange(0.5, 11.5, 1)  # Bins from 1 to 10 for better visibility
ax1.hist(pt_freq_data, bins=bins, alpha=0.6, color='#2E86AB', 
         label='Portuguese Histogram', density=True, edgecolor='white', linewidth=0.5)
ax1.hist(en_freq_data, bins=bins, alpha=0.6, color='#A23B72', 
         label='English Histogram', density=True, edgecolor='white', linewidth=0.5)

# Add KDE curves for both datasets with proper data filtering
# Filter data to focus on the meaningful range (1-10 occurrences)
pt_freq_filtered = pt_freq_data[pt_freq_data <= 10]
en_freq_filtered = en_freq_data[en_freq_data <= 10]

if len(pt_freq_filtered) > 1:
    sns.kdeplot(data=pt_freq_filtered, ax=ax1, color='#1B5E7A', 
                linewidth=2.5, label='Portuguese KDE')
if len(en_freq_filtered) > 1:
    sns.kdeplot(data=en_freq_filtered, ax=ax1, color='#7A1B4B', 
                linewidth=2.5, label='English KDE')

# Set appropriate x-axis limits to show the relevant data range
ax1.set_xlim(0.5, 10.5)

ax1.set_title('Distribution of Article Title Frequencies Across Languages', 
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Frequency of Unique Titles', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_facecolor('white')

# Set integer ticks on x-axis for better readability
ax1.set_xticks(range(1, 11))

# Bottom plot: Violin plot comparing text length distributions
# Create violin plot with custom colors
violin_parts = ax2.violinplot([df_pt['text_length'], df_en['text_length']], 
                              positions=[1, 2], widths=0.6, showmeans=True, 
                              showmedians=True, showextrema=True)

# Customize violin plot colors
colors = ['#2E86AB', '#A23B72']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('white')
    pc.set_linewidth(1)

# Customize other violin plot elements
violin_parts['cmeans'].set_color('black')
violin_parts['cmeans'].set_linewidth(2)
violin_parts['cmedians'].set_color('white')
violin_parts['cmedians'].set_linewidth(2)
violin_parts['cbars'].set_color('black')
violin_parts['cmaxes'].set_color('black')
violin_parts['cmins'].set_color('black')

ax2.set_title('Text Length Distribution Comparison Between Languages', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Language', fontsize=12)
ax2.set_ylabel('Text Length (characters)', fontsize=12)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Portuguese', 'English'])
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_facecolor('white')

# Add sample size annotations with improved positioning to avoid overlap
y_max = ax2.get_ylim()[1]
annotation_height = y_max * 0.95  # Position higher to avoid touching violin plots

ax2.text(1, annotation_height, f'n = {len(df_pt)}', 
         ha='center', va='center', fontsize=10, 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                  edgecolor='#2E86AB', linewidth=1.5, alpha=0.9))
ax2.text(2, annotation_height, f'n = {len(df_en)}', 
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                  edgecolor='#A23B72', linewidth=1.5, alpha=0.9))

# Create custom legend for violin plot
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2E86AB', alpha=0.7, label='Portuguese'),
                   Patch(facecolor='#A23B72', alpha=0.7, label='English')]
ax2.legend(handles=legend_elements, frameon=True, fancybox=True, shadow=True, loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()