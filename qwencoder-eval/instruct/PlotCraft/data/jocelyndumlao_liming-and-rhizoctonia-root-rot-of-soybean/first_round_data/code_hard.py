import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('Dataset.xlsx')

# Data preprocessing
df['Site_Type'] = df['Site'].apply(lambda x: 'Inside' if 'Patch' in str(x) else 'Outside')
soil_types = df['Field'].unique()

# Create the comprehensive 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.patch.set_facecolor('white')

# Color palettes
colors_soil = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
colors_dose = ['#440154', '#31688E', '#35B779', '#FDE725']

# Row 1 - Disease Progression Analysis

# Subplot 1: AUDPC trends with error bars
ax1 = axes[0, 0]
for i, soil in enumerate(soil_types):
    soil_data = df[df['Field'] == soil]
    dose_means = soil_data.groupby('Dose')['AUDPC'].agg(['mean', 'std']).reset_index()
    
    ax1.errorbar(dose_means['Dose'], dose_means['mean'], yerr=dose_means['std'], 
                marker='o', linewidth=2, capsize=5, label=f'{soil}', 
                color=colors_soil[i], alpha=0.8)
    
    # Add scatter points
    ax1.scatter(soil_data['Dose'], soil_data['AUDPC'], 
               color=colors_soil[i], alpha=0.3, s=15)

ax1.set_title('AUDPC Trends Across Limestone Doses', fontweight='bold', fontsize=10)
ax1.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax1.set_ylabel('AUDPC')
ax1.legend(title='Soil Type', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Disease incidence bars with yield line
ax2 = axes[0, 1]
x_pos = np.arange(len(df['Dose'].unique()))
width = 0.15

for i, soil in enumerate(soil_types):
    soil_data = df[df['Field'] == soil]
    inc_means = soil_data.groupby('Dose')['INC'].mean()
    ax2.bar(x_pos + i*width, inc_means.values, width, 
           label=f'{soil}', color=colors_soil[i], alpha=0.7)

ax2.set_title('Disease Incidence by Dose and Soil Type', fontweight='bold', fontsize=10)
ax2.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax2.set_ylabel('Disease Incidence (%)')
ax2.set_xticks(x_pos + width*1.5)
ax2.set_xticklabels(df['Dose'].unique())
ax2.legend(fontsize=8)

# Subplot 3: Disease reduction area chart
ax3 = axes[0, 2]
doses = sorted(df['Dose'].unique())
for i, soil in enumerate(soil_types):
    soil_data = df[df['Field'] == soil]
    baseline = soil_data[soil_data['Dose'] == 0]['AUDPC'].mean()
    reductions = []
    for dose in doses:
        current = soil_data[soil_data['Dose'] == dose]['AUDPC'].mean()
        reduction = max(0, baseline - current)
        reductions.append(reduction)
    
    ax3.plot(doses, reductions, marker='o', linewidth=2, 
            label=soil, color=colors_soil[i])

ax3.set_title('Disease Reduction by Limestone Dose', fontweight='bold', fontsize=10)
ax3.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax3.set_ylabel('Disease Reduction (AUDPC units)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Row 2 - Plant Development Metrics

# Subplot 4: Plant height metrics
ax4 = axes[1, 0]
for i, location in enumerate(['Inside', 'Outside']):
    loc_data = df[df['Site_Type'] == location]
    plh_means = loc_data.groupby('Dose')['PLH'].mean()
    insh_means = loc_data.groupby('Dose')['INSH'].mean()
    
    ax4.plot(plh_means.index, plh_means.values, 
            marker='o', linewidth=2, label=f'PLH {location}', 
            color=colors_dose[i])
    ax4.plot(insh_means.index, insh_means.values, 
            marker='s', linewidth=2, label=f'INSH {location}', 
            color=colors_dose[i+2], linestyle='--')

ax4.set_title('Plant Height Metrics by Location', fontweight='bold', fontsize=10)
ax4.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax4.set_ylabel('Height (cm)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Subplot 5: Bubble plot for branching
ax5 = axes[1, 1]
sizes = (df['TNOD'] - df['TNOD'].min()) / (df['TNOD'].max() - df['TNOD'].min()) * 200 + 20
scatter = ax5.scatter(df['Dose'], df['BRA'], s=sizes, c=df['FNOD'], 
                     alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(df['Dose'], df['BRA'], 1)
p = np.poly1d(z)
ax5.plot(sorted(df['Dose'].unique()), p(sorted(df['Dose'].unique())), 
         color='red', linewidth=2, linestyle='--')

ax5.set_title('Branches vs Dose\n(Size=Total Nodes, Color=Fertile Nodes)', fontweight='bold', fontsize=10)
ax5.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax5.set_ylabel('Number of Branches')
plt.colorbar(scatter, ax=ax5, label='Fertile Nodes')

# Subplot 6: Violin plots for pods
ax6 = axes[1, 2]
doses_unique = sorted(df['Dose'].unique())
pod_data = [df[df['Dose'] == dose]['PODS'].values for dose in doses_unique]

bp = ax6.boxplot(pod_data, positions=doses_unique, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#8DA0CB')
    patch.set_alpha(0.7)

ax6.set_title('Pods per Plant Distribution', fontweight='bold', fontsize=10)
ax6.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax6.set_ylabel('Pods per Plant')

# Row 3 - Yield and Quality Analysis

# Subplot 7: Correlation heatmap
ax7 = axes[2, 0]
yield_components = ['Dose', 'YIELD', 'GRAINS', 'WEIGHT']
corr_matrix = df[yield_components].corr()

im = ax7.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
for i in range(len(yield_components)):
    for j in range(len(yield_components)):
        ax7.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha="center", va="center", color="black", fontweight='bold')

ax7.set_title('Yield Components Correlation', fontweight='bold', fontsize=10)
ax7.set_xticks(range(len(yield_components)))
ax7.set_yticks(range(len(yield_components)))
ax7.set_xticklabels(yield_components, rotation=45)
ax7.set_yticklabels(yield_components)

# Subplot 8: Slope chart for yield changes
ax8 = axes[2, 1]
for i, soil in enumerate(soil_types):
    soil_data = df[df['Field'] == soil]
    yield_means = soil_data.groupby('Dose')['YIELD'].mean()
    
    ax8.plot(yield_means.index, yield_means.values, 
            marker='o', linewidth=2, label=soil, color=colors_soil[i])

ax8.set_title('Yield Response by Soil Type', fontweight='bold', fontsize=10)
ax8.set_xlabel('Limestone Dose (Mg ha⁻¹)')
ax8.set_ylabel('Yield (kg ha⁻¹)')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Subplot 9: Performance comparison at optimal dose
ax9 = axes[2, 2]
optimal_dose = 6
metrics = ['YIELD', 'PLH', 'BRA']
soil_performance = []

for soil in soil_types:
    soil_data = df[(df['Field'] == soil) & (df['Dose'] == optimal_dose)]
    performance = [
        soil_data['YIELD'].mean(),
        soil_data['PLH'].mean(),
        soil_data['BRA'].mean()
    ]
    soil_performance.append(performance)

x_pos = np.arange(len(metrics))
width = 0.2

for i, soil in enumerate(soil_types):
    ax9.bar(x_pos + i*width, soil_performance[i], width, 
           label=soil, color=colors_soil[i], alpha=0.8)

ax9.set_title('Performance at Optimal Dose (6 Mg ha⁻¹)', fontweight='bold', fontsize=10)
ax9.set_xlabel('Metrics')
ax9.set_ylabel('Values')
ax9.set_xticks(x_pos + width*1.5)
ax9.set_xticklabels(metrics)
ax9.legend(fontsize=8)

# Overall layout adjustment
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.35, wspace=0.35)

# Save the plot
plt.savefig('limestone_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()