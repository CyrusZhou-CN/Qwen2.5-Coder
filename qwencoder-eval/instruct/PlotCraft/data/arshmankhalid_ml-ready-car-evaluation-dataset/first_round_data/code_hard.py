import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Car_Evaluation_Label_encode.csv')

# Create figure with optimized layout
plt.style.use('default')
fig = plt.figure(figsize=(18, 14), facecolor='white')
fig.patch.set_facecolor('white')

# Define color palettes
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F7B0F']
class_colors = {0: '#2E86AB', 1: '#A23B72', 2: '#F18F01', 3: '#C73E1D'}

# 1. Top-left: Stacked bar chart with overlaid line plot
ax1 = plt.subplot(3, 3, 1, facecolor='white')
buying_class = df.groupby(['Buying_Price', 'class']).size().unstack(fill_value=0)
buying_class.plot(kind='bar', stacked=True, ax=ax1, color=[class_colors.get(i, '#666666') for i in buying_class.columns])
acceptance_rates = df.groupby('Buying_Price')['class'].apply(lambda x: (x > 0).mean())
ax1_twin = ax1.twinx()
ax1_twin.plot(range(len(acceptance_rates)), acceptance_rates.values, 'ro-', linewidth=2, markersize=4)
ax1.set_title('Buying Price vs Class Distribution', fontweight='bold', fontsize=10)
ax1.set_xlabel('Buying Price Level', fontsize=8)
ax1.set_ylabel('Count', fontsize=8)
ax1_twin.set_ylabel('Acceptance Rate', fontsize=8)
ax1.tick_params(axis='x', rotation=45, labelsize=7)
ax1.legend(title='Class', fontsize=6, title_fontsize=7)

# 2. Top-middle: Grouped violin plots
ax2 = plt.subplot(3, 3, 2, facecolor='white')
# Simplified violin plot to avoid timeout
safety_data = []
for safety in sorted(df['safety'].unique()):
    safety_subset = df[df['safety'] == safety]['class']
    safety_data.append(safety_subset.values)

parts = ax2.violinplot(safety_data, positions=range(len(safety_data)), showmeans=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_alpha(0.7)

ax2.set_xticks(range(len(safety_data)))
ax2.set_xticklabels([f'Safety {i}' for i in sorted(df['safety'].unique())], fontsize=8)
ax2.set_title('Safety Level Distribution', fontweight='bold', fontsize=10)
ax2.set_ylabel('Class', fontsize=8)

# 3. Top-right: Correlation heatmap
ax3 = plt.subplot(3, 3, 3, facecolor='white')
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0, ax=ax3,
           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f',
           annot_kws={'size': 6})
ax3.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=10)
ax3.tick_params(labelsize=7)

# 4. Middle-left: Simplified parallel coordinates
ax4 = plt.subplot(3, 3, 4, facecolor='white')
features = ['Buying_Price', 'Maintenance_Price', 'No_of_Doors', 'persons', 'lug_boot', 'safety']
normalized_df = df[features].copy()
for col in features:
    normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Sample data to avoid timeout
sample_size = min(200, len(df))
sample_indices = np.random.choice(len(df), sample_size, replace=False)

for class_val in df['class'].unique():
    class_mask = df['class'] == class_val
    class_sample = np.intersect1d(sample_indices, np.where(class_mask)[0])
    
    if len(class_sample) > 0:
        class_data = normalized_df.iloc[class_sample]
        for i in range(len(class_data)):
            ax4.plot(range(len(features)), class_data.iloc[i], 
                    color=class_colors[class_val], alpha=0.3, linewidth=0.5)

ax4.set_xticks(range(len(features)))
ax4.set_xticklabels([f.replace('_', '\n') for f in features], rotation=0, fontsize=7)
ax4.set_title('Parallel Coordinates (Sample)', fontweight='bold', fontsize=10)
ax4.set_ylabel('Normalized Values', fontsize=8)

# 5. Middle-middle: Treemap visualization
ax5 = plt.subplot(3, 3, 5, facecolor='white')
doors_persons = df.groupby(['No_of_Doors', 'persons']).size().reset_index(name='count')

# Create simple treemap
x_pos, y_pos = 0, 0
colors_treemap = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

for i, row in doors_persons.iterrows():
    doors, persons, count = row['No_of_Doors'], row['persons'], row['count']
    width = np.sqrt(count) * 0.02
    height = width
    
    rect = patches.Rectangle((x_pos, y_pos), width, height, 
                           facecolor=colors_treemap[i % len(colors_treemap)], 
                           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.add_patch(rect)
    
    ax5.text(x_pos + width/2, y_pos + height/2, f'D{doors}\nP{persons}', 
            ha='center', va='center', fontsize=6, fontweight='bold')
    
    x_pos += width + 0.05
    if x_pos > 1.2:
        x_pos = 0
        y_pos += height + 0.05

ax5.set_xlim(-0.05, 1.5)
ax5.set_ylim(-0.05, 1)
ax5.set_title('Doors-Persons Composition', fontweight='bold', fontsize=10)
ax5.axis('off')

# 6. Middle-right: Radar chart
ax6 = plt.subplot(3, 3, 6, facecolor='white', projection='polar')
lug_boot_means = df.groupby('lug_boot')[features].mean()

angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

for i, lug_val in enumerate(sorted(df['lug_boot'].unique())):
    means = lug_boot_means.loc[lug_val].values.tolist()
    means += means[:1]
    
    ax6.plot(angles, means, 'o-', linewidth=2, label=f'Luggage {lug_val}', 
            color=colors[i % len(colors)], markersize=4)
    ax6.fill(angles, means, alpha=0.25, color=colors[i % len(colors)])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=7)
ax6.set_title('Feature Scores by Luggage Boot', fontweight='bold', fontsize=10, pad=15)
ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=6)

# 7. Bottom-left: PCA scatter plot
ax7 = plt.subplot(3, 3, 7, facecolor='white')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for class_val in df['class'].unique():
    mask = df['class'] == class_val
    ax7.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=class_colors[class_val], label=f'Class {class_val}', 
               alpha=0.6, s=20)

ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=8)
ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=8)
ax7.set_title('PCA Feature Space', fontweight='bold', fontsize=10)
ax7.legend(fontsize=6)
ax7.grid(True, alpha=0.3)

# 8. Bottom-middle: Flow diagram
ax8 = plt.subplot(3, 3, 8, facecolor='white')
safety_class = df.groupby(['safety', 'class']).size().unstack(fill_value=0)

# Simplified flow visualization
y_positions = np.linspace(0.2, 0.8, len(safety_class.index))
class_positions = np.linspace(0.2, 0.8, len(safety_class.columns))

# Draw connections
for i, safety_level in enumerate(safety_class.index):
    for j, class_val in enumerate(safety_class.columns):
        flow_size = safety_class.loc[safety_level, class_val]
        if flow_size > 50:  # Only show significant flows
            x1, y1 = 0.2, y_positions[i]
            x2, y2 = 0.8, class_positions[j]
            
            ax8.plot([x1, x2], [y1, y2], color=class_colors[class_val], 
                    linewidth=flow_size/100, alpha=0.6)

# Add labels
for i, safety in enumerate(safety_class.index):
    ax8.text(0.1, y_positions[i], f'Safety {safety}', ha='center', va='center', 
            fontsize=8, fontweight='bold')

for j, class_val in enumerate(safety_class.columns):
    ax8.text(0.9, class_positions[j], f'Class {class_val}', ha='center', va='center', 
            fontsize=8, fontweight='bold')

ax8.set_xlim(0, 1)
ax8.set_ylim(0, 1)
ax8.set_title('Safety to Class Flow', fontweight='bold', fontsize=10)
ax8.axis('off')

# 9. Bottom-right: Network-style visualization
ax9 = plt.subplot(3, 3, 9, facecolor='white')

# Create a simplified network visualization
class_counts = df['class'].value_counts().sort_index()
safety_counts = df['safety'].value_counts().sort_index()

# Draw class nodes
class_positions = [(0.7, 0.2), (0.7, 0.4), (0.7, 0.6), (0.7, 0.8)]
for i, (class_val, count) in enumerate(class_counts.items()):
    if i < len(class_positions):
        x, y = class_positions[i]
        circle = plt.Circle((x, y), count/2000, color=class_colors[class_val], alpha=0.7)
        ax9.add_patch(circle)
        ax9.text(x, y, f'C{class_val}', ha='center', va='center', fontweight='bold', fontsize=8)

# Draw safety nodes
safety_positions = [(0.3, 0.3), (0.3, 0.5), (0.3, 0.7)]
for i, (safety_val, count) in enumerate(safety_counts.items()):
    if i < len(safety_positions):
        x, y = safety_positions[i]
        circle = plt.Circle((x, y), count/2000, color=colors[i], alpha=0.7)
        ax9.add_patch(circle)
        ax9.text(x, y, f'S{safety_val}', ha='center', va='center', fontweight='bold', fontsize=8)

# Draw connections
for i, safety_val in enumerate(safety_counts.index[:3]):
    for j, class_val in enumerate(class_counts.index[:4]):
        connection_strength = len(df[(df['safety'] == safety_val) & (df['class'] == class_val)])
        if connection_strength > 50:
            x1, y1 = safety_positions[i]
            x2, y2 = class_positions[j]
            ax9.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=connection_strength/100)

ax9.set_xlim(0, 1)
ax9.set_ylim(0, 1)
ax9.set_title('Feature-Class Network', fontweight='bold', fontsize=10)
ax9.axis('off')

# Adjust layout and save
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.35, wspace=0.35)
plt.savefig('car_evaluation_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()