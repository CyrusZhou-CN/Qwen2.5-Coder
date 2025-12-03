import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('hypertension_dataset.csv')

# Data preprocessing
df_clean = df.copy()
# Fill missing values
df_clean['Medication'] = df_clean['Medication'].fillna('None')
df_clean = df_clean.dropna()

# Convert categorical variables
df_clean['Has_Hypertension_num'] = (df_clean['Has_Hypertension'] == 'Yes').astype(int)

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('white')

# Define colors
hyp_colors = ['#3498db', '#e74c3c']  # Blue for No, Red for Yes

# 1. Top-left: Scatter plot (Age vs BMI)
ax1 = plt.subplot(3, 3, 1)
for i, hyp_status in enumerate(['No', 'Yes']):
    mask = df_clean['Has_Hypertension'] == hyp_status
    plt.scatter(df_clean[mask]['Age'], df_clean[mask]['BMI'], 
               alpha=0.6, c=hyp_colors[i], label=f'Hypertension: {hyp_status}', s=25)
plt.xlabel('Age (years)', fontweight='bold')
plt.ylabel('BMI', fontweight='bold')
plt.title('Age vs BMI by Hypertension Status', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Top-center: Box plot (Stress scores by exercise levels)
ax2 = plt.subplot(3, 3, 2)
exercise_levels = sorted(df_clean['Exercise_Level'].unique())
stress_data = []
labels = []

for ex_level in exercise_levels:
    for hyp_status in ['No', 'Yes']:
        mask = (df_clean['Exercise_Level'] == ex_level) & (df_clean['Has_Hypertension'] == hyp_status)
        stress_data.append(df_clean[mask]['Stress_Score'].values)
        labels.append(f'{ex_level}\n{hyp_status}')

bp = plt.boxplot(stress_data, labels=labels, patch_artist=True)
colors = [hyp_colors[0], hyp_colors[1]] * len(exercise_levels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.xticks(rotation=45)
plt.ylabel('Stress Score', fontweight='bold')
plt.title('Stress Scores by Exercise Level\nand Hypertension Status', fontweight='bold', fontsize=11)

# 3. Top-right: Stacked bar chart (Hypertension by BP history)
ax3 = plt.subplot(3, 3, 3)
bp_history_order = ['Normal', 'Prehypertension', 'Hypertension']
bp_hyp_counts = df_clean.groupby(['BP_History', 'Has_Hypertension']).size().unstack(fill_value=0)

x_pos = np.arange(len(bp_history_order))
width = 0.6

if 'No' in bp_hyp_counts.columns:
    no_counts = [bp_hyp_counts.loc[bp, 'No'] if bp in bp_hyp_counts.index else 0 for bp in bp_history_order]
else:
    no_counts = [0] * len(bp_history_order)

if 'Yes' in bp_hyp_counts.columns:
    yes_counts = [bp_hyp_counts.loc[bp, 'Yes'] if bp in bp_hyp_counts.index else 0 for bp in bp_history_order]
else:
    yes_counts = [0] * len(bp_history_order)

plt.bar(x_pos, no_counts, width, label='No Hypertension', color=hyp_colors[0], alpha=0.8)
plt.bar(x_pos, yes_counts, width, bottom=no_counts, label='Has Hypertension', color=hyp_colors[1], alpha=0.8)

plt.xlabel('BP History', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Hypertension Distribution by BP History', fontweight='bold', fontsize=11)
plt.xticks(x_pos, bp_history_order, rotation=45)
plt.legend()

# 4. Middle-left: Radar chart
ax4 = plt.subplot(3, 3, 4, projection='polar')
numerical_features = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']

# Calculate means for each group
hyp_means = df_clean[df_clean['Has_Hypertension'] == 'Yes'][numerical_features].mean()
no_hyp_means = df_clean[df_clean['Has_Hypertension'] == 'No'][numerical_features].mean()

# Normalize values to 0-1 scale
min_vals = df_clean[numerical_features].min()
max_vals = df_clean[numerical_features].max()
hyp_norm = (hyp_means - min_vals) / (max_vals - min_vals)
no_hyp_norm = (no_hyp_means - min_vals) / (max_vals - min_vals)

angles = np.linspace(0, 2*np.pi, len(numerical_features), endpoint=False).tolist()
angles += angles[:1]

hyp_values = hyp_norm.tolist() + [hyp_norm.iloc[0]]
no_hyp_values = no_hyp_norm.tolist() + [no_hyp_norm.iloc[0]]

plt.plot(angles, hyp_values, 'o-', linewidth=2, label='Has Hypertension', color=hyp_colors[1])
plt.fill(angles, hyp_values, alpha=0.25, color=hyp_colors[1])
plt.plot(angles, no_hyp_values, 'o-', linewidth=2, label='No Hypertension', color=hyp_colors[0])
plt.fill(angles, no_hyp_values, alpha=0.25, color=hyp_colors[0])

plt.xticks(angles[:-1], numerical_features)
plt.title('Mean Values by Hypertension Status', fontweight='bold', fontsize=11, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

# 5. Middle-center: Correlation heatmap
ax5 = plt.subplot(3, 3, 5)
corr_matrix = df_clean[numerical_features].corr()

im = plt.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, shrink=0.8)

# Add correlation values
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold', fontsize=9)

plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.index)
plt.title('Correlation Matrix', fontweight='bold', fontsize=11)

# 6. Middle-right: Medication usage by family history
ax6 = plt.subplot(3, 3, 6)
family_history_groups = sorted(df_clean['Family_History'].unique())

# Calculate hypertension rates by family history
hyp_rates = []
for fh in family_history_groups:
    mask = df_clean['Family_History'] == fh
    rate = df_clean[mask]['Has_Hypertension_num'].mean() * 100
    hyp_rates.append(rate)

x_pos = np.arange(len(family_history_groups))
plt.bar(x_pos, hyp_rates, color=['#2ecc71', '#e67e22'], alpha=0.8)

plt.xlabel('Family History', fontweight='bold')
plt.ylabel('Hypertension Rate (%)', fontweight='bold')
plt.title('Hypertension Rates by Family History', fontweight='bold', fontsize=11)
plt.xticks(x_pos, family_history_groups)

# Add value labels on bars
for i, rate in enumerate(hyp_rates):
    plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')

# 7. Bottom-left: Parallel coordinates (simplified)
ax7 = plt.subplot(3, 3, 7)

# Sample data for parallel coordinates
sample_size = min(100, len(df_clean))
sample_indices = np.random.choice(len(df_clean), sample_size, replace=False)

features_for_parallel = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_clean[features_for_parallel])

for idx in sample_indices:
    color = hyp_colors[1] if df_clean.iloc[idx]['Has_Hypertension'] == 'Yes' else hyp_colors[0]
    alpha = 0.8 if df_clean.iloc[idx]['Has_Hypertension'] == 'Yes' else 0.4
    plt.plot(range(len(features_for_parallel)), normalized_data[idx], 
            color=color, alpha=alpha, linewidth=1.5)

plt.xticks(range(len(features_for_parallel)), features_for_parallel, rotation=45)
plt.ylabel('Normalized Values', fontweight='bold')
plt.title('Parallel Coordinates Plot\n(Sample of 100 patients)', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

# 8. Bottom-center: Exercise level distribution
ax8 = plt.subplot(3, 3, 8)
exercise_hyp = df_clean.groupby(['Exercise_Level', 'Has_Hypertension']).size().unstack(fill_value=0)

exercise_levels = sorted(df_clean['Exercise_Level'].unique())
x_pos = np.arange(len(exercise_levels))
width = 0.35

no_counts = [exercise_hyp.loc[ex, 'No'] if ex in exercise_hyp.index and 'No' in exercise_hyp.columns else 0 for ex in exercise_levels]
yes_counts = [exercise_hyp.loc[ex, 'Yes'] if ex in exercise_hyp.index and 'Yes' in exercise_hyp.columns else 0 for ex in exercise_levels]

plt.bar(x_pos - width/2, no_counts, width, label='No Hypertension', color=hyp_colors[0], alpha=0.8)
plt.bar(x_pos + width/2, yes_counts, width, label='Has Hypertension', color=hyp_colors[1], alpha=0.8)

plt.xlabel('Exercise Level', fontweight='bold')
plt.ylabel('Patient Count', fontweight='bold')
plt.title('Patient Distribution by Exercise Level', fontweight='bold', fontsize=11)
plt.xticks(x_pos, exercise_levels)
plt.legend()

# 9. Bottom-right: PCA scatter plot
ax9 = plt.subplot(3, 3, 9)

# Perform PCA on numerical features
pca_features = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[pca_features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Sample data for better performance
sample_size = min(500, len(df_clean))
sample_indices = np.random.choice(len(df_clean), sample_size, replace=False)

for i, hyp_status in enumerate(['No', 'Yes']):
    mask = df_clean.iloc[sample_indices]['Has_Hypertension'] == hyp_status
    sample_mask_indices = sample_indices[mask]
    
    plt.scatter(pca_result[sample_mask_indices, 0], pca_result[sample_mask_indices, 1], 
               c=hyp_colors[i], alpha=0.6, s=30, label=f'Hypertension: {hyp_status}')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
plt.title('PCA Analysis\n(Sample of 500 patients)', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=1.5)
plt.savefig('hypertension_analysis_grid.png', dpi=300, bbox_inches='tight')
plt.show()