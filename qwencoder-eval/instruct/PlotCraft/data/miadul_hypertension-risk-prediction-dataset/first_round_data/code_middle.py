import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df = pd.read_csv('hypertension_dataset.csv')

# Data preprocessing
# Convert Has_Hypertension to binary for easier plotting
df['Hypertension_Binary'] = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

# Select numerical variables for correlation analysis
numerical_vars = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
df_numerical = df[numerical_vars].dropna()

# Create 2x2 subplot grid with white background
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Top-left: Correlation heatmap
correlation_matrix = df_numerical.corr()
im = axes[0, 0].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[0, 0].set_xticks(range(len(numerical_vars)))
axes[0, 0].set_yticks(range(len(numerical_vars)))
axes[0, 0].set_xticklabels(numerical_vars, rotation=45, ha='right')
axes[0, 0].set_yticklabels(numerical_vars)
axes[0, 0].set_title('Correlation Heatmap of Numerical Variables', fontweight='bold', fontsize=12)

# Add correlation values to heatmap
for i in range(len(numerical_vars)):
    for j in range(len(numerical_vars)):
        text = axes[0, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')

# Add colorbar for heatmap
cbar = plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Top-right: Scatter plot matrix (simplified version)
# Create a subset for cleaner visualization
vars_subset = ['Age', 'BMI', 'Salt_Intake', 'Stress_Score']
df_clean = df[vars_subset + ['Has_Hypertension']].dropna()

# Create scatter plot matrix manually
colors = {'Yes': '#e74c3c', 'No': '#3498db'}
for i, var1 in enumerate(vars_subset[:2]):  # Simplified to 2x2 matrix
    for j, var2 in enumerate(vars_subset[:2]):
        if i == 0 and j == 0:  # Top-left of matrix
            for hyp_status in ['Yes', 'No']:
                mask = df_clean['Has_Hypertension'] == hyp_status
                axes[0, 1].scatter(df_clean.loc[mask, var2], df_clean.loc[mask, var1], 
                                 c=colors[hyp_status], alpha=0.6, s=20, label=f'Hypertension: {hyp_status}')
            axes[0, 1].set_xlabel('Age', fontweight='bold')
            axes[0, 1].set_ylabel('BMI', fontweight='bold')
            axes[0, 1].set_title('Scatter Plot: BMI vs Age by Hypertension Status', fontweight='bold', fontsize=12)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

# Bottom-left: BMI vs Salt_Intake with stress score sizing
df_scatter = df[['BMI', 'Salt_Intake', 'Stress_Score', 'Has_Hypertension']].dropna()
for hyp_status in ['Yes', 'No']:
    mask = df_scatter['Has_Hypertension'] == hyp_status
    sizes = df_scatter.loc[mask, 'Stress_Score'] * 10 + 20  # Scale stress score for sizing
    axes[1, 0].scatter(df_scatter.loc[mask, 'Salt_Intake'], df_scatter.loc[mask, 'BMI'], 
                      c=colors[hyp_status], s=sizes, alpha=0.6, label=f'Hypertension: {hyp_status}')

axes[1, 0].set_xlabel('Salt Intake (g/day)', fontweight='bold')
axes[1, 0].set_ylabel('BMI', fontweight='bold')
axes[1, 0].set_title('BMI vs Salt Intake\n(Point size = Stress Score)', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bottom-right: Box plot of stress scores by BP_History and hypertension status
df_box = df[['Stress_Score', 'BP_History', 'Has_Hypertension']].dropna()

# Create grouped box plot
bp_categories = df_box['BP_History'].unique()
bp_categories = [cat for cat in bp_categories if pd.notna(cat)]  # Remove NaN values

positions = []
box_data = []
labels = []
colors_box = []

pos = 1
for bp_cat in sorted(bp_categories):
    for hyp_status in ['No', 'Yes']:
        mask = (df_box['BP_History'] == bp_cat) & (df_box['Has_Hypertension'] == hyp_status)
        data = df_box.loc[mask, 'Stress_Score']
        if len(data) > 0:
            box_data.append(data)
            positions.append(pos)
            labels.append(f'{bp_cat}\n{hyp_status}')
            colors_box.append('#3498db' if hyp_status == 'No' else '#e74c3c')
            pos += 1
        pos += 0.5  # Add space between BP categories

# Create box plot
bp = axes[1, 1].boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

# Color the boxes
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1, 1].set_xticks(positions)
axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
axes[1, 1].set_ylabel('Stress Score', fontweight='bold')
axes[1, 1].set_title('Stress Score Distribution by BP History\nand Hypertension Status', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Create custom legend for box plot
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='No Hypertension'),
                  Patch(facecolor='#e74c3c', alpha=0.7, label='Has Hypertension')]
axes[1, 1].legend(handles=legend_elements, loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.show()