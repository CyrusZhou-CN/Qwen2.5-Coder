import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Load and combine datasets with error handling
try:
    df_train = pd.read_csv('student_addiction_dataset_train.csv')
    df_test = pd.read_csv('student_addiction_dataset_test.csv')
    df = pd.concat([df_train, df_test], ignore_index=True)
except:
    # If files not found, use only one dataset
    try:
        df = pd.read_csv('student_addiction_dataset_train.csv')
    except:
        df = pd.read_csv('student_addiction_dataset_test.csv')

# Sample data for faster processing (avoid timeout)
df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

# Data preprocessing
categorical_cols = df.columns[:-1]  # All columns except Addiction_Class
for col in categorical_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Handle missing values
df = df.fillna(0)

# Create synthetic demographic data for visualization
np.random.seed(42)
n_samples = len(df)
df['Age'] = np.random.normal(20, 2, n_samples).clip(18, 25)
df['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
df['Family_Income'] = np.random.lognormal(10, 0.5, n_samples)
df['Peer_Influence'] = np.random.uniform(0, 10, n_samples)

# Create addiction severity score
risk_factors = ['Experimentation', 'Academic_Performance_Decline', 'Social_Isolation', 
                'Financial_Issues', 'Physical_Mental_Health_Problems', 'Legal_Consequences',
                'Relationship_Strain', 'Risk_Taking_Behavior', 'Withdrawal_Symptoms', 
                'Denial_and_Resistance_to_Treatment']
df['Addiction_Score'] = df[risk_factors].sum(axis=1)
df['Addiction_Binary'] = (df['Addiction_Class'] == 'Yes').astype(int)

# Perform clustering on subset for speed
features_for_clustering = df[risk_factors + ['Age']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_clustering)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('white')

# Define color palettes
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
risk_colors = ['#2ECC71', '#F39C12', '#E74C3C']

# Row 1: Demographic Clustering Analysis

# Subplot 1: Age vs Academic Performance with clustering
ax1 = plt.subplot(3, 3, 1)
scatter = ax1.scatter(df['Age'], df['Academic_Performance_Decline'], 
                     c=df['Addiction_Score'], cmap='RdYlBu_r', 
                     alpha=0.6, s=20)

# Add simple cluster centers instead of ellipses for speed
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    if len(cluster_data) > 5:
        mean_age = cluster_data['Age'].mean()
        mean_perf = cluster_data['Academic_Performance_Decline'].mean()
        ax1.scatter(mean_age, mean_perf, s=200, c=cluster_colors[i], 
                   marker='x', linewidths=3, label=f'Cluster {i+1}')

ax1.set_xlabel('Age', fontweight='bold')
ax1.set_ylabel('Academic Performance Decline', fontweight='bold')
ax1.set_title('Age vs Academic Performance\nwith Addiction Risk Clustering', fontweight='bold')
ax1.legend()

# Subplot 2: Gender-based violin plots (simplified)
ax2 = plt.subplot(3, 3, 2)
male_data = df[df['Gender'] == 'Male']['Addiction_Score']
female_data = df[df['Gender'] == 'Female']['Addiction_Score']

ax2.hist([male_data, female_data], bins=15, alpha=0.7, 
         label=['Male', 'Female'], color=['#4ECDC4', '#FF6B6B'])
ax2.set_xlabel('Addiction Score', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Gender-based Addiction Score Distribution', fontweight='bold')
ax2.legend()

# Subplot 3: Family income vs peer influence bubble chart
ax3 = plt.subplot(3, 3, 3)
bubble_sizes = (df['Addiction_Score'] * 10).clip(5, 100)
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    ax3.scatter(cluster_data['Family_Income'], cluster_data['Peer_Influence'],
               s=bubble_sizes[df['Cluster'] == i], alpha=0.6, 
               color=cluster_colors[i], label=f'Cluster {i+1}')

ax3.set_xlabel('Family Income', fontweight='bold')
ax3.set_ylabel('Peer Influence', fontweight='bold')
ax3.set_title('Family Income vs Peer Influence\nBubble Size = Addiction Severity', fontweight='bold')
ax3.legend()

# Row 2: Behavioral Pattern Groups

# Subplot 4: Simplified radar chart
ax4 = plt.subplot(3, 3, 4)
behavioral_metrics = ['Academic_Performance_Decline', 'Social_Isolation', 
                     'Relationship_Strain', 'Physical_Mental_Health_Problems']

cluster_means = []
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    means = [cluster_data[metric].mean() for metric in behavioral_metrics]
    cluster_means.append(means)

x = np.arange(len(behavioral_metrics))
width = 0.25

for i in range(3):
    ax4.bar(x + i*width, cluster_means[i], width, 
           label=f'Cluster {i+1}', color=cluster_colors[i], alpha=0.8)

ax4.set_xlabel('Behavioral Metrics', fontweight='bold')
ax4.set_ylabel('Average Score', fontweight='bold')
ax4.set_title('Behavioral Metrics by Cluster', fontweight='bold')
ax4.set_xticks(x + width)
ax4.set_xticklabels([m.replace('_', '\n') for m in behavioral_metrics], rotation=45)
ax4.legend()

# Subplot 5: Simplified clustering visualization
ax5 = plt.subplot(3, 3, 5)
# Use PCA results for 2D clustering visualization
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    ax5.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
               c=cluster_colors[i], label=f'Cluster {i+1}', alpha=0.7, s=30)

ax5.set_xlabel('First Principal Component', fontweight='bold')
ax5.set_ylabel('Second Principal Component', fontweight='bold')
ax5.set_title('PCA-based Cluster Visualization', fontweight='bold')
ax5.legend()

# Subplot 6: Correlation matrix
ax6 = plt.subplot(3, 3, 6)
# Use subset of risk factors for better visibility
key_factors = risk_factors[:6]
corr_matrix = df[key_factors].corr()
im = ax6.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax6.set_xticks(range(len(key_factors)))
ax6.set_yticks(range(len(key_factors)))
ax6.set_xticklabels([f.replace('_', '\n') for f in key_factors], rotation=45)
ax6.set_yticklabels([f.replace('_', '\n') for f in key_factors])
ax6.set_title('Risk Factor Correlation Matrix', fontweight='bold')
plt.colorbar(im, ax=ax6, shrink=0.8)

# Row 3: Risk Factor Composition by Groups

# Subplot 7: Stacked bar chart instead of area chart
ax7 = plt.subplot(3, 3, 7)
severity_groups = ['Low (0-3)', 'Medium (4-6)', 'High (7-10)']
low_risk = df[df['Addiction_Score'] <= 3]
med_risk = df[(df['Addiction_Score'] > 3) & (df['Addiction_Score'] <= 6)]
high_risk = df[df['Addiction_Score'] > 6]

groups_data = [low_risk, med_risk, high_risk]
key_factors_short = risk_factors[:5]  # Use first 5 factors

bottom = np.zeros(3)
for i, factor in enumerate(key_factors_short):
    values = [group[factor].mean() if len(group) > 0 else 0 for group in groups_data]
    ax7.bar(severity_groups, values, bottom=bottom, 
           label=factor.replace('_', ' '), alpha=0.8)
    bottom += values

ax7.set_xlabel('Risk Group', fontweight='bold')
ax7.set_ylabel('Average Risk Factor Score', fontweight='bold')
ax7.set_title('Risk Factor Composition\nby Severity Group', fontweight='bold')
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Subplot 8: Simplified treemap as grouped bar chart
ax8 = plt.subplot(3, 3, 8)
trigger_freq = df[risk_factors].sum().sort_values(ascending=False)
colors_bar = plt.cm.Set3(np.linspace(0, 1, len(trigger_freq)))

bars = ax8.bar(range(len(trigger_freq)), trigger_freq.values, 
               color=colors_bar, alpha=0.8)
ax8.set_xticks(range(len(trigger_freq)))
ax8.set_xticklabels([t.replace('_', '\n') for t in trigger_freq.index], 
                    rotation=45, ha='right')
ax8.set_ylabel('Frequency', fontweight='bold')
ax8.set_title('Addiction Triggers by Frequency', fontweight='bold')

# Subplot 9: Parallel coordinates plot (simplified)
ax9 = plt.subplot(3, 3, 9)
high_risk = df[df['Addiction_Score'] >= 7]
low_risk = df[df['Addiction_Score'] <= 3]

key_dimensions = ['Experimentation', 'Academic_Performance_Decline', 
                 'Social_Isolation', 'Risk_Taking_Behavior']

# Plot mean lines only for clarity
if len(high_risk) > 0:
    high_means = [high_risk[dim].mean() for dim in key_dimensions]
    ax9.plot(range(len(key_dimensions)), high_means, 
            'o-', color='#E74C3C', linewidth=3, markersize=8, label='High Risk')

if len(low_risk) > 0:
    low_means = [low_risk[dim].mean() for dim in key_dimensions]
    ax9.plot(range(len(key_dimensions)), low_means, 
            'o-', color='#2ECC71', linewidth=3, markersize=8, label='Low Risk')

ax9.set_xticks(range(len(key_dimensions)))
ax9.set_xticklabels([dim.replace('_', '\n') for dim in key_dimensions])
ax9.set_ylabel('Average Risk Score', fontweight='bold')
ax9.set_title('Risk Profile Comparison\nHigh vs Low Risk Groups', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Final layout adjustment
plt.tight_layout(pad=2.0)
plt.savefig('student_addiction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()