import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Load all datasets with error handling
datasets = {}
dataset_files = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
dataset_names = ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']

for name, file in zip(dataset_names, dataset_files):
    try:
        datasets[name] = pd.read_csv(file)
        print(f"Loaded {name}: {datasets[name].shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Create figure with 3x2 subplot grid
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# Define consistent color schemes
fraud_colors = ['#2E86AB', '#F24236']  # Blue for non-fraud, Red for fraud
employment_colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

# Top row: Base, Variant I, Variant II (simplified version)
top_variants = ['Base', 'Variant I', 'Variant II']

for i, variant in enumerate(top_variants):
    if variant not in datasets:
        continue
        
    ax = fig.add_subplot(gs[0, i])
    df = datasets[variant]
    
    # Sample data for performance (use smaller sample)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Clean data
    df_sample = df_sample.dropna(subset=['income', 'credit_risk_score', 'fraud_bool'])
    
    # Main scatter plot: credit_risk_score vs income colored by fraud_bool
    fraud_mask = df_sample['fraud_bool'] == 1
    non_fraud = df_sample[~fraud_mask]
    fraud = df_sample[fraud_mask]
    
    # Plot scatter points
    if len(non_fraud) > 0:
        ax.scatter(non_fraud['income'], non_fraud['credit_risk_score'], 
                  c=fraud_colors[0], alpha=0.5, s=8, label='Non-Fraud')
    
    if len(fraud) > 0:
        ax.scatter(fraud['income'], fraud['credit_risk_score'], 
                  c=fraud_colors[1], alpha=0.8, s=12, label='Fraud')
    
    # Add simple box plot overlay for employment status
    if 'employment_status' in df_sample.columns:
        employment_stats = df_sample['employment_status'].value_counts().head(3).index.tolist()
        
        # Create box plot data
        box_data = []
        box_labels = []
        for emp_status in employment_stats:
            emp_data = df_sample[df_sample['employment_status'] == emp_status]['credit_risk_score']
            if len(emp_data) > 10:  # Only include if sufficient data
                box_data.append(emp_data)
                box_labels.append(emp_status)
        
        if box_data:
            # Create inset for box plots
            ax_inset = fig.add_axes([ax.get_position().x0 + 0.02, ax.get_position().y1 - 0.08, 
                                   0.12, 0.06])
            bp = ax_inset.boxplot(box_data, labels=box_labels, patch_artist=True)
            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(employment_colors[j % len(employment_colors)])
                patch.set_alpha(0.7)
            ax_inset.set_title('Employment Status', fontsize=8)
            ax_inset.tick_params(axis='x', labelsize=6)
            ax_inset.tick_params(axis='y', labelsize=6)
    
    # Styling
    ax.set_xlabel('Income', fontweight='bold')
    ax.set_ylabel('Credit Risk Score', fontweight='bold')
    ax.set_title(f'{variant}: Credit Risk vs Income', fontweight='bold', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

# Bottom row: Variant III, Variant IV, Variant V (simplified version)
bottom_variants = ['Variant III', 'Variant IV', 'Variant V']

for i, variant in enumerate(bottom_variants):
    if variant not in datasets:
        continue
        
    ax = fig.add_subplot(gs[1, i])
    df = datasets[variant]
    
    # Sample data for performance
    sample_size = min(3000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Clean data
    required_cols = ['velocity_6h', 'velocity_24h', 'fraud_bool']
    df_sample = df_sample.dropna(subset=required_cols)
    
    # Create hexbin plot with reduced gridsize
    if len(df_sample) > 0:
        hb = ax.hexbin(df_sample['velocity_6h'], df_sample['velocity_24h'], 
                       gridsize=20, cmap='Blues', alpha=0.6, mincnt=1)
        
        # Overlay fraud cases
        fraud_cases = df_sample[df_sample['fraud_bool'] == 1]
        if len(fraud_cases) > 0:
            ax.scatter(fraud_cases['velocity_6h'], fraud_cases['velocity_24h'], 
                      c='red', s=8, alpha=0.8, label='Fraud Cases')
    
    # Add violin plot for session_length by device_os (simplified)
    if 'session_length_in_minutes' in df_sample.columns and 'device_os' in df_sample.columns:
        device_os_counts = df_sample['device_os'].value_counts().head(3)
        
        # Create inset for violin plot
        ax_inset = fig.add_axes([ax.get_position().x1 - 0.15, ax.get_position().y0 + 0.02, 
                               0.12, 0.08])
        
        violin_data = []
        violin_labels = []
        for device in device_os_counts.index:
            device_data = df_sample[df_sample['device_os'] == device]['session_length_in_minutes']
            device_data = device_data.dropna()
            if len(device_data) > 10:
                violin_data.append(device_data)
                violin_labels.append(device[:3])  # Truncate labels
        
        if violin_data:
            parts = ax_inset.violinplot(violin_data, showmeans=True)
            ax_inset.set_xticks(range(1, len(violin_labels) + 1))
            ax_inset.set_xticklabels(violin_labels, fontsize=6)
            ax_inset.set_title('Session Length\nby Device OS', fontsize=8)
            ax_inset.tick_params(axis='both', labelsize=6)
    
    # Add correlation heatmap as inset
    corr_features = ['velocity_6h', 'velocity_24h', 'velocity_4w', 'credit_risk_score']
    available_features = [f for f in corr_features if f in df_sample.columns]
    
    if len(available_features) >= 3:
        corr_data = df_sample[available_features].corr()
        
        # Create inset for heatmap
        ax_heatmap = fig.add_axes([ax.get_position().x0 + 0.02, ax.get_position().y1 - 0.12, 
                                 0.12, 0.1])
        
        im = ax_heatmap.imshow(corr_data.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_heatmap.set_xticks(range(len(available_features)))
        ax_heatmap.set_yticks(range(len(available_features)))
        ax_heatmap.set_xticklabels([f.replace('_', '\n')[:8] for f in available_features], 
                                  fontsize=6, rotation=45)
        ax_heatmap.set_yticklabels([f.replace('_', '\n')[:8] for f in available_features], 
                                  fontsize=6)
        ax_heatmap.set_title('Correlation', fontsize=8, fontweight='bold')
        
        # Add correlation values as text
        for x in range(len(available_features)):
            for y in range(len(available_features)):
                text = ax_heatmap.text(x, y, f'{corr_data.iloc[y, x]:.2f}',
                                     ha="center", va="center", color="black", fontsize=5)
    
    # Add colorbar for hexbin
    if 'hb' in locals():
        cb = plt.colorbar(hb, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label('Count', fontsize=8)
    
    # Styling
    ax.set_xlabel('Velocity 6h', fontweight='bold')
    ax.set_ylabel('Velocity 24h', fontweight='bold')
    ax.set_title(f'{variant}: Velocity & Device Analysis', fontweight='bold', fontsize=11)
    if len(df_sample[df_sample['fraud_bool'] == 1]) > 0:
        ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Fraud Detection Analysis Across Dataset Variants', 
             fontsize=14, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('fraud_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()