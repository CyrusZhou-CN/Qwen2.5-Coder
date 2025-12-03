import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for professional appearance
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Load all datasets
datasets = {}
dataset_files = ['Base.csv', 'Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
dataset_names = ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']

for file, name in zip(dataset_files, dataset_names):
    try:
        datasets[name] = pd.read_csv(file)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Create figure with better proportions (less wide)
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('white')

# Create 2x3 subplot grid
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# Define colors for fraud/non-fraud
fraud_colors = ['#2E86AB', '#F24236']  # Professional blue and red
employment_colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

# Top row: Scatter plots with marginal histograms for Base, Variant I, Variant II
top_datasets = ['Base', 'Variant I', 'Variant II']

for i, dataset_name in enumerate(top_datasets):
    if dataset_name not in datasets:
        continue
        
    df = datasets[dataset_name]
    
    # Sample data for performance
    sample_size = min(3000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Create subplot with marginal histograms using gridspec
    ax_main = fig.add_subplot(gs[0, i])
    
    # Create scatter plot colored by fraud_bool
    for fraud_val, color, label in zip([0, 1], fraud_colors, ['Non-Fraud', 'Fraud']):
        mask = df_sample['fraud_bool'] == fraud_val
        if mask.sum() > 0:
            ax_main.scatter(df_sample[mask]['credit_risk_score'], 
                          df_sample[mask]['income'],
                          c=color, alpha=0.7, s=20, label=label, edgecolors='white', linewidth=0.5)
    
    # Add marginal histograms as insets
    # Top histogram (income distribution)
    hist_top = ax_main.inset_axes([0, 1.02, 1, 0.2])
    for fraud_val, color in zip([0, 1], fraud_colors):
        mask = df_sample['fraud_bool'] == fraud_val
        if mask.sum() > 0:
            hist_top.hist(df_sample[mask]['income'], bins=30, alpha=0.7, color=color, density=True)
    hist_top.set_xlim(ax_main.get_xlim())
    hist_top.set_xticks([])
    hist_top.set_ylabel('Density', fontsize=8)
    hist_top.tick_params(labelsize=7)
    
    # Right histogram (credit_risk_score distribution)
    hist_right = ax_main.inset_axes([1.02, 0, 0.2, 1])
    for fraud_val, color in zip([0, 1], fraud_colors):
        mask = df_sample['fraud_bool'] == fraud_val
        if mask.sum() > 0:
            hist_right.hist(df_sample[mask]['credit_risk_score'], bins=30, alpha=0.7, 
                           color=color, orientation='horizontal', density=True)
    hist_right.set_ylim(ax_main.get_ylim())
    hist_right.set_yticks([])
    hist_right.set_xlabel('Density', fontsize=8)
    hist_right.tick_params(labelsize=7)
    
    # Main plot formatting
    ax_main.set_xlabel('Credit Risk Score', fontweight='bold', fontsize=11)
    ax_main.set_ylabel('Income', fontweight='bold', fontsize=11)
    ax_main.set_title(f'{dataset_name}\nCredit Risk vs Income with Marginal Distributions', 
                     fontweight='bold', fontsize=12, pad=15)
    ax_main.legend(frameon=True, fontsize=10, loc='upper right')
    ax_main.grid(True, alpha=0.3, linewidth=0.5)
    
    # Set reasonable axis limits
    ax_main.set_xlim(df_sample['credit_risk_score'].quantile(0.01), 
                    df_sample['credit_risk_score'].quantile(0.99))
    ax_main.set_ylim(df_sample['income'].quantile(0.01), 
                    df_sample['income'].quantile(0.99))

# Bottom row: Violin plots with strip and box plots for Variant III, IV, V
bottom_datasets = ['Variant III', 'Variant IV', 'Variant V']

for i, dataset_name in enumerate(bottom_datasets):
    if dataset_name not in datasets:
        continue
        
    df = datasets[dataset_name]
    
    # Sample data for performance
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    ax = fig.add_subplot(gs[1, i])
    
    # Get unique employment statuses and limit to top 5 for readability
    employment_counts = df_sample['employment_status'].value_counts()
    top_statuses = employment_counts.head(5).index.tolist()
    df_filtered = df_sample[df_sample['employment_status'].isin(top_statuses)]
    
    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{dataset_name}\nSession Length by Employment Status', 
                    fontweight='bold', fontsize=12)
        continue
    
    # Create violin plot
    try:
        # Violin plot as base
        parts = ax.violinplot([df_filtered[df_filtered['employment_status'] == status]['session_length_in_minutes'].values 
                              for status in top_statuses], 
                             positions=range(len(top_statuses)), 
                             showmeans=False, showmedians=False, showextrema=False)
        
        # Color the violin plots
        for i_part, pc in enumerate(parts['bodies']):
            pc.set_facecolor(employment_colors[i_part % len(employment_colors)])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # Add box plots on top
        box_data = [df_filtered[df_filtered['employment_status'] == status]['session_length_in_minutes'].values 
                   for status in top_statuses]
        bp = ax.boxplot(box_data, positions=range(len(top_statuses)), 
                       widths=0.3, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.8),
                       medianprops=dict(color='red', linewidth=2))
        
        # Add strip plot overlay
        for j, status in enumerate(top_statuses):
            status_data = df_filtered[df_filtered['employment_status'] == status]['session_length_in_minutes']
            # Sample points for strip plot to avoid overcrowding
            if len(status_data) > 100:
                status_sample = status_data.sample(100, random_state=42)
            else:
                status_sample = status_data
            
            # Add jitter to x-coordinates
            x_jitter = np.random.normal(j, 0.05, len(status_sample))
            ax.scatter(x_jitter, status_sample, alpha=0.4, s=8, color='black')
        
        # Set x-axis labels
        ax.set_xticks(range(len(top_statuses)))
        ax.set_xticklabels(top_statuses, rotation=45, ha='right')
        
    except Exception as e:
        print(f"Error creating violin plot for {dataset_name}: {e}")
        # Fallback to simple box plot
        df_filtered.boxplot(column='session_length_in_minutes', by='employment_status', ax=ax)
    
    ax.set_xlabel('Employment Status', fontweight='bold', fontsize=11)
    ax.set_ylabel('Session Length (minutes)', fontweight='bold', fontsize=11)
    ax.set_title(f'{dataset_name}\nSession Length Distribution by Employment Status', 
                fontweight='bold', fontsize=12, pad=10)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Set reasonable y-axis limits
    y_data = df_filtered['session_length_in_minutes']
    ax.set_ylim(max(0, y_data.quantile(0.01)), y_data.quantile(0.95))

# Add overall title with perfect centering
fig.suptitle('Fraud Detection Pattern Analysis Across Dataset Variants', 
             fontsize=16, fontweight='bold', y=0.95, x=0.5)

# Final layout adjustment
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.show()