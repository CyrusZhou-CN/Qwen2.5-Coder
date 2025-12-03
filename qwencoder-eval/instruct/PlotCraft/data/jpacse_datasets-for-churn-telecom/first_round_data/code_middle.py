import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df = pd.read_csv('cell2celltrain.csv')

# Select the key numerical variables for correlation analysis
correlation_vars = ['MonthlyRevenue', 'MonthlyMinutes', 'DroppedCalls', 
                   'UnansweredCalls', 'CustomerCareCalls', 'MonthsInService']

# Create correlation matrix
corr_data = df[correlation_vars].corr()

# Find the three most correlated variable pairs (excluding self-correlations)
corr_abs = corr_data.abs()
np.fill_diagonal(corr_abs.values, 0)  # Remove diagonal values
upper_triangle = np.triu(corr_abs, k=1)
max_corr_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
max_corr_vars = [correlation_vars[max_corr_idx[0]], correlation_vars[max_corr_idx[1]]]

# Find second highest correlation
upper_triangle[max_corr_idx] = 0
second_max_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
second_var = correlation_vars[second_max_idx[0]] if correlation_vars[second_max_idx[0]] not in max_corr_vars else correlation_vars[second_max_idx[1]]

# Select top 3 most correlated variables
top_vars = max_corr_vars + [second_var]

# Create figure with 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

# Top subplot: Correlation heatmap
sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8},
            ax=ax1, linewidths=0.5)
ax1.set_title('Correlation Heatmap of Key Customer Service Variables', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('')
ax1.set_ylabel('')

# Rotate labels for better readability
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

# Bottom subplot: Scatter plot matrix of top 3 correlated variables
# Prepare data for scatter plot matrix
plot_data = df[top_vars + ['Churn']].dropna()

# Create color mapping for churn status
colors = {'Yes': '#e74c3c', 'No': '#3498db'}
plot_data['color'] = plot_data['Churn'].map(colors)

# Create scatter plot matrix manually
n_vars = len(top_vars)
for i in range(n_vars):
    for j in range(n_vars):
        subplot_idx = i * n_vars + j + 1
        
        if i == j:
            # Diagonal: histogram
            ax_sub = plt.subplot(n_vars, n_vars, subplot_idx)
            for churn_status in ['No', 'Yes']:
                data_subset = plot_data[plot_data['Churn'] == churn_status][top_vars[i]]
                ax_sub.hist(data_subset, alpha=0.6, label=f'Churn: {churn_status}', 
                           color=colors[churn_status], bins=30, density=True)
            ax_sub.set_xlabel(top_vars[i] if i == n_vars-1 else '')
            ax_sub.set_ylabel('Density' if j == 0 else '')
            if i == 0 and j == 0:
                ax_sub.legend()
        else:
            # Off-diagonal: scatter plot
            ax_sub = plt.subplot(n_vars, n_vars, subplot_idx)
            for churn_status in ['No', 'Yes']:
                data_subset = plot_data[plot_data['Churn'] == churn_status]
                ax_sub.scatter(data_subset[top_vars[j]], data_subset[top_vars[i]], 
                              alpha=0.5, s=10, color=colors[churn_status], 
                              label=f'Churn: {churn_status}')
            ax_sub.set_xlabel(top_vars[j] if i == n_vars-1 else '')
            ax_sub.set_ylabel(top_vars[i] if j == 0 else '')
            if i == 0 and j == 1:
                ax_sub.legend()

# Position the scatter plot matrix in the bottom subplot area
plt.sca(ax2)
ax2.set_position([0.1, 0.05, 0.8, 0.35])

# Clear the bottom subplot and create the scatter plot matrix there
ax2.clear()
ax2.axis('off')

# Create a new figure area for the scatter plot matrix
gs = fig.add_gridspec(3, 3, left=0.1, right=0.9, bottom=0.05, top=0.4, 
                      hspace=0.3, wspace=0.3)

for i in range(3):
    for j in range(3):
        ax_sub = fig.add_subplot(gs[i, j])
        
        if i == j:
            # Diagonal: histogram
            for churn_status in ['No', 'Yes']:
                data_subset = plot_data[plot_data['Churn'] == churn_status][top_vars[i]]
                ax_sub.hist(data_subset, alpha=0.6, label=f'Churn: {churn_status}', 
                           color=colors[churn_status], bins=25, density=True)
            ax_sub.set_xlabel(top_vars[i] if i == 2 else '')
            ax_sub.set_ylabel('Density' if j == 0 else '')
            if i == 0 and j == 0:
                ax_sub.legend(fontsize=8)
        else:
            # Off-diagonal: scatter plot
            for churn_status in ['No', 'Yes']:
                data_subset = plot_data[plot_data['Churn'] == churn_status]
                ax_sub.scatter(data_subset[top_vars[j]], data_subset[top_vars[i]], 
                              alpha=0.4, s=8, color=colors[churn_status], 
                              label=f'Churn: {churn_status}')
            ax_sub.set_xlabel(top_vars[j] if i == 2 else '')
            ax_sub.set_ylabel(top_vars[i] if j == 0 else '')
            if i == 0 and j == 1:
                ax_sub.legend(fontsize=8)
        
        # Style the subplots
        ax_sub.grid(True, alpha=0.3)
        ax_sub.tick_params(labelsize=8)

# Add title for the scatter plot matrix
fig.text(0.5, 0.45, f'Scatter Plot Matrix: Top 3 Correlated Variables by Churn Status', 
         ha='center', fontsize=14, fontweight='bold')

# Set white background
fig.patch.set_facecolor('white')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05)
plt.show()