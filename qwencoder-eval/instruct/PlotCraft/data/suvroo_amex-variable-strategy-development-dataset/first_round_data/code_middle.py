import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_excel('Book1.xlsx')

# Select only numerical columns for correlation analysis
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove unique_identifier as it's not meaningful for correlation
if 'unique_identifier' in numerical_cols:
    numerical_cols.remove('unique_identifier')

# Sample data for faster processing (use 10% of data)
df_sample = df.sample(n=min(6000, len(df)), random_state=42)

# Create correlation matrix with default_ind
correlation_data = df_sample[numerical_cols].corr()

# Get correlations with default_ind and sort by absolute value
default_correlations = correlation_data['default_ind'].abs().sort_values(ascending=False)

# Remove default_ind itself and get top 5 variables most correlated with default
top_5_vars = default_correlations.drop('default_ind').head(5).index.tolist()

# Create the main composite figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Top Left: Full correlation heatmap (subset for readability)
important_vars = default_correlations.head(15).index.tolist()  # Top 15 most correlated
correlation_subset = correlation_data.loc[important_vars, important_vars]
mask = np.triu(np.ones_like(correlation_subset, dtype=bool))
sns.heatmap(correlation_subset, mask=mask, annot=False, 
            cmap='RdBu_r', center=0, square=True, linewidths=0.1, 
            cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('Correlation Heatmap: Top Variables vs Default Risk', 
              fontweight='bold', fontsize=12)
ax1.tick_params(axis='both', labelsize=8)

# Top Right: Bar chart of top correlations with default
default_corr_top5 = correlation_data['default_ind'][top_5_vars].sort_values(key=abs, ascending=True)
colors = ['#A23B72' if x < 0 else '#2E86AB' for x in default_corr_top5.values]
bars = ax2.barh(range(len(default_corr_top5)), default_corr_top5.values, color=colors)
ax2.set_yticks(range(len(default_corr_top5)))
ax2.set_yticklabels(default_corr_top5.index, fontsize=10)
ax2.set_xlabel('Correlation with Default Risk', fontweight='bold', fontsize=10)
ax2.set_title('Top 5 Variables: Correlation with Default Risk', 
              fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
ax2.axvline(x=0, color='black', linewidth=1)

# Add correlation values on bars
for i, (bar, val) in enumerate(zip(bars, default_corr_top5.values)):
    ax2.text(val + (0.005 if val > 0 else -0.005), i, f'{val:.3f}', 
             va='center', ha='left' if val > 0 else 'right', fontsize=9)

# Bottom Left: Scatter plot of most correlated variable
top_var = top_5_vars[0]
sample_data = df_sample[[top_var, 'default_ind']].dropna()
colors_scatter = ['#2E86AB' if x == 0 else '#A23B72' for x in sample_data['default_ind']]
ax3.scatter(sample_data[top_var], sample_data['default_ind'], 
           c=colors_scatter, alpha=0.6, s=10)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    sample_data[top_var], sample_data['default_ind'])
line_x = np.array([sample_data[top_var].min(), sample_data[top_var].max()])
line_y = slope * line_x + intercept
ax3.plot(line_x, line_y, 'black', linewidth=2, alpha=0.8)

ax3.set_xlabel(top_var, fontweight='bold', fontsize=10)
ax3.set_ylabel('Default Indicator', fontweight='bold', fontsize=10)
ax3.set_title(f'Scatter Plot: {top_var} vs Default Risk\n(r={r_value:.3f})', 
              fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Bottom Right: Distribution comparison for top variable
default_0 = sample_data[sample_data['default_ind'] == 0][top_var]
default_1 = sample_data[sample_data['default_ind'] == 1][top_var]

ax4.hist(default_0, alpha=0.7, color='#2E86AB', bins=30, label='No Default', density=True)
ax4.hist(default_1, alpha=0.7, color='#A23B72', bins=30, label='Default', density=True)
ax4.set_xlabel(top_var, fontweight='bold', fontsize=10)
ax4.set_ylabel('Density', fontweight='bold', fontsize=10)
ax4.set_title(f'Distribution Comparison: {top_var}', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Financial Variables and Default Risk Analysis: Comprehensive Correlation View', 
             fontweight='bold', fontsize=14, y=0.95)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Create second figure for pair plot of top 3 variables
fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
fig2.patch.set_facecolor('white')

# Use top 3 variables for pair plot to avoid timeout
top_3_vars = top_5_vars[:3]
plot_data = df_sample[top_3_vars + ['default_ind']].dropna()

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histogram
            for default_val in [0, 1]:
                subset = plot_data[plot_data['default_ind'] == default_val]
                color = '#2E86AB' if default_val == 0 else '#A23B72'
                label = 'No Default' if default_val == 0 else 'Default'
                ax.hist(subset[top_3_vars[i]], alpha=0.7, color=color, bins=20,
                       label=label, density=True)
            ax.set_xlabel(top_3_vars[i], fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            if i == 0:
                ax.legend(fontsize=9)
        else:
            # Off-diagonal: scatter plot with regression line
            x_var = top_3_vars[j]
            y_var = top_3_vars[i]
            
            # Sample further for scatter plots to avoid overcrowding
            plot_sample = plot_data.sample(n=min(1000, len(plot_data)), random_state=42)
            colors_scatter = ['#2E86AB' if x == 0 else '#A23B72' for x in plot_sample['default_ind']]
            
            # Scatter plot
            ax.scatter(plot_sample[x_var], plot_sample[y_var], 
                      c=colors_scatter, alpha=0.6, s=5)
            
            # Add regression line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    plot_sample[x_var], plot_sample[y_var])
                line_x = np.array([plot_sample[x_var].min(), plot_sample[x_var].max()])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'black', linewidth=1.5, alpha=0.8)
                
                # Add correlation coefficient as text
                ax.text(0.05, 0.95, f'r={r_value:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8, verticalalignment='top')
            except:
                pass
            
            ax.set_xlabel(x_var, fontsize=10)
            ax.set_ylabel(y_var, fontsize=10)
        
        # Clean up axes
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)

# Set title for pair plot
fig2.suptitle('Pair Plot: Top 3 Variables Most Correlated with Default Risk', 
              fontweight='bold', fontsize=14, y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Print summary statistics
print("Top 5 Variables Most Correlated with Default Risk:")
print("=" * 50)
for var in top_5_vars:
    corr_val = correlation_data.loc['default_ind', var]
    print(f"{var}: {corr_val:.4f}")

print(f"\nDataset shape: {df.shape}")
print(f"Sample used for analysis: {df_sample.shape}")
print(f"Default rate in sample: {df_sample['default_ind'].mean():.3f}")

plt.savefig('amex_default_risk_analysis.png', dpi=300, bbox_inches='tight')