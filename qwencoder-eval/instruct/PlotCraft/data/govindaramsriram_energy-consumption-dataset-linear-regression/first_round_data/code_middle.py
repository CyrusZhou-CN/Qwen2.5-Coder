import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Load and combine datasets
train_df = pd.read_csv('train_energy_data.csv')
test_df = pd.read_csv('test_energy_data.csv')
df = pd.concat([train_df, test_df], ignore_index=True)

# Select numerical variables for correlation analysis
numerical_vars = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Energy Consumption']
df_num = df[numerical_vars + ['Building Type']]

# Calculate correlation matrix
corr_matrix = df[numerical_vars].corr()

# Set up the figure with white background
plt.style.use('default')
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.patch.set_facecolor('white')

# Define colors and markers for building types
building_types = df['Building Type'].unique()
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
markers = ['o', 's', '^']
building_colors = dict(zip(building_types, colors))
building_markers = dict(zip(building_types, markers))

# Variable pairs for upper triangle
var_pairs = []
for i in range(len(numerical_vars)):
    for j in range(i+1, len(numerical_vars)):
        var_pairs.append((i, j))

# Create scatter plots in upper triangle
plot_idx = 0
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        
        if i < j:  # Upper triangle - scatter plots
            if plot_idx < len(var_pairs):
                var_i, var_j = var_pairs[plot_idx]
                x_var = numerical_vars[var_i]
                y_var = numerical_vars[var_j]
                
                # Plot scatter points for each building type
                for building_type in building_types:
                    mask = df['Building Type'] == building_type
                    x_data = df[mask][x_var]
                    y_data = df[mask][y_var]
                    
                    ax.scatter(x_data, y_data, 
                             c=building_colors[building_type], 
                             marker=building_markers[building_type],
                             alpha=0.6, s=30, 
                             label=building_type, edgecolors='white', linewidth=0.5)
                    
                    # Add trend line for each building type
                    if len(x_data) > 1:
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                        ax.plot(x_trend, p(x_trend), 
                               color=building_colors[building_type], 
                               linestyle='--', alpha=0.8, linewidth=2)
                
                # Calculate and display correlation coefficient
                correlation = corr_matrix.loc[x_var, y_var]
                ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(x_var, fontweight='bold')
                ax.set_ylabel(y_var, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add legend only to the first subplot
                if plot_idx == 0:
                    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
                
                plot_idx += 1
            else:
                ax.set_visible(False)
                
        elif i == j and i == 1:  # Center diagonal - correlation heatmap
            # Create correlation heatmap
            im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values as text annotations
            for row in range(len(numerical_vars)):
                for col in range(len(numerical_vars)):
                    text_color = 'white' if abs(corr_matrix.iloc[row, col]) > 0.5 else 'black'
                    ax.text(col, row, f'{corr_matrix.iloc[row, col]:.2f}',
                           ha='center', va='center', fontweight='bold', 
                           color=text_color, fontsize=10)
            
            # Set ticks and labels
            ax.set_xticks(range(len(numerical_vars)))
            ax.set_yticks(range(len(numerical_vars)))
            ax.set_xticklabels([var.replace(' ', '\n') for var in numerical_vars], 
                              rotation=45, ha='right', fontweight='bold')
            ax.set_yticklabels([var.replace(' ', '\n') for var in numerical_vars], 
                              fontweight='bold')
            ax.set_title('Correlation Matrix', fontweight='bold', fontsize=14, pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation Coefficient', fontweight='bold')
            
        else:  # Lower triangle and other diagonal elements - hide
            ax.set_visible(False)

# Set main title
fig.suptitle('Energy Consumption Correlation Analysis:\nScatter Plot Matrix with Building Type Differentiation', 
             fontsize=16, fontweight='bold', y=0.95)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3)

plt.show()