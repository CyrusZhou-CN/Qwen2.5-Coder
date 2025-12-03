import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler

# Load and combine data
train_df = pd.read_csv('train_energy_data.csv')
test_df = pd.read_csv('test_energy_data.csv')
df = pd.concat([train_df, test_df], ignore_index=True)

# Create figure with white background and professional styling
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Define color palette for building types
building_colors = {'Residential': '#2E86AB', 'Commercial': '#A23B72', 'Industrial': '#F18F01'}

# Top-left: Scatter plot with best-fit line and marginal histograms
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('white')

# Create scatter plot with points colored by building type and sized by occupants
for building_type in df['Building Type'].unique():
    mask = df['Building Type'] == building_type
    scatter = ax1.scatter(df[mask]['Square Footage'], df[mask]['Energy Consumption'], 
                         c=building_colors[building_type], s=df[mask]['Number of Occupants']*2,
                         alpha=0.6, label=building_type, edgecolors='white', linewidth=0.5)

# Add best-fit line
z = np.polyfit(df['Square Footage'], df['Energy Consumption'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Square Footage'].min(), df['Square Footage'].max(), 100)
ax1.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=2, label=f'Best fit (R²={stats.pearsonr(df["Square Footage"], df["Energy Consumption"])[0]**2:.3f})')

ax1.set_xlabel('Square Footage', fontsize=12, fontweight='bold')
ax1.set_ylabel('Energy Consumption', fontsize=12, fontweight='bold')
ax1.set_title('Energy Consumption vs Square Footage\n(Point size = Number of Occupants)', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Top-right: Correlation heatmap with bubble overlay
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('white')

# Select numerical columns for correlation
numerical_cols = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Energy Consumption']
corr_matrix = df[numerical_cols].corr()

# Create heatmap
im = ax2.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add bubble overlay
for i in range(len(numerical_cols)):
    for j in range(len(numerical_cols)):
        if i != j:  # Don't show bubbles on diagonal
            bubble_size = abs(corr_matrix.iloc[i, j]) * 1000
            circle = plt.Circle((j, i), radius=np.sqrt(bubble_size)/50, 
                              color='white', alpha=0.7, linewidth=2, fill=False)
            ax2.add_patch(circle)
        
        # Add correlation values
        text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
        ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold', color=text_color)

ax2.set_xticks(range(len(numerical_cols)))
ax2.set_yticks(range(len(numerical_cols)))
ax2.set_xticklabels([col.replace(' ', '\n') for col in numerical_cols], rotation=45, ha='right')
ax2.set_yticklabels(numerical_cols)
ax2.set_title('Correlation Matrix with Bubble Overlay\n(Bubble size = Correlation strength)', fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# Bottom-left: Violin plot with strip plot overlay
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('white')

# Prepare data for violin plot
building_types = df['Building Type'].unique()
day_types = df['Day of Week'].unique()
positions = []
violin_data = []
colors = []
labels = []

pos = 0
for building_type in building_types:
    for day_type in day_types:
        mask = (df['Building Type'] == building_type) & (df['Day of Week'] == day_type)
        if mask.sum() > 0:
            violin_data.append(df[mask]['Energy Consumption'])
            positions.append(pos)
            colors.append(building_colors[building_type])
            labels.append(f'{building_type}\n{day_type}')
            pos += 1

# Create violin plot
parts = ax3.violinplot(violin_data, positions=positions, widths=0.7, showmeans=True, showmedians=True)

# Color the violins
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Add strip plot overlay
for i, (pos, data) in enumerate(zip(positions, violin_data)):
    y_jitter = np.random.normal(pos, 0.1, len(data))
    ax3.scatter(y_jitter, data, alpha=0.4, s=20, color=colors[i], edgecolors='white', linewidth=0.5)

ax3.set_xticks(positions)
ax3.set_xticklabels(labels, rotation=45, ha='right')
ax3.set_ylabel('Energy Consumption', fontsize=12, fontweight='bold')
ax3.set_title('Energy Consumption Distribution\nby Building Type and Day Type', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, axis='y')

# Bottom-right: Parallel coordinates with density overlay
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('white')

# Prepare data for parallel coordinates
parallel_cols = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Energy Consumption']
parallel_data = df[parallel_cols + ['Building Type']].copy()

# Normalize data for parallel coordinates
scaler = StandardScaler()
parallel_data[parallel_cols] = scaler.fit_transform(parallel_data[parallel_cols])

# Create parallel coordinates plot
x_positions = range(len(parallel_cols))

for building_type in building_types:
    mask = parallel_data['Building Type'] == building_type
    building_data = parallel_data[mask][parallel_cols]
    
    for idx in building_data.index:
        y_values = building_data.loc[idx].values
        ax4.plot(x_positions, y_values, color=building_colors[building_type], 
                alpha=0.3, linewidth=1)

# Add density contours for each pair of adjacent variables
for i in range(len(parallel_cols)-1):
    x_data = parallel_data[parallel_cols[i]]
    y_data = parallel_data[parallel_cols[i+1]]
    
    # Create density contour
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x_data, y_data])
        kde = gaussian_kde(xy)
        
        # Create grid for contour
        x_grid = np.linspace(x_data.min(), x_data.max(), 20)
        y_grid = np.linspace(y_data.min(), y_data.max(), 20)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        Z = kde(positions).reshape(X_grid.shape)
        
        # Overlay contour (simplified representation)
        ax4.contour(np.linspace(i, i+1, 20), np.linspace(-3, 3, 20), Z, 
                   levels=3, alpha=0.3, colors='gray', linewidths=0.5)
    except:
        pass  # Skip if density calculation fails

# Create legend
legend_elements = [plt.Line2D([0], [0], color=building_colors[bt], lw=3, label=bt) 
                  for bt in building_types]
ax4.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

ax4.set_xticks(x_positions)
ax4.set_xticklabels([col.replace(' ', '\n') for col in parallel_cols], fontsize=10)
ax4.set_ylabel('Normalized Values', fontsize=12, fontweight='bold')
ax4.set_title('Parallel Coordinates with Density Overlay\n(Lines colored by Building Type)', fontsize=14, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()