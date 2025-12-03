import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

# Load data and sample for computational efficiency
df = pd.read_csv('Ninapro_DB1.csv')
# Sample 15000 points for analysis to ensure reasonable computation time
df_sample = df.sample(n=15000, random_state=42)

# Extract EMG and glove columns
emg_cols = [col for col in df_sample.columns if col.startswith('emg_')]
glove_cols = [col for col in df_sample.columns if col.startswith('glove_')]

# Calculate variability to find most variable sensors
emg_var = df_sample[emg_cols].var()
glove_var = df_sample[glove_cols].var()
most_var_emg = emg_var.idxmax()
most_var_glove = glove_var.idxmax()

# Find top variable glove sensors for pairwise analysis
top_glove_sensors = glove_var.nlargest(4).index.tolist()

# Calculate cross-modal correlations to find strongest EMG-glove pair
cross_corr = np.corrcoef(df_sample[emg_cols].T, df_sample[glove_cols].T)
emg_glove_corr = cross_corr[:len(emg_cols), len(emg_cols):]
max_corr_idx = np.unravel_index(np.argmax(np.abs(emg_glove_corr)), emg_glove_corr.shape)
strongest_emg = emg_cols[max_corr_idx[0]]
strongest_glove = glove_cols[max_corr_idx[1]]

# Define color schemes
emg_colors = ['#2E4A87', '#4A6FA5', '#6B8EC3']  # Blue tones for EMG
glove_colors = ['#D2691E', '#FF6347', '#FF8C00']  # Orange/red tones for glove

# Create the 3x3 subplot grid with proper spacing
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: EMG 5 vs EMG 6 with marginal histograms
ax1 = plt.subplot(3, 3, 1)
x, y = df_sample['emg_5'], df_sample['emg_6']

# Create main scatter plot
ax1.scatter(x, y, alpha=0.6, c=emg_colors[0], s=15, edgecolors='none')
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax1.plot(x_line, p(x_line), color=emg_colors[1], linewidth=3)

# Add marginal histograms using divider
divider = make_axes_locatable(ax1)
ax_hist_x = divider.append_axes("top", size="20%", pad=0.1, sharex=ax1)
ax_hist_y = divider.append_axes("right", size="20%", pad=0.1, sharey=ax1)

# Top histogram
ax_hist_x.hist(x, bins=30, alpha=0.7, color=emg_colors[0], density=True)
ax_hist_x.set_xticks([])
ax_hist_x.set_ylabel('Density', fontweight='bold', fontsize=9)

# Right histogram
ax_hist_y.hist(y, bins=30, alpha=0.7, color=emg_colors[0], density=True, orientation='horizontal')
ax_hist_y.set_yticks([])
ax_hist_y.set_xlabel('Density', fontweight='bold', fontsize=9)

corr_coef, p_val = stats.pearsonr(x, y)
ax1.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_val:.3e}', transform=ax1.transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), 
         fontweight='bold', fontsize=10)
ax1.set_xlabel('EMG Channel 5', fontweight='bold', fontsize=11)
ax1.set_ylabel('EMG Channel 6', fontweight='bold', fontsize=11)
ax1.set_title('EMG 5 vs EMG 6 with Marginal Histograms', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, linewidth=0.5)

# Row 1, Subplot 2: EMG 7 vs EMG 9 with marginal density plots
ax2 = plt.subplot(3, 3, 2)
x, y = df_sample['emg_7'], df_sample['emg_9']

# Create main scatter plot
ax2.scatter(x, y, alpha=0.6, c=emg_colors[0], s=15, edgecolors='none')
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax2.plot(x_line, p(x_line), color=emg_colors[1], linewidth=3)

# Add marginal density plots
divider2 = make_axes_locatable(ax2)
ax_dens_x = divider2.append_axes("top", size="20%", pad=0.1, sharex=ax2)
ax_dens_y = divider2.append_axes("right", size="20%", pad=0.1, sharey=ax2)

# Top density plot
x_range = np.linspace(x.min(), x.max(), 100)
kde_x = stats.gaussian_kde(x)
ax_dens_x.plot(x_range, kde_x(x_range), color=emg_colors[1], linewidth=2)
ax_dens_x.fill_between(x_range, kde_x(x_range), alpha=0.3, color=emg_colors[0])
ax_dens_x.set_xticks([])
ax_dens_x.set_ylabel('Density', fontweight='bold', fontsize=9)

# Right density plot
y_range = np.linspace(y.min(), y.max(), 100)
kde_y = stats.gaussian_kde(y)
ax_dens_y.plot(kde_y(y_range), y_range, color=emg_colors[1], linewidth=2)
ax_dens_y.fill_betweenx(y_range, kde_y(y_range), alpha=0.3, color=emg_colors[0])
ax_dens_y.set_yticks([])
ax_dens_y.set_xlabel('Density', fontweight='bold', fontsize=9)

corr_coef, p_val = stats.pearsonr(x, y)
ax2.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_val:.3e}', transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), 
         fontweight='bold', fontsize=10)
ax2.set_xlabel('EMG Channel 7', fontweight='bold', fontsize=11)
ax2.set_ylabel('EMG Channel 9', fontweight='bold', fontsize=11)
ax2.set_title('EMG 7 vs EMG 9 with Marginal Density Plots', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, linewidth=0.5)

# Row 1, Subplot 3: Bubble plot EMG 0, 4, 8
ax3 = plt.subplot(3, 3, 3)
x, y = df_sample['emg_0'], df_sample['emg_4']
sizes = (df_sample['emg_8'] - df_sample['emg_8'].min()) / (df_sample['emg_8'].max() - df_sample['emg_8'].min()) * 200 + 20
scatter = ax3.scatter(x, y, s=sizes, alpha=0.6, c=emg_colors[0], edgecolors='white', linewidth=0.5)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax3.plot(x_line, p(x_line), color=emg_colors[1], linewidth=3)
corr_coef, p_val = stats.pearsonr(x, y)
ax3.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_val:.3e}', transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), 
         fontweight='bold', fontsize=10)
ax3.set_xlabel('EMG Channel 0', fontweight='bold', fontsize=11)
ax3.set_ylabel('EMG Channel 4', fontweight='bold', fontsize=11)
ax3.set_title('EMG Bubble Plot (Size = EMG 8)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, linewidth=0.5)

# Row 2, Subplot 4: Glove 10 vs Glove 19 with confidence intervals
ax4 = plt.subplot(3, 3, 4)
x, y = df_sample['glove_10'], df_sample['glove_19']
ax4.scatter(x, y, alpha=0.6, c=glove_colors[0], s=15, edgecolors='none')
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = p(x_line)
ax4.plot(x_line, y_line, color=glove_colors[1], linewidth=3)
# Add confidence interval
residuals = y - p(x)
std_err = np.std(residuals)
ax4.fill_between(x_line, y_line - 1.96*std_err, y_line + 1.96*std_err, 
                 alpha=0.3, color=glove_colors[1], label='95% CI')
corr_coef, p_val = stats.pearsonr(x, y)
ax4.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_val:.3e}', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), 
         fontweight='bold', fontsize=10)
ax4.set_xlabel('Glove Sensor 10', fontweight='bold', fontsize=11)
ax4.set_ylabel('Glove Sensor 19', fontweight='bold', fontsize=11)
ax4.set_title('Glove 10 vs 19 with Confidence Intervals', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, linewidth=0.5)
ax4.legend(fontsize=9)

# Row 2, Subplot 5: Correlation heatmap with hierarchical clustering dendrograms
ax5 = plt.subplot(3, 3, 5)
glove_corr = df_sample[glove_cols].corr()

# Perform hierarchical clustering
linkage_matrix = linkage(pdist(glove_corr), method='ward')
dendro = dendrogram(linkage_matrix, labels=glove_corr.columns, no_plot=True)
cluster_order = dendro['leaves']
glove_corr_clustered = glove_corr.iloc[cluster_order, cluster_order]

# Create dendrograms
divider5 = make_axes_locatable(ax5)
ax_dendro_top = divider5.append_axes("top", size="15%", pad=0.1)
ax_dendro_left = divider5.append_axes("left", size="15%", pad=0.1)

# Top dendrogram
dendro_top = dendrogram(linkage_matrix, ax=ax_dendro_top, orientation='top', 
                       labels=None, leaf_rotation=90, color_threshold=0)
ax_dendro_top.set_xticks([])
ax_dendro_top.set_yticks([])

# Left dendrogram
dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left', 
                        labels=None, color_threshold=0)
ax_dendro_left.set_xticks([])
ax_dendro_left.set_yticks([])

# Main heatmap
im = ax5.imshow(glove_corr_clustered, cmap='Reds', aspect='auto', vmin=-1, vmax=1)
ax5.set_xticks(range(len(glove_cols)))
ax5.set_yticks(range(len(glove_cols)))
ax5.set_xticklabels([glove_cols[i].replace('glove_', 'G') for i in cluster_order], 
                    rotation=45, ha='right', fontsize=8)
ax5.set_yticklabels([glove_cols[i].replace('glove_', 'G') for i in cluster_order], fontsize=8)
ax5.set_title('Glove Sensor Correlation Heatmap\nwith Hierarchical Clustering', fontweight='bold', fontsize=12)

# Add colorbar
divider5_cb = make_axes_locatable(ax5)
cax = divider5_cb.append_axes("right", size="5%", pad=0.3)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# Row 2, Subplot 6: Pairwise scatter plot matrix for top glove sensors
ax6 = plt.subplot(3, 3, 6)
selected_sensors = top_glove_sensors[:4]  # Use top 4 sensors

# Create a simple pairplot-like visualization
n_sensors = len(selected_sensors)
grid_size = 2  # 2x2 grid for 4 sensors

for i in range(grid_size):
    for j in range(grid_size):
        idx = i * grid_size + j
        if idx < n_sensors:
            sensor = selected_sensors[idx]
            # Create small subplot within the main subplot
            sub_x = j * 0.45 + 0.05
            sub_y = (1 - i) * 0.45 + 0.05
            sub_w = 0.4
            sub_h = 0.4
            
            # Create inset axes
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            sub_ax = inset_axes(ax6, width="40%", height="40%", 
                               bbox_to_anchor=(sub_x, sub_y, sub_w, sub_h), 
                               bbox_transform=ax6.transAxes, loc='lower left')
            
            if i == j:  # Diagonal: histogram
                data = df_sample[sensor]
                sub_ax.hist(data, bins=15, alpha=0.7, color=glove_colors[0], density=True)
                sub_ax.set_title(f'{sensor.replace("glove_", "G")}', fontsize=8, fontweight='bold')
            else:  # Off-diagonal: scatter plot
                if idx < n_sensors - 1:
                    x_data = df_sample[selected_sensors[j]]
                    y_data = df_sample[selected_sensors[i]]
                    sub_ax.scatter(x_data, y_data, alpha=0.5, c=glove_colors[0], s=5)
                    corr = np.corrcoef(x_data, y_data)[0, 1]
                    sub_ax.text(0.05, 0.95, f'r={corr:.2f}', transform=sub_ax.transAxes, 
                               fontsize=7, fontweight='bold')
            
            sub_ax.tick_params(labelsize=6)

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_title('Pairwise Scatter Plot Matrix\nTop 4 Variable Glove Sensors', fontweight='bold', fontsize=12)

# Row 3, Subplot 7: Most variable EMG vs Glove with polynomial fit and residual plot
ax7 = plt.subplot(3, 3, 7)
x, y = df_sample[most_var_emg], df_sample[most_var_glove]

# Create main plot area (top 70%)
ax7_main = plt.subplot2grid((10, 3), (0, 0), rowspan=7, colspan=1, fig=fig)
ax7_main.scatter(x, y, alpha=0.6, c='purple', s=15, edgecolors='none')

# Polynomial fit
z = np.polyfit(x, y, 2)
p = np.poly1d(z)
x_smooth = np.linspace(x.min(), x.max(), 100)
ax7_main.plot(x_smooth, p(x_smooth), color='red', linewidth=3, label='Polynomial Fit')

# Create residual plot (bottom 30%)
ax7_resid = plt.subplot2grid((10, 3), (7, 0), rowspan=3, colspan=1, fig=fig, sharex=ax7_main)
residuals = y - p(x)
ax7_resid.scatter(x, residuals, alpha=0.6, c='red', s=10)
ax7_resid.axhline(y=0, color='black', linestyle='--', alpha=0.7)
ax7_resid.set_ylabel('Residuals', fontweight='bold', fontsize=10)

# Add statistics
rmse = np.sqrt(np.mean(residuals**2))
corr_coef, p_val = stats.pearsonr(x, y)
ax7_main.text(0.05, 0.95, f'r = {corr_coef:.3f}\nRMSE = {rmse:.3f}', transform=ax7_main.transAxes,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), 
              fontweight='bold', fontsize=10)

ax7_main.set_ylabel(f'{most_var_glove.replace("glove_", "Glove Sensor ")}', fontweight='bold', fontsize=11)
ax7_resid.set_xlabel(f'{most_var_emg.replace("emg_", "EMG Channel ")}', fontweight='bold', fontsize=11)
ax7_main.set_title('Most Variable EMG-Glove Correlation\nwith Polynomial Fit and Residuals', fontweight='bold', fontsize=12)
ax7_main.grid(True, alpha=0.3, linewidth=0.5)
ax7_resid.grid(True, alpha=0.3, linewidth=0.5)
ax7_main.legend(fontsize=9)

# Row 3, Subplot 8: Network graph visualization
ax8 = plt.subplot(3, 3, 8)
top_emg = emg_var.nlargest(5).index
top_glove = glove_var.nlargest(5).index

# Create network graph
G = nx.Graph()
# Add nodes
for emg in top_emg:
    G.add_node(emg, node_type='emg')
for glove in top_glove:
    G.add_node(glove, node_type='glove')

# Add edges based on correlation strength
threshold = 0.2  # Lower threshold to show more connections
for emg in top_emg:
    for glove in top_glove:
        corr = np.corrcoef(df_sample[emg], df_sample[glove])[0, 1]
        if abs(corr) > threshold:
            G.add_edge(emg, glove, weight=abs(corr))

# Create bipartite layout
emg_nodes = [n for n in G.nodes() if n.startswith('emg_')]
glove_nodes = [n for n in G.nodes() if n.startswith('glove_')]

pos = {}
# Position EMG nodes in a circle on the left
for i, node in enumerate(emg_nodes):
    angle = 2 * np.pi * i / len(emg_nodes)
    pos[node] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))

# Position glove nodes in a circle on the right
for i, node in enumerate(glove_nodes):
    angle = 2 * np.pi * i / len(glove_nodes)
    pos[node] = (1 + 0.3 * np.cos(angle), 0.3 * np.sin(angle))

# Draw network with edges
node_colors = [emg_colors[0] if n.startswith('emg_') else glove_colors[0] for n in G.nodes()]
edge_weights = [G[u][v]['weight'] * 8 for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, ax=ax8, node_color=node_colors, node_size=1000, alpha=0.8)
nx.draw_networkx_edges(G, pos, ax=ax8, width=edge_weights, edge_color='gray', alpha=0.6)
nx.draw_networkx_labels(G, pos, ax=ax8, 
                       labels={n: n.replace('emg_', 'E').replace('glove_', 'G') for n in G.nodes()},
                       font_size=9, font_weight='bold')

ax8.set_title('EMG-Glove Correlation Network\n(Edge thickness = |correlation|)', 
              fontweight='bold', fontsize=12)
ax8.set_xlim(-0.8, 1.8)
ax8.set_ylim(-0.8, 0.8)
ax8.axis('off')

# Row 3, Subplot 9: 2D histogram of strongest correlation pair
ax9 = plt.subplot(3, 3, 9)
x, y = df_sample[strongest_emg], df_sample[strongest_glove]
# Create 2D histogram
hist, xedges, yedges = np.histogram2d(x, y, bins=25)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax9.imshow(hist.T, extent=extent, origin='lower', cmap='plasma', aspect='auto')

# Add contour lines
X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
ax9.contour(X, Y, hist.T, levels=5, colors='white', alpha=0.8, linewidths=1.5)

corr_coef, p_val = stats.pearsonr(x, y)
ax9.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_val:.3e}', transform=ax9.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'), 
         fontweight='bold', fontsize=10)
ax9.set_xlabel(f'{strongest_emg.replace("emg_", "EMG Channel ")}', fontweight='bold', fontsize=11)
ax9.set_ylabel(f'{strongest_glove.replace("glove_", "Glove Sensor ")}', fontweight='bold', fontsize=11)
ax9.set_title('Strongest EMG-Glove Joint Distribution\nwith Density Contours', fontweight='bold', fontsize=12)

# Add colorbar
divider9 = make_axes_locatable(ax9)
cax9 = divider9.append_axes("right", size="5%", pad=0.1)
cbar9 = plt.colorbar(im, cax=cax9)
cbar9.set_label('Frequency', fontweight='bold')

# Overall title with proper positioning and formatting
fig.suptitle('Comprehensive EMG-Cyberglove Correlation Analysis', 
             fontsize=16, fontweight='bold', y=0.96)

# Adjust layout with proper spacing
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
plt.show()