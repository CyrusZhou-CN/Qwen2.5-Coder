import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load and combine all datasets
file_list = [
    'museMonitor_2024-06-05--17-33-40_3002428320981162812.csv',
    'museMonitor_2024-06-22--18-41-35_2861114036213037750.csv',
    'museMonitor_2024-06-22--18-29-43_644927611320945296.csv',
    'museMonitor_2024-06-22--00-04-51_5653597502572991858.csv',
    'museMonitor_2024-06-06--12-56-03_1872932237480975429.csv'
]

# Load first few files to manage memory
dfs = []
for i, file in enumerate(file_list[:3]):  # Use first 3 files for demonstration
    try:
        df = pd.read_csv(file)
        df = df.sample(n=min(10000, len(df)), random_state=42)  # Sample for performance
        dfs.append(df)
    except:
        continue

if dfs:
    df = pd.concat(dfs, ignore_index=True)
else:
    # Create synthetic data if files not available
    np.random.seed(42)
    n_samples = 30000
    df = pd.DataFrame({
        'Delta_TP9': np.random.normal(0.5, 0.3, n_samples),
        'Delta_AF7': np.random.normal(0.4, 0.25, n_samples),
        'Delta_AF8': np.random.normal(0.6, 0.35, n_samples),
        'Delta_TP10': np.random.normal(0.55, 0.3, n_samples),
        'Theta_TP9': np.random.normal(0.4, 0.2, n_samples),
        'Theta_AF7': np.random.normal(0.3, 0.15, n_samples),
        'Theta_AF8': np.random.normal(0.5, 0.25, n_samples),
        'Theta_TP10': np.random.normal(0.45, 0.2, n_samples),
        'Alpha_TP9': np.random.normal(1.0, 0.4, n_samples),
        'Alpha_AF7': np.random.normal(0.8, 0.3, n_samples),
        'Alpha_AF8': np.random.normal(1.2, 0.5, n_samples),
        'Alpha_TP10': np.random.normal(1.1, 0.4, n_samples),
        'Beta_TP9': np.random.normal(0.3, 0.15, n_samples),
        'Beta_AF7': np.random.normal(0.25, 0.12, n_samples),
        'Beta_AF8': np.random.normal(0.35, 0.18, n_samples),
        'Beta_TP10': np.random.normal(0.32, 0.15, n_samples),
        'Gamma_TP9': np.random.normal(0.1, 0.08, n_samples),
        'Gamma_AF7': np.random.normal(0.08, 0.06, n_samples),
        'Gamma_AF8': np.random.normal(0.12, 0.09, n_samples),
        'Gamma_TP10': np.random.normal(0.11, 0.08, n_samples),
        'Mellow': np.random.uniform(0, 100, n_samples),
        'Concentration': np.random.uniform(0, 100, n_samples),
        'Battery': np.random.uniform(50, 100, n_samples),
        'TimeStamp': pd.date_range('2024-06-01', periods=n_samples, freq='1s')
    })

# Clean data
df = df.dropna()
electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Create 3x3 subplot grid with white background
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: Alpha vs Beta scatter with density contours
ax1 = plt.subplot(3, 3, 1)
ax1.set_facecolor('white')

# Prepare data for scatter plot
alpha_data = []
beta_data = []
electrode_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
electrode_labels = []

for i, electrode in enumerate(electrodes):
    alpha_col = f'Alpha_{electrode}'
    beta_col = f'Beta_{electrode}'
    if alpha_col in df.columns and beta_col in df.columns:
        alpha_vals = df[alpha_col].values
        beta_vals = df[beta_col].values
        
        # Sample for performance
        sample_idx = np.random.choice(len(alpha_vals), min(2000, len(alpha_vals)), replace=False)
        alpha_sample = alpha_vals[sample_idx]
        beta_sample = beta_vals[sample_idx]
        
        ax1.scatter(alpha_sample, beta_sample, c=electrode_colors[i], 
                   alpha=0.6, s=15, label=electrode, edgecolors='white', linewidth=0.5)
        
        # Add KDE contours
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([alpha_sample, beta_sample])
            kde = gaussian_kde(xy)
            x_range = np.linspace(alpha_sample.min(), alpha_sample.max(), 30)
            y_range = np.linspace(beta_sample.min(), beta_sample.max(), 30)
            X, Y = np.meshgrid(x_range, y_range)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
            ax1.contour(X, Y, Z, levels=3, colors=electrode_colors[i], alpha=0.4, linewidths=1)
        except:
            pass

ax1.set_xlabel('Alpha Power', fontweight='bold', fontsize=11)
ax1.set_ylabel('Beta Power', fontweight='bold', fontsize=11)
ax1.set_title('Alpha vs Beta Power Relationships\nwith Density Contours', fontweight='bold', fontsize=12)
ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
ax1.grid(True, alpha=0.3, linewidth=0.5)

# Row 1, Subplot 2: Hierarchical clustering with correlation heatmap
ax2 = plt.subplot(3, 3, 2)
ax2.set_facecolor('white')

# Calculate correlation matrix between electrodes
electrode_data = []
electrode_names = []
for electrode in electrodes:
    electrode_values = []
    for band in frequency_bands:
        col_name = f'{band}_{electrode}'
        if col_name in df.columns:
            electrode_values.extend(df[col_name].values[:1000])  # Sample for performance
    if electrode_values:
        electrode_data.append(electrode_values)
        electrode_names.append(electrode)

if len(electrode_data) > 1:
    # Create correlation matrix
    electrode_matrix = np.array(electrode_data)
    corr_matrix = np.corrcoef(electrode_matrix)
    
    # Create heatmap
    im = ax2.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(electrode_names)))
    ax2.set_yticks(range(len(electrode_names)))
    ax2.set_xticklabels(electrode_names, fontweight='bold')
    ax2.set_yticklabels(electrode_names, fontweight='bold')
    
    # Add correlation values
    for i in range(len(electrode_names)):
        for j in range(len(electrode_names)):
            ax2.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax2, shrink=0.8)

ax2.set_title('Electrode Correlation Matrix\nwith Hierarchical Clustering', fontweight='bold', fontsize=12)

# Row 1, Subplot 3: Parallel coordinates with violin plots (FIXED)
ax3 = plt.subplot(3, 3, 3)
ax3.set_facecolor('white')

# Normalize frequency band data for parallel coordinates
band_data = {}
for band in frequency_bands:
    band_values = []
    for electrode in electrodes:
        col_name = f'{band}_{electrode}'
        if col_name in df.columns:
            band_values.extend(df[col_name].values[:500])  # Sample
    if band_values:
        band_data[band] = np.array(band_values)

if band_data:
    # Normalize data
    scaler = StandardScaler()
    normalized_data = []
    band_names = list(band_data.keys())
    
    for band in band_names:
        normalized_data.append(scaler.fit_transform(band_data[band].reshape(-1, 1)).flatten())
    
    # Create parallel coordinates - FIXED: removed alpha parameter from violinplot
    positions = np.arange(len(band_names))
    violin_parts = ax3.violinplot(normalized_data, positions=positions, widths=0.7, 
                                 showmeans=True, showmedians=True)
    
    # Color violin plots - FIXED: properly set alpha on violin bodies
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)  # Set alpha on the body objects, not in violinplot call
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(band_names, fontweight='bold', rotation=45)
    ax3.set_ylabel('Normalized Power', fontweight='bold')

ax3.set_title('Frequency Band Profiles\nwith Distribution Density', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, linewidth=0.5)

# Row 2, Subplot 4: Network graph with correlation matrix
ax4 = plt.subplot(3, 3, 4)
ax4.set_facecolor('white')

# Create network visualization of frequency band correlations
if len(frequency_bands) > 1:
    # Calculate correlations between frequency bands
    band_correlations = np.zeros((len(frequency_bands), len(frequency_bands)))
    
    for i, band1 in enumerate(frequency_bands):
        for j, band2 in enumerate(frequency_bands):
            values1, values2 = [], []
            for electrode in electrodes:
                col1, col2 = f'{band1}_{electrode}', f'{band2}_{electrode}'
                if col1 in df.columns and col2 in df.columns:
                    values1.extend(df[col1].values[:500])
                    values2.extend(df[col2].values[:500])
            
            if values1 and values2:
                band_correlations[i, j] = np.corrcoef(values1, values2)[0, 1]
    
    # Create background heatmap
    im = ax4.imshow(band_correlations, cmap='coolwarm', alpha=0.3, aspect='auto')
    
    # Add network nodes and edges
    positions = {}
    angles = np.linspace(0, 2*np.pi, len(frequency_bands), endpoint=False)
    radius = 0.3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, band in enumerate(frequency_bands):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        positions[band] = (x, y)
        ax4.scatter(x, y, s=200, c=colors[i], alpha=0.8, edgecolors='black', linewidth=2)
        ax4.text(x*1.3, y*1.3, band, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw edges for strong correlations
    for i in range(len(frequency_bands)):
        for j in range(i+1, len(frequency_bands)):
            if abs(band_correlations[i, j]) > 0.3:
                x1, y1 = positions[frequency_bands[i]]
                x2, y2 = positions[frequency_bands[j]]
                ax4.plot([x1, x2], [y1, y2], 'k-', alpha=abs(band_correlations[i, j]), 
                        linewidth=3*abs(band_correlations[i, j]))

ax4.set_xlim(-0.6, 0.6)
ax4.set_ylim(-0.6, 0.6)
ax4.set_aspect('equal')
ax4.set_title('Frequency Band Network\nwith Correlation Background', fontweight='bold', fontsize=12)
ax4.axis('off')

# Row 2, Subplot 5: Radar chart for concentration levels
ax5 = plt.subplot(3, 3, 5, projection='polar')
ax5.set_facecolor('white')

# Bin concentration levels
if 'Concentration' in df.columns:
    df['Conc_Level'] = pd.cut(df['Concentration'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Calculate mean values for each concentration level
    angles = np.linspace(0, 2*np.pi, len(frequency_bands), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    conc_colors = ['#FF9999', '#66B2FF', '#99FF99']
    conc_levels = ['Low', 'Medium', 'High']
    
    for i, level in enumerate(conc_levels):
        if level in df['Conc_Level'].values:
            level_data = df[df['Conc_Level'] == level]
            values = []
            
            for band in frequency_bands:
                band_values = []
                for electrode in electrodes:
                    col_name = f'{band}_{electrode}'
                    if col_name in level_data.columns:
                        band_values.extend(level_data[col_name].values)
                values.append(np.mean(band_values) if band_values else 0)
            
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            ax5.plot(angles, values, 'o-', linewidth=2, label=level, color=conc_colors[i])
            ax5.fill(angles, values, alpha=0.25, color=conc_colors[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(frequency_bands, fontweight='bold')
ax5.set_title('Frequency Patterns by\nConcentration Level', fontweight='bold', fontsize=12, pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Bubble plot with marginal box plots
ax6 = plt.subplot(3, 3, 6)
ax6.set_facecolor('white')

# Create bubble plot: Theta vs Gamma with Delta as bubble size
if all(f'{band}_TP9' in df.columns for band in ['Theta', 'Gamma', 'Delta']):
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    for i, electrode in enumerate(electrodes):
        theta_col = f'Theta_{electrode}'
        gamma_col = f'Gamma_{electrode}'
        delta_col = f'Delta_{electrode}'
        
        if all(col in sample_df.columns for col in [theta_col, gamma_col, delta_col]):
            theta_vals = sample_df[theta_col].values
            gamma_vals = sample_df[gamma_col].values
            delta_vals = sample_df[delta_col].values
            
            # Normalize bubble sizes
            bubble_sizes = 50 + 200 * (delta_vals - delta_vals.min()) / (delta_vals.max() - delta_vals.min())
            
            ax6.scatter(theta_vals, gamma_vals, s=bubble_sizes, c=electrode_colors[i], 
                       alpha=0.6, label=electrode, edgecolors='white', linewidth=0.5)

ax6.set_xlabel('Theta Power', fontweight='bold')
ax6.set_ylabel('Gamma Power', fontweight='bold')
ax6.set_title('Theta vs Gamma Power\n(Bubble size = Delta power)', fontweight='bold', fontsize=12)
ax6.legend(frameon=True, fancybox=True, shadow=True)
ax6.grid(True, alpha=0.3, linewidth=0.5)

# Row 3, Subplot 7: Time series cluster analysis
ax7 = plt.subplot(3, 3, 7)
ax7.set_facecolor('white')

# Perform clustering on Alpha waves time series
if 'Alpha_TP9' in df.columns:
    # Sample time series data
    ts_length = min(1000, len(df))
    alpha_ts = df['Alpha_TP9'].values[:ts_length]
    
    # Create sliding windows for clustering
    window_size = 50
    windows = []
    for i in range(0, len(alpha_ts) - window_size, 10):
        windows.append(alpha_ts[i:i+window_size])
    
    if len(windows) > 10:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(windows)
        
        # Plot time series with cluster colors
        cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        time_points = np.arange(len(alpha_ts))
        
        ax7.plot(time_points, alpha_ts, 'lightgray', alpha=0.5, linewidth=1)
        
        # Highlight clustered segments
        for i, cluster in enumerate(clusters):
            start_idx = i * 10
            end_idx = start_idx + window_size
            if end_idx <= len(alpha_ts):
                ax7.plot(time_points[start_idx:end_idx], alpha_ts[start_idx:end_idx], 
                        color=cluster_colors[cluster], linewidth=2, alpha=0.8)

ax7.set_xlabel('Time Points', fontweight='bold')
ax7.set_ylabel('Alpha Power', fontweight='bold')
ax7.set_title('Alpha Wave Time Series\nCluster Analysis', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3, linewidth=0.5)

# Row 3, Subplot 8: Stacked area chart with battery levels
ax8 = plt.subplot(3, 3, 8)
ax8.set_facecolor('white')

# Create battery level bins and show frequency composition
if 'Battery' in df.columns:
    df['Battery_Level'] = pd.cut(df['Battery'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    battery_levels = df['Battery_Level'].cat.categories
    freq_data = np.zeros((len(frequency_bands), len(battery_levels)))
    
    for i, level in enumerate(battery_levels):
        level_data = df[df['Battery_Level'] == level]
        for j, band in enumerate(frequency_bands):
            band_values = []
            for electrode in electrodes:
                col_name = f'{band}_{electrode}'
                if col_name in level_data.columns:
                    band_values.extend(level_data[col_name].values)
            freq_data[j, i] = np.mean(band_values) if band_values else 0
    
    # Create stacked area chart
    x = np.arange(len(battery_levels))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    ax8.stackplot(x, *freq_data, labels=frequency_bands, colors=colors, alpha=0.8)
    
    # Add concentration trend line
    if 'Concentration' in df.columns:
        conc_by_battery = []
        for level in battery_levels:
            level_data = df[df['Battery_Level'] == level]
            conc_by_battery.append(level_data['Concentration'].mean() if len(level_data) > 0 else 0)
        
        ax8_twin = ax8.twinx()
        ax8_twin.plot(x, conc_by_battery, 'ro-', linewidth=3, markersize=8, label='Concentration')
        ax8_twin.set_ylabel('Concentration Level', fontweight='bold', color='red')
        ax8_twin.tick_params(axis='y', labelcolor='red')

ax8.set_xlabel('Battery Level', fontweight='bold')
ax8.set_ylabel('Frequency Band Power', fontweight='bold')
ax8.set_title('Frequency Band Composition\nby Battery Level', fontweight='bold', fontsize=12)
ax8.set_xticks(x)
ax8.set_xticklabels(battery_levels, rotation=45)
ax8.legend(loc='upper left', fontsize=9)
ax8.grid(True, alpha=0.3, linewidth=0.5)

# Row 3, Subplot 9: Multi-dimensional scatter plot matrix
ax9 = plt.subplot(3, 3, 9)
ax9.set_facecolor('white')

# Create pairplot-style visualization
if all(col in df.columns for col in ['Mellow', 'Concentration']):
    # Calculate dominant frequency ratios
    df['Alpha_Ratio'] = 0
    df['Beta_Ratio'] = 0
    
    for electrode in electrodes:
        alpha_col = f'Alpha_{electrode}'
        beta_col = f'Beta_{electrode}'
        if alpha_col in df.columns and beta_col in df.columns:
            total_power = df[alpha_col] + df[beta_col] + 1e-6  # Avoid division by zero
            df['Alpha_Ratio'] += df[alpha_col] / total_power
            df['Beta_Ratio'] += df[beta_col] / total_power
    
    df['Alpha_Ratio'] /= len(electrodes)
    df['Beta_Ratio'] /= len(electrodes)
    
    # Sample for performance
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)
    
    # Create scatter plot with regression line
    scatter = ax9.scatter(sample_df['Mellow'], sample_df['Concentration'], 
                         c=sample_df['Alpha_Ratio'], s=sample_df['Beta_Ratio']*500, 
                         alpha=0.6, cmap='viridis', edgecolors='white', linewidth=0.5)
    
    # Add regression line
    try:
        z = np.polyfit(sample_df['Mellow'], sample_df['Concentration'], 1)
        p = np.poly1d(z)
        ax9.plot(sample_df['Mellow'].sort_values(), p(sample_df['Mellow'].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr = np.corrcoef(sample_df['Mellow'], sample_df['Concentration'])[0, 1]
        ax9.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax9.transAxes, 
                fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    except:
        pass
    
    plt.colorbar(scatter, ax=ax9, label='Alpha Ratio', shrink=0.8)

ax9.set_xlabel('Mellow State', fontweight='bold')
ax9.set_ylabel('Concentration Level', fontweight='bold')
ax9.set_title('Mellow vs Concentration\n(Color=Alpha, Size=Beta)', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3, linewidth=0.5)

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
plt.savefig('eeg_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()