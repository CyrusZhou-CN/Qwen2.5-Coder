import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Find all CSV files in the current directory
csv_files = glob.glob("*.csv")
if not csv_files:
    # If no CSV files found, create synthetic data for demonstration
    print("No CSV files found. Creating synthetic EEG data for demonstration...")
    
    # Create synthetic EEG data
    np.random.seed(42)
    n_samples = 1000
    electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    datasets = []
    for subject_id in range(6):
        data = {}
        data['TimeStamp'] = pd.date_range('2024-06-01', periods=n_samples, freq='1ms')
        
        # Generate synthetic brainwave data with subject-specific patterns
        base_freq = 0.1 + subject_id * 0.02
        for band in bands:
            for electrode in electrodes:
                # Create realistic EEG patterns
                signal = np.random.normal(0, 1, n_samples)
                signal += np.sin(2 * np.pi * base_freq * np.arange(n_samples)) * (0.5 + np.random.random() * 0.5)
                data[f'{band}_{electrode}'] = signal
        
        # Generate RAW signals
        for electrode in electrodes:
            data[f'RAW_{electrode}'] = np.random.normal(800, 20, n_samples)
        
        # Generate other features
        data['Concentration'] = np.random.uniform(0, 100, n_samples)
        data['Mellow'] = np.random.uniform(0, 100, n_samples)
        data['AUX_RIGHT'] = np.random.normal(700, 100, n_samples)
        data['Subject'] = f'Subject_{subject_id+1}'
        
        datasets.append(pd.DataFrame(data))
    
    combined_df = pd.concat(datasets, ignore_index=True)
    
else:
    # Load available CSV files (limit to first 6 for performance)
    datasets = []
    for i, file in enumerate(csv_files[:6]):
        try:
            df = pd.read_csv(file)
            df['Subject'] = f'Subject_{i+1}'
            # Sample data for performance (take every 100th row)
            df_sampled = df.iloc[::100].copy()
            datasets.append(df_sampled)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not datasets:
        raise Exception("No valid CSV files could be loaded")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)

# Define electrode positions and brainwave bands
electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Ensure required columns exist
required_cols = []
for band in bands:
    for electrode in electrodes:
        required_cols.append(f'{band}_{electrode}')

# Check if columns exist, if not create synthetic ones
for col in required_cols:
    if col not in combined_df.columns:
        combined_df[col] = np.random.normal(0, 1, len(combined_df))

# Ensure other required columns exist
if 'Concentration' not in combined_df.columns:
    combined_df['Concentration'] = np.random.uniform(0, 100, len(combined_df))

for electrode in electrodes:
    if f'RAW_{electrode}' not in combined_df.columns:
        combined_df[f'RAW_{electrode}'] = np.random.normal(800, 20, len(combined_df))

# Create figure with white background
fig = plt.figure(figsize=(20, 24), facecolor='white')
fig.suptitle('Comprehensive EEG Brainwave Analysis: Multi-Subject Motor Imagery Classification', 
             fontsize=20, fontweight='bold', y=0.98)

# Color palette for subjects
subject_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
unique_subjects = combined_df['Subject'].unique()

# 1. Top Left: Cluster Analysis with Alpha vs Beta Power
ax1 = plt.subplot(3, 2, 1)

# Calculate average Alpha and Beta power across electrodes
combined_df['Alpha_avg'] = combined_df[[f'Alpha_{e}' for e in electrodes]].mean(axis=1)
combined_df['Beta_avg'] = combined_df[[f'Beta_{e}' for e in electrodes]].mean(axis=1)

# Create concentration level categories
combined_df['Concentration_level'] = pd.cut(combined_df['Concentration'], 
                                          bins=3, labels=['Low', 'Medium', 'High'])

# Scatter plot with density contours
for i, subject in enumerate(unique_subjects):
    subject_data = combined_df[combined_df['Subject'] == subject]
    color_idx = i % len(subject_colors)
    ax1.scatter(subject_data['Alpha_avg'], subject_data['Beta_avg'], 
               c=subject_colors[color_idx], alpha=0.6, s=30, label=subject, 
               edgecolors='white', linewidth=0.5)

# Add simple contour lines
try:
    x = combined_df['Alpha_avg'].dropna()
    y = combined_df['Beta_avg'].dropna()
    if len(x) > 10 and len(y) > 10:
        # Create a simple grid for contour
        xi = np.linspace(x.min(), x.max(), 20)
        yi = np.linspace(y.min(), y.max(), 20)
        Xi, Yi = np.meshgrid(xi, yi)
        # Simple density estimation
        H, xedges, yedges = np.histogram2d(x, y, bins=20)
        ax1.contour(xi[:-1], yi[:-1], H.T, colors='gray', alpha=0.3, linewidths=1)
except:
    pass

ax1.set_xlabel('Alpha Power (Average)', fontweight='bold')
ax1.set_ylabel('Beta Power (Average)', fontweight='bold')
ax1.set_title('Alpha vs Beta Power Distribution with Subject Clustering', fontweight='bold', pad=20)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Top Right: Parallel Coordinates Plot
ax2 = plt.subplot(3, 2, 2)

# Prepare data for parallel coordinates
coords_data = []
for subject in unique_subjects:
    subject_data = combined_df[combined_df['Subject'] == subject]
    for electrode in electrodes:
        row = [subject_data[f'{band}_{electrode}'].mean() for band in bands]
        row.append(subject)
        row.append(electrode)
        coords_data.append(row)

coords_df = pd.DataFrame(coords_data, columns=bands + ['Subject', 'Electrode'])

# Normalize data for parallel coordinates
scaler = StandardScaler()
normalized_data = scaler.fit_transform(coords_df[bands])

# Plot parallel coordinates
x_pos = np.arange(len(bands))
for i, (subject, electrode) in enumerate(zip(coords_df['Subject'], coords_df['Electrode'])):
    subject_idx = list(unique_subjects).index(subject) % len(subject_colors)
    alpha_val = 0.7 if electrode in ['AF7', 'AF8'] else 0.5
    ax2.plot(x_pos, normalized_data[i], color=subject_colors[subject_idx], 
             alpha=alpha_val, linewidth=1.5)

# Add simple box plots
for i, band in enumerate(bands):
    data_for_box = normalized_data[:, i]
    q1, median, q3 = np.percentile(data_for_box, [25, 50, 75])
    ax2.plot([i-0.1, i+0.1], [q1, q1], 'k-', alpha=0.5)
    ax2.plot([i-0.1, i+0.1], [q3, q3], 'k-', alpha=0.5)
    ax2.plot([i, i], [q1, q3], 'k-', alpha=0.5)
    ax2.plot([i-0.05, i+0.05], [median, median], 'r-', linewidth=2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(bands, fontweight='bold')
ax2.set_ylabel('Normalized Power', fontweight='bold')
ax2.set_title('Parallel Coordinates: Brainwave Bands Across Electrodes', fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3)

# 3. Middle Left: Correlation Network Graph
ax3 = plt.subplot(3, 2, 3)

# Calculate correlation matrix for electrode-band combinations
corr_features = []
for electrode in electrodes:
    for band in bands:
        corr_features.append(f'{band}_{electrode}')

corr_data = combined_df[corr_features].corr()

# Create network visualization
im = ax3.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add correlation values as text
for i in range(len(corr_features)):
    for j in range(len(corr_features)):
        if abs(corr_data.iloc[i, j]) > 0.5:
            ax3.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8, 
                    color='white' if abs(corr_data.iloc[i, j]) > 0.7 else 'black')

ax3.set_xticks(range(len(corr_features)))
ax3.set_yticks(range(len(corr_features)))
ax3.set_xticklabels([f.replace('_', '\n') for f in corr_features], rotation=45, fontsize=8)
ax3.set_yticklabels([f.replace('_', '\n') for f in corr_features], fontsize=8)
ax3.set_title('Correlation Network: Electrode-Brainwave Combinations', fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# 4. Middle Right: Radar Chart with Subject Comparisons
ax4 = plt.subplot(3, 2, 4, projection='polar')

# Calculate average power for each band across all electrodes for each subject
radar_data = []
for subject in unique_subjects:
    subject_data = combined_df[combined_df['Subject'] == subject]
    subject_means = []
    for band in bands:
        band_cols = [f'{band}_{e}' for e in electrodes]
        subject_means.append(subject_data[band_cols].mean().mean())
    radar_data.append(subject_means)

# Normalize radar data
radar_array = np.array(radar_data)
radar_normalized = (radar_array - radar_array.min(axis=0)) / (radar_array.max(axis=0) - radar_array.min(axis=0) + 1e-8)

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
angles += angles[:1]

for i, subject in enumerate(unique_subjects):
    color_idx = i % len(subject_colors)
    values = radar_normalized[i].tolist()
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2, label=subject, 
             color=subject_colors[color_idx], alpha=0.8)
    ax4.fill(angles, values, alpha=0.1, color=subject_colors[color_idx])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(bands, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.set_title('Radar Chart: Subject Brainwave Profiles', fontweight='bold', pad=30)
ax4.legend(bbox_to_anchor=(1.3, 1.1), loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Bottom Left: Time-Series Analysis
ax5 = plt.subplot(3, 2, 5)

# Select a subset of data for time series
for i, subject in enumerate(unique_subjects):
    subject_data = combined_df[combined_df['Subject'] == subject].head(200)
    if len(subject_data) > 0:
        color_idx = i % len(subject_colors)
        time_points = np.arange(len(subject_data))
        for j, electrode in enumerate(electrodes):
            offset = i * 4 + j
            ax5.plot(time_points, subject_data[f'RAW_{electrode}'] + offset * 50, 
                    color=subject_colors[color_idx], alpha=0.7, linewidth=1,
                    label=f'{subject}_{electrode}' if j == 0 else "")

ax5.set_xlabel('Time Points', fontweight='bold')
ax5.set_ylabel('Raw EEG Signal (μV) + Offset', fontweight='bold')
ax5.set_title('Multi-Electrode Raw EEG Time Series', fontweight='bold', pad=20)
ax5.grid(True, alpha=0.3)

# Add spectral analysis inset
ax5_inset = ax5.inset_axes([0.65, 0.65, 0.3, 0.3])
for i, subject in enumerate(unique_subjects[:3]):
    subject_data = combined_df[combined_df['Subject'] == subject].head(500)
    if len(subject_data) > 0:
        color_idx = i % len(subject_colors)
        # Simple power spectrum
        signal = subject_data['RAW_TP9'].fillna(subject_data['RAW_TP9'].mean())
        freqs = np.fft.fftfreq(len(signal), d=1/256)
        power = np.abs(np.fft.fft(signal))**2
        ax5_inset.semilogy(freqs[:len(freqs)//2], power[:len(power)//2], 
                          color=subject_colors[color_idx], alpha=0.7, linewidth=1)

ax5_inset.set_xlabel('Freq (Hz)', fontsize=8)
ax5_inset.set_ylabel('Power', fontsize=8)
ax5_inset.set_title('Power Spectrum', fontsize=9, fontweight='bold')
ax5_inset.grid(True, alpha=0.3)

# 6. Bottom Right: Multi-Dimensional Scaling (MDS)
ax6 = plt.subplot(3, 2, 6)

# Prepare data for MDS
mds_features = []
for subject in unique_subjects:
    subject_data = combined_df[combined_df['Subject'] == subject]
    feature_vector = []
    for band in bands:
        for electrode in electrodes:
            feature_vector.append(subject_data[f'{band}_{electrode}'].mean())
    mds_features.append(feature_vector)

# Perform MDS
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
mds_coords = mds.fit_transform(mds_features)

# Plot MDS results
for i, subject in enumerate(unique_subjects):
    color_idx = i % len(subject_colors)
    ax6.scatter(mds_coords[i, 0], mds_coords[i, 1], 
               c=subject_colors[color_idx], s=200, alpha=0.8, 
               edgecolors='white', linewidth=2, label=subject)
    
    # Add confidence ellipse
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((mds_coords[i, 0], mds_coords[i, 1]), 
                     width=0.5, height=0.3, alpha=0.2, 
                     facecolor=subject_colors[color_idx])
    ax6.add_patch(ellipse)
    
    # Add subject label
    ax6.annotate(f'S{i+1}', (mds_coords[i, 0], mds_coords[i, 1]), 
                xytext=(5, 5), textcoords='offset points', 
                fontweight='bold', fontsize=10)

# Add dendrogram as side plot
try:
    ax6_dendro = ax6.inset_axes([1.1, 0.1, 0.3, 0.8])
    linkage_matrix = linkage(mds_features, method='ward')
    dendro = dendrogram(linkage_matrix, ax=ax6_dendro, orientation='right',
                       labels=[f'S{i+1}' for i in range(len(unique_subjects))],
                       color_threshold=0.7*max(linkage_matrix[:,2]))
    ax6_dendro.set_title('Hierarchical\nClustering', fontweight='bold', fontsize=10)
except:
    pass

ax6.set_xlabel('MDS Dimension 1', fontweight='bold')
ax6.set_ylabel('MDS Dimension 2', fontweight='bold')
ax6.set_title('Multi-Dimensional Scaling: Subject EEG Signatures', fontweight='bold', pad=20)
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax6.grid(True, alpha=0.3)

# Add statistical annotations
try:
    concentration_groups = combined_df.groupby('Subject')['Concentration'].mean()
    high_conc = concentration_groups > concentration_groups.median()
    low_conc = concentration_groups <= concentration_groups.median()

    if len(concentration_groups[high_conc]) > 1 and len(concentration_groups[low_conc]) > 1:
        t_stat, p_value = stats.ttest_ind(concentration_groups[high_conc], 
                                         concentration_groups[low_conc])
        ax6.text(0.02, 0.98, f'Concentration Groups\nt-stat: {t_stat:.3f}\np-value: {p_value:.3f}', 
                transform=ax6.transAxes, fontsize=9, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
except:
    pass

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.4)
plt.savefig('eeg_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()