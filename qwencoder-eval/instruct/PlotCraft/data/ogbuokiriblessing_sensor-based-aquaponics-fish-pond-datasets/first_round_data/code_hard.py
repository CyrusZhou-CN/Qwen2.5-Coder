import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Load and process the main dataset
def load_main_dataset():
    try:
        # Try to load the first available dataset
        df = pd.read_csv('IoTPond10.csv')
        print(f"Loaded IoTPond10.csv with shape: {df.shape}")
        
        # Standardize column names
        df.columns = ['created_at', 'entry_id', 'Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Population', 'Length', 'Weight']
        df['pond_id'] = 10
        
        # Try to load additional datasets if available
        additional_datasets = []
        
        try:
            df6 = pd.read_csv('IoTPond6.csv')
            df6.columns = ['created_at', 'entry_id', 'Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Population', 'Length', 'Weight']
            df6['pond_id'] = 6
            additional_datasets.append(df6)
            print(f"Loaded IoTPond6.csv with shape: {df6.shape}")
        except:
            print("IoTPond6.csv not found, continuing with available data")
        
        try:
            df3 = pd.read_csv('IoTPond3.csv')
            df3.columns = ['created_at', 'entry_id', 'Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Population', 'Length', 'Weight']
            df3['pond_id'] = 3
            additional_datasets.append(df3)
            print(f"Loaded IoTPond3.csv with shape: {df3.shape}")
        except:
            print("IoTPond3.csv not found, continuing with available data")
        
        try:
            df1 = pd.read_csv('IoTpond1.csv')
            df1.columns = ['created_at', 'entry_id', 'Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Population', 'Length', 'Weight']
            df1['pond_id'] = 1
            additional_datasets.append(df1)
            print(f"Loaded IoTpond1.csv with shape: {df1.shape}")
        except:
            print("IoTpond1.csv not found, continuing with available data")
        
        # Combine datasets if additional ones were loaded
        if additional_datasets:
            all_datasets = [df] + additional_datasets
            df = pd.concat(all_datasets, ignore_index=True)
            print(f"Combined dataset shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        # Create synthetic data as fallback
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'Temperature': np.random.normal(25, 2, n_samples),
            'Turbidity': np.random.randint(20, 100, n_samples),
            'Dissolved_Oxygen': np.random.normal(8, 2, n_samples),
            'pH': np.random.normal(7.5, 0.5, n_samples),
            'Ammonia': np.random.exponential(2, n_samples),
            'Nitrate': np.random.randint(100, 300, n_samples),
            'Population': np.random.randint(40, 80, n_samples),
            'Length': np.random.normal(7, 1, n_samples),
            'Weight': np.random.normal(3.5, 0.8, n_samples),
            'pond_id': np.random.choice([1, 2, 3, 4], n_samples)
        })
        print("Using synthetic data as fallback")
        return df

# Load data
df = load_main_dataset()

# Clean data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Remove extreme outliers
numeric_cols = ['Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate', 'Population', 'Length', 'Weight']
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df = df[(df[col] >= Q1) & (df[col] <= Q3)]

print(f"Final cleaned dataset shape: {df.shape}")

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Prepare water quality parameters for clustering
water_quality_params = ['Temperature', 'Turbidity', 'Dissolved_Oxygen', 'pH', 'Ammonia', 'Nitrate']
wq_data = df[water_quality_params].copy()

# Standardize data for clustering
scaler = StandardScaler()
wq_scaled = scaler.fit_transform(wq_data)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['pond_cluster'] = kmeans.fit_predict(wq_scaled)

# Create population size categories
df['pop_category'] = pd.cut(df['Population'], bins=[0, 60, 70, np.inf], labels=['Small (<60)', 'Medium (60-70)', 'Large (>70)'])

# Calculate pond efficiency
df['efficiency'] = df['Weight'] / df['Population']

# Define colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Row 1: Pond Clustering Analysis

# Subplot 1: Scatter plot with KDE contours (Temperature vs pH)
ax1 = plt.subplot(3, 3, 1)
for i, cluster in enumerate(sorted(df['pond_cluster'].unique())):
    cluster_data = df[df['pond_cluster'] == cluster]
    ax1.scatter(cluster_data['Temperature'], cluster_data['pH'], 
               c=colors[i], alpha=0.6, s=30, label=f'Cluster {cluster}')

# Add simple contours
try:
    from scipy.stats import gaussian_kde
    x = df['Temperature'].values
    y = df['pH'].values
    
    # Create a grid for contour plotting
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.mgrid[x_min:x_max:(x_max-x_min)/20, y_min:y_max:(y_max-y_min)/20]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax1.contour(xx, yy, f, colors='gray', alpha=0.3, linewidths=0.8)
except:
    pass

ax1.set_xlabel('Temperature (°C)', fontweight='bold')
ax1.set_ylabel('pH', fontweight='bold')
ax1.set_title('Temperature vs pH with Pond Clusters', fontweight='bold', fontsize=12)
ax1.legend(frameon=True, fancybox=True)
ax1.grid(True, alpha=0.3)

# Subplot 2: Parallel coordinates plot
ax2 = plt.subplot(3, 3, 2)
cluster_means = df.groupby('pond_cluster')[water_quality_params].mean()

# Normalize data for parallel coordinates
normalized_means = pd.DataFrame()
for param in water_quality_params:
    param_min, param_max = df[param].min(), df[param].max()
    if param_max != param_min:
        normalized_means[param] = (cluster_means[param] - param_min) / (param_max - param_min)
    else:
        normalized_means[param] = 0.5

# Plot parallel coordinates
for i, cluster in enumerate(sorted(df['pond_cluster'].unique())):
    ax2.plot(range(len(water_quality_params)), normalized_means.loc[cluster].values, 
            color=colors[i], linewidth=3, marker='o', markersize=8, 
            label=f'Cluster {cluster}', alpha=0.8)

ax2.set_xticks(range(len(water_quality_params)))
ax2.set_xticklabels([param.replace('_', ' ') for param in water_quality_params], rotation=45, ha='right')
ax2.set_ylabel('Normalized Values', fontweight='bold')
ax2.set_title('Water Quality Parameter Profiles by Cluster', fontweight='bold', fontsize=12)
ax2.legend(frameon=True, fancybox=True)
ax2.grid(True, alpha=0.3)

# Subplot 3: Radar chart
ax3 = plt.subplot(3, 3, 3, projection='polar')
angles = np.linspace(0, 2 * np.pi, len(water_quality_params), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, cluster in enumerate(sorted(df['pond_cluster'].unique())):
    values = normalized_means.loc[cluster].tolist()
    values += values[:1]
    
    ax3.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', 
            color=colors[i], markersize=6)
    ax3.fill(angles, values, alpha=0.25, color=colors[i])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels([param.replace('_', ' ') for param in water_quality_params])
ax3.set_title('Water Quality Radar Chart by Cluster', fontweight='bold', fontsize=12, pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Row 2: Fish Population Groupings

# Subplot 4: Bubble scatter plot
ax4 = plt.subplot(3, 3, 4)
for i, cluster in enumerate(sorted(df['pond_cluster'].unique())):
    cluster_data = df[df['pond_cluster'] == cluster]
    scatter = ax4.scatter(cluster_data['Length'], cluster_data['Weight'], 
                         s=cluster_data['Population']*2, c=colors[i], 
                         alpha=0.6, label=f'Cluster {cluster}')
    
    # Add regression line
    if len(cluster_data) > 1:
        try:
            z = np.polyfit(cluster_data['Length'], cluster_data['Weight'], 1)
            p = np.poly1d(z)
            x_reg = np.linspace(cluster_data['Length'].min(), cluster_data['Length'].max(), 100)
            ax4.plot(x_reg, p(x_reg), color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
        except:
            pass

ax4.set_xlabel('Fish Length (cm)', fontweight='bold')
ax4.set_ylabel('Fish Weight (g)', fontweight='bold')
ax4.set_title('Fish Length vs Weight by Pond Clusters', fontweight='bold', fontsize=12)
ax4.legend(frameon=True, fancybox=True)
ax4.grid(True, alpha=0.3)

# Subplot 5: Violin plots with strip plots
ax5 = plt.subplot(3, 3, 5)
pop_categories = df['pop_category'].dropna().unique()
if len(pop_categories) > 0:
    positions = np.arange(len(pop_categories))
    
    # Create violin plots for Length
    length_data = [df[df['pop_category'] == cat]['Length'].dropna().values for cat in pop_categories]
    length_data = [data for data in length_data if len(data) > 0]  # Remove empty arrays
    
    if length_data:
        try:
            parts = ax5.violinplot(length_data, positions=positions-0.2, widths=0.3, showmeans=True)
            for pc, color in zip(parts['bodies'], ['#FF9999', '#66B2FF', '#99FF99'][:len(parts['bodies'])]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        except:
            pass
    
    # Create violin plots for Weight
    weight_data = [df[df['pop_category'] == cat]['Weight'].dropna().values for cat in pop_categories]
    weight_data = [data for data in weight_data if len(data) > 0]  # Remove empty arrays
    
    if weight_data:
        try:
            parts2 = ax5.violinplot(weight_data, positions=positions+0.2, widths=0.3, showmeans=True)
            for pc, color in zip(parts2['bodies'], ['#FF6666', '#3399FF', '#66FF66'][:len(parts2['bodies'])]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        except:
            pass
    
    ax5.set_xticks(positions)
    ax5.set_xticklabels(pop_categories)

ax5.set_xlabel('Population Category', fontweight='bold')
ax5.set_ylabel('Fish Characteristics', fontweight='bold')
ax5.set_title('Fish Characteristics by Population Size', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Subplot 6: Correlation heatmap
ax6 = plt.subplot(3, 3, 6)
fish_wq_params = ['Length', 'Weight', 'Temperature', 'pH', 'Dissolved_Oxygen', 'Ammonia']
corr_matrix = df[fish_wq_params].corr()

# Create heatmap
im = ax6.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax6.set_xticks(range(len(corr_matrix.columns)))
ax6.set_yticks(range(len(corr_matrix.columns)))
ax6.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax6.set_yticklabels(corr_matrix.columns)
ax6.set_title('Fish-Water Quality Correlation Matrix', fontweight='bold', fontsize=12)

# Add correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

# Row 3: Water Quality Network Analysis

# Subplot 7: Network graph on correlation matrix
ax7 = plt.subplot(3, 3, 7)
wq_corr = df[water_quality_params].corr()
im7 = ax7.imshow(wq_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Add network connections for strong correlations
threshold = 0.3
for i in range(len(water_quality_params)):
    for j in range(i+1, len(water_quality_params)):
        if abs(wq_corr.iloc[i, j]) > threshold:
            ax7.plot([j, i], [i, j], 'k-', linewidth=abs(wq_corr.iloc[i, j])*3, alpha=0.6)

ax7.set_xticks(range(len(water_quality_params)))
ax7.set_yticks(range(len(water_quality_params)))
ax7.set_xticklabels([param.replace('_', ' ') for param in water_quality_params], rotation=45, ha='right')
ax7.set_yticklabels([param.replace('_', ' ') for param in water_quality_params])
ax7.set_title('Water Quality Parameter Network', fontweight='bold', fontsize=12)

# Subplot 8: Time series with filled areas
ax8 = plt.subplot(3, 3, 8)
# Sample time series data (using entry_id as proxy for time)
if 'entry_id' in df.columns:
    sample_data = df.sample(n=min(500, len(df))).sort_values('entry_id')
else:
    sample_data = df.sample(n=min(500, len(df))).reset_index()
    sample_data['entry_id'] = sample_data.index

# Plot main parameters
params_to_plot = ['Temperature', 'pH', 'Dissolved_Oxygen']
colors_ts = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, param in enumerate(params_to_plot):
    # Normalize for comparison
    param_min, param_max = sample_data[param].min(), sample_data[param].max()
    if param_max != param_min:
        normalized = (sample_data[param] - param_min) / (param_max - param_min)
    else:
        normalized = pd.Series([0.5] * len(sample_data))
    
    ax8.plot(sample_data['entry_id'], normalized, color=colors_ts[i], linewidth=2, label=param, alpha=0.8)
    
    # Add filled area showing range
    try:
        rolling_mean = normalized.rolling(window=min(20, len(normalized)//5), center=True).mean()
        rolling_std = normalized.rolling(window=min(20, len(normalized)//5), center=True).std()
        ax8.fill_between(sample_data['entry_id'], 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        color=colors_ts[i], alpha=0.2)
    except:
        pass

ax8.set_xlabel('Entry ID (Time Proxy)', fontweight='bold')
ax8.set_ylabel('Normalized Values', fontweight='bold')
ax8.set_title('Water Quality Trends Over Time', fontweight='bold', fontsize=12)
ax8.legend(frameon=True, fancybox=True)
ax8.grid(True, alpha=0.3)

# Subplot 9: 3D scatter projected to 2D with density contours
ax9 = plt.subplot(3, 3, 9)

# Use the three most variable parameters for PCA
param_vars = df[water_quality_params].var()
top_3_params = param_vars.nlargest(3).index.tolist()

# Create efficiency clusters
eff_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['eff_cluster'] = eff_kmeans.fit_predict(df[['efficiency']])

# Create 2D projection using PCA
pca_data = df[top_3_params].dropna()
if len(pca_data) > 0:
    pca = PCA(n_components=2)
    pca_scaled = StandardScaler().fit_transform(pca_data)
    projected = pca.fit_transform(pca_scaled)
    
    # Plot with efficiency clusters
    eff_colors = ['#E74C3C', '#F39C12', '#27AE60']
    for i in range(3):
        mask = df['eff_cluster'] == i
        if mask.sum() > 0:
            mask_indices = np.where(mask)[0]
            valid_indices = mask_indices[mask_indices < len(projected)]
            if len(valid_indices) > 0:
                ax9.scatter(projected[valid_indices, 0], projected[valid_indices, 1], 
                           c=eff_colors[i], alpha=0.6, s=30, label=f'Efficiency Cluster {i}')
    
    # Add density contours
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(projected.T)
        x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
        y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:(x_max-x_min)/10, y_min:y_max:(y_max-y_min)/10]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kde(positions).T, xx.shape)
        ax9.contour(xx, yy, f, colors='gray', alpha=0.4, linewidths=0.8)
    except:
        pass

ax9.set_xlabel(f'PC1 ({", ".join(top_3_params)})', fontweight='bold')
ax9.set_ylabel('PC2', fontweight='bold')
ax9.set_title('3D Water Quality PCA with Efficiency Clusters', fontweight='bold', fontsize=12)
ax9.legend(frameon=True, fancybox=True)
ax9.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=3.0)
plt.savefig('aquaponics_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()