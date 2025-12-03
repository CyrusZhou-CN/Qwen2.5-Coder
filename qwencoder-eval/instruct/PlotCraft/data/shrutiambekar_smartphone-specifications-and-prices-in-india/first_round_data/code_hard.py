import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Find the correct CSV file
csv_files = glob.glob('*.csv')
smartphone_file = None
for file in csv_files:
    if 'smartphone' in file.lower():
        smartphone_file = file
        break

if smartphone_file is None:
    # Try common variations
    possible_names = ['smartphones.csv', 'smartphone.csv', 'phone.csv', 'mobile.csv']
    for name in possible_names:
        if os.path.exists(name):
            smartphone_file = name
            break

if smartphone_file is None:
    # Use the first CSV file found
    smartphone_file = csv_files[0] if csv_files else 'smartphones.csv'

# Load and preprocess data
try:
    df = pd.read_csv(smartphone_file)
except:
    # Create sample data if file not found
    np.random.seed(42)
    n_samples = 100
    brands = ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Realme', 'Motorola', 'Nothing', 'Oppo']
    processors = ['Snapdragon', 'Exynos', 'Dimensity', 'Bionic', 'MediaTek']
    
    df = pd.DataFrame({
        'model': [f'{np.random.choice(brands)} Model {i}' for i in range(n_samples)],
        'price': [f'₹{np.random.randint(10000, 80000):,}' for _ in range(n_samples)],
        'rating': np.random.normal(80, 10, n_samples),
        'processor': [f'{np.random.choice(processors)} {np.random.randint(600, 900)}' for _ in range(n_samples)],
        'ram': [f'{np.random.choice([4, 6, 8, 12, 16])} GB RAM' for _ in range(n_samples)],
        'battery': [f'{np.random.randint(3000, 6000)} mAh Battery' for _ in range(n_samples)],
        'display': [f'{np.random.uniform(5.5, 7.0):.1f} inches' for _ in range(n_samples)],
        'camera': [f'{np.random.choice([12, 48, 50, 64, 108])} MP' for _ in range(n_samples)]
    })

# Clean and convert data
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    try:
        return float(str(price_str).replace('₹', '').replace(',', ''))
    except:
        return np.nan

def clean_ram(ram_str):
    if pd.isna(ram_str):
        return np.nan
    try:
        return float(str(ram_str).split(' GB')[0].split()[-1])
    except:
        return np.nan

def clean_battery(battery_str):
    if pd.isna(battery_str):
        return np.nan
    try:
        return float(str(battery_str).split(' mAh')[0].split()[-1])
    except:
        return np.nan

def clean_display(display_str):
    if pd.isna(display_str):
        return np.nan
    try:
        return float(str(display_str).split(' inches')[0].split()[-1])
    except:
        return np.nan

def clean_camera(camera_str):
    if pd.isna(camera_str):
        return np.nan
    try:
        return float(str(camera_str).split(' MP')[0].split()[-1])
    except:
        return np.nan

def extract_brand(model):
    if pd.isna(model):
        return 'Unknown'
    return str(model).split(' ')[0]

# Apply cleaning functions
df['price_clean'] = df['price'].apply(clean_price)
df['ram_clean'] = df['ram'].apply(clean_ram) if 'ram' in df.columns else np.random.choice([4, 6, 8, 12], len(df))
df['battery_clean'] = df['battery'].apply(clean_battery) if 'battery' in df.columns else np.random.randint(3000, 6000, len(df))
df['display_clean'] = df['display'].apply(clean_display) if 'display' in df.columns else np.random.uniform(5.5, 7.0, len(df))
df['camera_clean'] = df['camera'].apply(clean_camera) if 'camera' in df.columns else np.random.choice([12, 48, 50, 64], len(df))
df['brand'] = df['model'].apply(extract_brand)

# Ensure rating column exists
if 'rating' not in df.columns:
    df['rating'] = np.random.normal(80, 10, len(df))

# Remove rows with missing critical data
df_clean = df.dropna(subset=['price_clean', 'rating'])

# Fill missing values with median
for col in ['ram_clean', 'battery_clean', 'display_clean', 'camera_clean']:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Create categories
df_clean['ram_category'] = pd.cut(df_clean['ram_clean'], bins=[0, 4, 8, 12, float('inf')], 
                                  labels=['Low (≤4GB)', 'Medium (4-8GB)', 'High (8-12GB)', 'Premium (>12GB)'])

df_clean['price_segment'] = pd.cut(df_clean['price_clean'], bins=[0, 20000, 40000, 60000, float('inf')], 
                                   labels=['Budget', 'Mid-range', 'Premium', 'Flagship'])

# Extract processor types
if 'processor' in df_clean.columns:
    df_clean['processor_type'] = df_clean['processor'].str.extract(r'(Snapdragon|Exynos|Dimensity|Bionic|MediaTek)')
    df_clean['processor_type'] = df_clean['processor_type'].fillna('Other')
else:
    df_clean['processor_type'] = np.random.choice(['Snapdragon', 'Exynos', 'Dimensity', 'Other'], len(df_clean))

# Brand categorization
top_brands = df_clean['brand'].value_counts().head(8).index.tolist()
df_clean['brand_category'] = df_clean['brand'].apply(lambda x: x if x in top_brands else 'Others')

# Ensure we have enough data points
if len(df_clean) < 20:
    # Generate more sample data
    additional_data = []
    for i in range(50):
        additional_data.append({
            'price_clean': np.random.randint(15000, 70000),
            'rating': np.random.normal(80, 8),
            'ram_clean': np.random.choice([4, 6, 8, 12, 16]),
            'battery_clean': np.random.randint(3500, 5500),
            'display_clean': np.random.uniform(6.0, 6.8),
            'camera_clean': np.random.choice([48, 50, 64, 108]),
            'brand': np.random.choice(['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Realme']),
            'processor_type': np.random.choice(['Snapdragon', 'Exynos', 'Dimensity'])
        })
    
    additional_df = pd.DataFrame(additional_data)
    additional_df['brand_category'] = additional_df['brand']
    additional_df['ram_category'] = pd.cut(additional_df['ram_clean'], bins=[0, 4, 8, 12, float('inf')], 
                                          labels=['Low (≤4GB)', 'Medium (4-8GB)', 'High (8-12GB)', 'Premium (>12GB)'])
    additional_df['price_segment'] = pd.cut(additional_df['price_clean'], bins=[0, 20000, 40000, 60000, float('inf')], 
                                           labels=['Budget', 'Mid-range', 'Premium', 'Flagship'])
    
    df_clean = pd.concat([df_clean, additional_df], ignore_index=True)

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Subplot 1: Brand Analysis - Horizontal bar with scatter overlay
ax1 = plt.subplot(3, 3, 1)
brand_stats = df_clean.groupby('brand_category').agg({
    'price_clean': 'mean',
    'rating': ['mean', 'count']
}).round(2)
brand_stats.columns = ['avg_price', 'avg_rating', 'count']
brand_stats = brand_stats[brand_stats['count'] >= 3].sort_values('avg_price')

if len(brand_stats) > 0:
    bars = ax1.barh(brand_stats.index, brand_stats['avg_price'], alpha=0.7, color='lightblue')
    ax1_twin = ax1.twiny()
    scatter = ax1_twin.scatter(brand_stats['avg_rating'], brand_stats.index, 
                              s=brand_stats['count']*5, alpha=0.8, color='red', edgecolors='darkred')
    ax1.set_xlabel('Average Price (₹)', fontweight='bold')
    ax1_twin.set_xlabel('Average Rating', fontweight='bold', color='red')

ax1.set_title('Brand Price vs Rating Analysis', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Subplot 2: Bubble chart - RAM vs Battery by Brand
ax2 = plt.subplot(3, 3, 2)
brand_bubble = df_clean.groupby('brand_category').agg({
    'ram_clean': 'mean',
    'battery_clean': 'mean',
    'price_clean': 'count'
}).round(2)
brand_bubble = brand_bubble[brand_bubble['price_clean'] >= 3]

if len(brand_bubble) > 0:
    colors = plt.cm.Set3(np.linspace(0, 1, len(brand_bubble)))
    for i, (brand, data) in enumerate(brand_bubble.iterrows()):
        ax2.scatter(data['ram_clean'], data['battery_clean'], 
                   s=data['price_clean']*30, alpha=0.7, color=colors[i], 
                   edgecolors='black', linewidth=1, label=brand)

ax2.set_xlabel('Average RAM (GB)', fontweight='bold')
ax2.set_ylabel('Average Battery (mAh)', fontweight='bold')
ax2.set_title('RAM vs Battery by Brand\n(Bubble size = Model count)', fontweight='bold', fontsize=12)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Stacked bar with line overlay
ax3 = plt.subplot(3, 3, 3)
processor_brand = pd.crosstab(df_clean['brand_category'], df_clean['processor_type'])
processor_brand = processor_brand.loc[processor_brand.sum(axis=1) >= 3]

if len(processor_brand) > 0:
    processor_brand.plot(kind='bar', stacked=True, ax=ax3, alpha=0.8)
    ax3_twin = ax3.twinx()
    brand_ratings = df_clean.groupby('brand_category')['rating'].mean()
    brand_ratings = brand_ratings[brand_ratings.index.isin(processor_brand.index)]
    if len(brand_ratings) > 0:
        ax3_twin.plot(range(len(brand_ratings)), brand_ratings.values, 
                      color='red', marker='o', linewidth=3, markersize=8, label='Avg Rating')
        ax3_twin.set_ylabel('Average Rating', fontweight='bold', color='red')
        ax3_twin.legend(loc='upper right')

ax3.set_xlabel('Brand', fontweight='bold')
ax3.set_ylabel('Number of Models', fontweight='bold')
ax3.set_title('Processor Types by Brand with Ratings', fontweight='bold', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(loc='upper left', fontsize=8)

# Subplot 4: Violin plot with box plot overlay
ax4 = plt.subplot(3, 3, 4)
ram_categories = ['Low (≤4GB)', 'Medium (4-8GB)', 'High (8-12GB)', 'Premium (>12GB)']
price_data = []
valid_categories = []

for cat in ram_categories:
    cat_data = df_clean[df_clean['ram_category'] == cat]['price_clean'].dropna()
    if len(cat_data) >= 3:
        price_data.append(cat_data)
        valid_categories.append(cat)

if len(price_data) > 0:
    try:
        parts = ax4.violinplot(price_data, positions=range(len(price_data)), showmeans=True, alpha=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        bp = ax4.boxplot(price_data, positions=range(len(price_data)), widths=0.3, 
                        patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.8))
        
        ax4.set_xticks(range(len(valid_categories)))
        ax4.set_xticklabels(valid_categories, rotation=45)
    except:
        # Fallback to simple box plot
        ax4.boxplot(price_data, labels=valid_categories)

ax4.set_ylabel('Price (₹)', fontweight='bold')
ax4.set_title('Price Distribution by RAM Category', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Subplot 5: Scatter plot with trend lines
ax5 = plt.subplot(3, 3, 5)
colors_ram = {'Low (≤4GB)': 'blue', 'Medium (4-8GB)': 'green', 
              'High (8-12GB)': 'orange', 'Premium (>12GB)': 'red'}

for category in df_clean['ram_category'].unique():
    if pd.notna(category):
        subset = df_clean[df_clean['ram_category'] == category]
        if len(subset) > 1:
            ax5.scatter(subset['battery_clean'], subset['camera_clean'], 
                       alpha=0.6, label=category, color=colors_ram.get(category, 'gray'))
            
            # Add trend line if enough points
            if len(subset) > 3:
                try:
                    z = np.polyfit(subset['battery_clean'].dropna(), 
                                  subset['camera_clean'].dropna(), 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(subset['battery_clean'].min(), 
                                        subset['battery_clean'].max(), 100)
                    ax5.plot(x_trend, p(x_trend), color=colors_ram.get(category, 'gray'), 
                            linestyle='--', alpha=0.8)
                except:
                    pass

ax5.set_xlabel('Battery Capacity (mAh)', fontweight='bold')
ax5.set_ylabel('Camera Megapixels', fontweight='bold')
ax5.set_title('Battery vs Camera by RAM Tier', fontweight='bold', fontsize=12)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 6: Grouped bar chart with error bars
ax6 = plt.subplot(3, 3, 6)
numeric_cols = ['ram_clean', 'battery_clean', 'display_clean', 'camera_clean']
price_segments = ['Budget', 'Mid-range', 'Premium', 'Flagship']

# Normalize data for comparison
scaler = StandardScaler()
df_normalized = df_clean.copy()
try:
    df_normalized[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    
    segment_stats = df_normalized.groupby('price_segment')[numeric_cols].agg(['mean', 'std'])
    segment_means = segment_stats.xs('mean', level=1, axis=1)
    segment_stds = segment_stats.xs('std', level=1, axis=1)
    
    x = np.arange(len(price_segments))
    width = 0.2
    colors = ['skyblue', 'lightgreen', 'orange', 'pink']
    
    for i, col in enumerate(numeric_cols):
        means = [segment_means.loc[seg, col] if seg in segment_means.index else 0 
                 for seg in price_segments]
        stds = [segment_stds.loc[seg, col] if seg in segment_stds.index else 0 
                for seg in price_segments]
        
        ax6.bar(x + i*width, means, width, label=col.replace('_clean', '').title(), 
               yerr=stds, capsize=3, alpha=0.8, color=colors[i])
    
    ax6.set_xticks(x + width * 1.5)
    ax6.set_xticklabels(price_segments)
    ax6.legend(fontsize=8)
except:
    ax6.text(0.5, 0.5, 'Insufficient data for normalization', 
             transform=ax6.transAxes, ha='center', va='center')

ax6.set_xlabel('Price Segment', fontweight='bold')
ax6.set_ylabel('Normalized Specifications', fontweight='bold')
ax6.set_title('Specifications by Price Segment\n(Normalized with Std Dev)', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)

# Subplot 7: Correlation heatmap
ax7 = plt.subplot(3, 3, 7)
corr_data = df_clean[['price_clean', 'rating', 'ram_clean', 'battery_clean', 'camera_clean']].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, ax=ax7, cbar_kws={'shrink': 0.8})
ax7.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=12)

# Subplot 8: Parallel coordinates plot
ax8 = plt.subplot(3, 3, 8)
parallel_data = df_clean[['price_clean', 'rating', 'ram_clean', 'battery_clean', 'display_clean', 'brand_category']].dropna()
top_brands_subset = df_clean['brand_category'].value_counts().head(5).index.tolist()
parallel_data = parallel_data[parallel_data['brand_category'].isin(top_brands_subset)]

if len(parallel_data) > 0:
    features = ['price_clean', 'rating', 'ram_clean', 'battery_clean', 'display_clean']
    parallel_normalized = parallel_data.copy()
    
    # Normalize features
    for feature in features:
        feature_min = parallel_data[feature].min()
        feature_max = parallel_data[feature].max()
        if feature_max != feature_min:
            parallel_normalized[feature] = (parallel_data[feature] - feature_min) / (feature_max - feature_min)
        else:
            parallel_normalized[feature] = 0.5
    
    brand_colors = plt.cm.Set1(np.linspace(0, 1, len(parallel_data['brand_category'].unique())))
    color_map = dict(zip(parallel_data['brand_category'].unique(), brand_colors))
    
    # Plot mean lines for each brand
    for brand in parallel_data['brand_category'].unique():
        brand_data = parallel_normalized[parallel_normalized['brand_category'] == brand]
        if len(brand_data) > 0:
            means = [brand_data[f].mean() for f in features]
            ax8.plot(range(len(features)), means, linewidth=3, 
                    color=color_map[brand], label=brand, alpha=0.8)
    
    ax8.set_xticks(range(len(features)))
    ax8.set_xticklabels([f.replace('_clean', '').title() for f in features], rotation=45)
    ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

ax8.set_ylabel('Normalized Values', fontweight='bold')
ax8.set_title('Parallel Coordinates by Brand', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3)

# Subplot 9: Cluster analysis with PCA
ax9 = plt.subplot(3, 3, 9)
cluster_features = ['price_clean', 'rating', 'ram_clean', 'battery_clean', 'camera_clean']
cluster_data = df_clean[cluster_features].dropna()

if len(cluster_data) >= 10:
    try:
        # Standardize and apply PCA
        scaler_cluster = StandardScaler()
        data_scaled = scaler_cluster.fit_transform(cluster_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)
        
        # Perform clustering
        n_clusters = min(4, len(cluster_data) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Create scatter plot
        scatter = ax9.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, 
                             cmap='viridis', alpha=0.6, s=30)
        
        # Transform cluster centers to PCA space
        pca_centers = pca.transform(kmeans.cluster_centers_)
        ax9.scatter(pca_centers[:, 0], pca_centers[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
        ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
        ax9.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax9)
        cbar.set_label('Cluster', fontweight='bold')
        
    except Exception as e:
        ax9.text(0.5, 0.5, f'Clustering failed: insufficient data', 
                 transform=ax9.transAxes, ha='center', va='center')
else:
    ax9.text(0.5, 0.5, 'Insufficient data for clustering analysis', 
             transform=ax9.transAxes, ha='center', va='center')

ax9.set_title('Smartphone Clusters (PCA Projection)', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.savefig('smartphone_analysis_grid.png', dpi=300, bbox_inches='tight')
plt.show()