import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('cars_ds_final.csv')

# Clean and convert key columns
def clean_price(price_str):
    if pd.isna(price_str) or price_str == 'NaN':
        return np.nan
    return float(price_str.replace('Rs. ', '').replace(',', ''))

def clean_numeric(value):
    if pd.isna(value) or value == 'NaN':
        return np.nan
    if isinstance(value, str):
        # Extract numeric part
        import re
        numbers = re.findall(r'[\d.]+', value)
        if numbers:
            return float(numbers[0])
    return float(value) if value != 'NaN' else np.nan

# Clean data
df['Price_Clean'] = df['Ex-Showroom_Price'].apply(clean_price)
df['Displacement_Clean'] = df['Displacement'].apply(clean_numeric)
df['Power_Clean'] = df['Power'].apply(clean_numeric)
df['Torque_Clean'] = df['Torque'].apply(clean_numeric)
df['Mileage_Clean'] = df['ARAI_Certified_Mileage'].apply(clean_numeric)

# Remove rows with missing critical data
df_clean = df.dropna(subset=['Price_Clean', 'Make', 'Body_Type', 'Fuel_Type'])

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('white')

# Row 1: Brand Analysis
# Subplot 1: Horizontal bar chart with scatter overlay
ax1 = plt.subplot(3, 3, 1)
brand_stats = df_clean.groupby('Make').agg({
    'Price_Clean': 'mean',
    'Make': 'count'
}).rename(columns={'Make': 'Count'})
brand_stats = brand_stats.sort_values('Count', ascending=True).tail(10)

bars = ax1.barh(range(len(brand_stats)), brand_stats['Count'], 
                color='lightblue', alpha=0.7, label='Car Count')
ax1_twin = ax1.twinx()
scatter = ax1_twin.scatter(brand_stats['Price_Clean']/100000, range(len(brand_stats)), 
                          color='red', s=100, alpha=0.8, label='Avg Price (₹L)', zorder=5)

ax1.set_yticks(range(len(brand_stats)))
ax1.set_yticklabels(brand_stats.index, fontsize=10)
ax1.set_xlabel('Number of Models', fontweight='bold')
ax1.set_title('Car Count by Brand with Average Price Overlay', fontweight='bold', fontsize=12)
ax1_twin.set_ylabel('Average Price (₹ Lakhs)', fontweight='bold')
ax1.legend(loc='lower right')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: Violin plot with box plot overlay
ax2 = plt.subplot(3, 3, 2)
top_brands = df_clean['Make'].value_counts().head(6).index
df_violin = df_clean[df_clean['Make'].isin(top_brands)]

parts = ax2.violinplot([df_violin[df_violin['Make']==brand]['Price_Clean'].dropna()/100000 
                       for brand in top_brands], 
                      positions=range(len(top_brands)), widths=0.8, showmeans=True)

for pc in parts['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_alpha(0.7)

# Overlay box plots
box_data = [df_violin[df_violin['Make']==brand]['Price_Clean'].dropna()/100000 
           for brand in top_brands]
bp = ax2.boxplot(box_data, positions=range(len(top_brands)), widths=0.3, 
                patch_artist=True, showfliers=True)

for patch in bp['boxes']:
    patch.set_facecolor('navy')
    patch.set_alpha(0.8)

ax2.set_xticks(range(len(top_brands)))
ax2.set_xticklabels(top_brands, rotation=45, ha='right')
ax2.set_ylabel('Price (₹ Lakhs)', fontweight='bold')
ax2.set_title('Price Distribution by Top Brands\n(Violin + Box Plot)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Subplot 3: Bubble chart
ax3 = plt.subplot(3, 3, 3)
bubble_data = df_clean.groupby('Make').agg({
    'Displacement_Clean': 'mean',
    'Mileage_Clean': 'mean',
    'Make': 'count',
    'Price_Clean': 'mean'
}).rename(columns={'Make': 'Count'})
bubble_data = bubble_data.dropna()

scatter = ax3.scatter(bubble_data['Displacement_Clean'], bubble_data['Mileage_Clean'],
                     s=bubble_data['Count']*20, c=bubble_data['Price_Clean']/100000,
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)

for i, make in enumerate(bubble_data.index):
    if bubble_data.iloc[i]['Count'] > 20:  # Only label major brands
        ax3.annotate(make, (bubble_data.iloc[i]['Displacement_Clean'], 
                           bubble_data.iloc[i]['Mileage_Clean']),
                    fontsize=8, ha='center')

ax3.set_xlabel('Average Engine Displacement (cc)', fontweight='bold')
ax3.set_ylabel('Average Mileage (km/l)', fontweight='bold')
ax3.set_title('Brand Positioning: Displacement vs Mileage\n(Bubble size = Model count)', 
              fontweight='bold', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Avg Price (₹L)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Row 2: Technical Specifications Clustering
# Subplot 4: Parallel coordinates plot
ax4 = plt.subplot(3, 3, 4)
tech_cols = ['Cylinders', 'Displacement_Clean', 'Power_Clean', 'Torque_Clean', 'Mileage_Clean']
df_tech = df_clean[tech_cols + ['Fuel_Type']].dropna()

# Normalize data for parallel coordinates
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_tech[tech_cols]), 
                           columns=tech_cols, index=df_tech.index)
df_normalized['Fuel_Type'] = df_tech['Fuel_Type']

fuel_types = df_normalized['Fuel_Type'].unique()[:4]  # Limit to 4 fuel types
colors_fuel_dict = dict(zip(fuel_types, ['red', 'blue', 'green', 'orange']))

for fuel in fuel_types:
    fuel_data = df_normalized[df_normalized['Fuel_Type'] == fuel]
    if len(fuel_data) > 0:
        sample_data = fuel_data.sample(min(50, len(fuel_data)))  # Sample for clarity
        for idx in sample_data.index:
            ax4.plot(range(len(tech_cols)), sample_data.loc[idx, tech_cols], 
                    color=colors_fuel_dict[fuel], alpha=0.3, linewidth=0.5)

ax4.set_xticks(range(len(tech_cols)))
ax4.set_xticklabels(['Cylinders', 'Displacement', 'Power', 'Torque', 'Mileage'], 
                   rotation=45, ha='right')
ax4.set_ylabel('Normalized Values', fontweight='bold')
ax4.set_title('Parallel Coordinates: Technical Specs by Fuel Type', 
              fontweight='bold', fontsize=12)

# Create legend
for fuel, color in colors_fuel_dict.items():
    ax4.plot([], [], color=color, label=fuel, linewidth=2)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Subplot 5: Correlation heatmap
ax5 = plt.subplot(3, 3, 5)
corr_cols = ['Price_Clean', 'Displacement_Clean', 'Power_Clean', 'Torque_Clean', 
             'Mileage_Clean', 'Seating_Capacity']
corr_data = df_clean[corr_cols].dropna()
correlation_matrix = corr_data.corr()

# Create heatmap
im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        text = ax5.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

ax5.set_xticks(range(len(correlation_matrix.columns)))
ax5.set_yticks(range(len(correlation_matrix.columns)))
ax5.set_xticklabels(['Price', 'Displacement', 'Power', 'Torque', 'Mileage', 'Seating'], 
                   rotation=45, ha='right')
ax5.set_yticklabels(['Price', 'Displacement', 'Power', 'Torque', 'Mileage', 'Seating'])
ax5.set_title('Technical Specifications Correlation Matrix', fontweight='bold', fontsize=12)

# Subplot 6: Scatter plot matrix style
ax6 = plt.subplot(3, 3, 6)
power_torque_data = df_clean[['Power_Clean', 'Torque_Clean', 'Body_Type']].dropna()
body_types = power_torque_data['Body_Type'].unique()[:6]
colors_body_dict = dict(zip(body_types, plt.cm.Set2(np.linspace(0, 1, len(body_types)))))

for body_type in body_types:
    data = power_torque_data[power_torque_data['Body_Type'] == body_type]
    ax6.scatter(data['Power_Clean'], data['Torque_Clean'], 
               c=[colors_body_dict[body_type]], label=body_type, alpha=0.7, s=30)

ax6.set_xlabel('Power (PS)', fontweight='bold')
ax6.set_ylabel('Torque (Nm)', fontweight='bold')
ax6.set_title('Power vs Torque by Body Type', fontweight='bold', fontsize=12)
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax6.grid(True, alpha=0.3)

# Row 3: Market Segmentation
# Subplot 7: Treemap simulation using rectangles
ax7 = plt.subplot(3, 3, 7)
treemap_data = df_clean.groupby(['Make', 'Body_Type']).agg({
    'Price_Clean': 'mean',
    'Make': 'count'
}).rename(columns={'Make': 'Count'})
treemap_data = treemap_data.reset_index()
treemap_data = treemap_data.sort_values('Count', ascending=False).head(15)

# Simple treemap visualization
y_pos = 0
# Fix: Use the actual length of treemap_data for color generation
colors = plt.cm.viridis(np.linspace(0, 1, len(treemap_data)))

for i, (idx, row) in enumerate(treemap_data.iterrows()):
    width = row['Count'] / treemap_data['Count'].max()
    height = 0.8
    rect = Rectangle((0, y_pos), width, height, 
                    facecolor=colors[i], alpha=0.7, edgecolor='black')
    ax7.add_patch(rect)
    
    # Add text
    ax7.text(width/2, y_pos + height/2, f"{row['Make']}\n{row['Body_Type']}", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    y_pos += 1

ax7.set_xlim(0, 1.2)
ax7.set_ylim(0, len(treemap_data))
ax7.set_title('Market Composition: Make × Body Type\n(Width = Model Count)', 
              fontweight='bold', fontsize=12)
ax7.set_xticks([])
ax7.set_yticks([])

# Subplot 8: Radar chart
ax8 = plt.subplot(3, 3, 8, projection='polar')
radar_data = df_clean.groupby('Body_Type').agg({
    'Power_Clean': 'mean',
    'Torque_Clean': 'mean',
    'Mileage_Clean': 'mean',
    'Seating_Capacity': 'mean'
}).dropna()

# Normalize data
radar_normalized = radar_data.div(radar_data.max())
categories = ['Power', 'Torque', 'Mileage', 'Seating']
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Fix: Use the actual length of radar_normalized for color generation
colors_radar = plt.cm.Set1(np.linspace(0, 1, len(radar_normalized)))

for i, (body_type, values) in enumerate(radar_normalized.iterrows()):
    values_list = values.tolist()
    values_list += values_list[:1]  # Complete the circle
    ax8.plot(angles, values_list, 'o-', linewidth=2, label=body_type, 
            color=colors_radar[i])
    ax8.fill(angles, values_list, alpha=0.25, color=colors_radar[i])

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(categories)
ax8.set_title('Body Type Performance Radar\n(Normalized Specifications)', 
              fontweight='bold', fontsize=12, pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

# Subplot 9: Dendrogram with heatmap
ax9 = plt.subplot(3, 3, 9)
cluster_cols = ['Power_Clean', 'Torque_Clean', 'Displacement_Clean', 'Mileage_Clean']
cluster_data = df_clean[cluster_cols + ['Model']].dropna()

# Sample data for clustering (too many models for clear visualization)
sample_size = min(30, len(cluster_data))  # Reduce sample size for better visualization
sample_data = cluster_data.sample(sample_size, random_state=42)
X = sample_data[cluster_cols]
X_scaled = StandardScaler().fit_transform(X)

# Perform hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')

# Create dendrogram with truncated labels for better readability
model_labels = [model[:15] + '...' if len(model) > 15 else model for model in sample_data['Model'].values]
dendrogram(linkage_matrix, ax=ax9, labels=model_labels,
          leaf_rotation=90, leaf_font_size=6)

ax9.set_title('Hierarchical Clustering of Car Models\n(Based on Technical Specs)', 
              fontweight='bold', fontsize=12)
ax9.set_xlabel('Car Models', fontweight='bold')
ax9.set_ylabel('Distance', fontweight='bold')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Add main title
fig.suptitle('Comprehensive Indian Car Market Analysis: Segmentation & Clustering Patterns', 
             fontsize=20, fontweight='bold', y=0.98)

plt.savefig('indian_car_market_analysis.png', dpi=300, bbox_inches='tight')
plt.show()