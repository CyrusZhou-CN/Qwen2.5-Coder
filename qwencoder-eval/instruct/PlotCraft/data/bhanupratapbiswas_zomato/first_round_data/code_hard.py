import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('zomato.csv')

# Clean the rate column more carefully
def clean_rate(rate_str):
    if pd.isna(rate_str):
        return np.nan
    
    # Convert to string and handle various formats
    rate_str = str(rate_str)
    
    # Look for patterns like "4.1/5", "NEW", "-", or complex strings
    if '/5' in rate_str:
        try:
            return float(rate_str.split('/5')[0])
        except:
            return np.nan
    elif rate_str in ['NEW', '-', 'nan']:
        return np.nan
    else:
        # For complex strings, try to extract first number
        import re
        numbers = re.findall(r'\d+\.?\d*', rate_str)
        if numbers:
            try:
                num = float(numbers[0])
                if 0 <= num <= 5:  # Valid rating range
                    return num
            except:
                pass
        return np.nan

# Apply the cleaning function
df['rate_numeric'] = df['rate'].apply(clean_rate)

# Clean cost and votes
df['cost'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
df['votes_numeric'] = pd.to_numeric(df['votes'], errors='coerce')

# Remove rows with missing critical data and filter reasonable ranges
df = df.dropna(subset=['rate_numeric', 'cost', 'location', 'cuisines'])
df = df[(df['rate_numeric'] >= 1) & (df['rate_numeric'] <= 5)]
df = df[(df['cost'] >= 50) & (df['cost'] <= 5000)]  # Reasonable cost range
df = df[df['votes_numeric'] >= 0]

# Get top locations and cuisines for analysis
top_locations = df['location'].value_counts().head(8).index.tolist()
df_top_loc = df[df['location'].isin(top_locations)]

# Process cuisines - take first cuisine for simplicity
df['primary_cuisine'] = df['cuisines'].str.split(',').str[0].str.strip()
top_cuisines = df['primary_cuisine'].value_counts().head(6).index.tolist()

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('white')

# Subplot 1: Location analysis - Bar + Line
ax1 = plt.subplot(3, 3, 1)
loc_counts = df_top_loc['location'].value_counts()
loc_ratings = df_top_loc.groupby('location')['rate_numeric'].mean()

# Bar chart
bars = ax1.barh(range(len(loc_counts)), loc_counts.values, alpha=0.7, color='lightblue')
ax1.set_yticks(range(len(loc_counts)))
ax1.set_yticklabels([loc[:15] for loc in loc_counts.index], fontsize=10)
ax1.set_xlabel('Restaurant Count', fontweight='bold')
ax1.set_title('Restaurant Count & Average Rating by Location', fontweight='bold', fontsize=14)

# Overlaid line plot
ax1_twin = ax1.twiny()
line_y = [list(loc_counts.index).index(loc) for loc in loc_ratings.index if loc in loc_counts.index]
line_x = [loc_ratings[loc] for loc in loc_ratings.index if loc in loc_counts.index]
ax1_twin.plot(line_x, line_y, 'ro-', linewidth=3, markersize=8, color='red', alpha=0.8)
ax1_twin.set_xlabel('Average Rating', color='red', fontweight='bold')
ax1_twin.tick_params(axis='x', labelcolor='red')
ax1_twin.grid(True, alpha=0.3)

# Subplot 2: Scatter plot with marginal analysis
ax2 = plt.subplot(3, 3, 2)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_locations)))
for i, loc in enumerate(top_locations):
    loc_data = df_top_loc[df_top_loc['location'] == loc]
    if len(loc_data) > 0:
        ax2.scatter(loc_data['cost'], loc_data['rate_numeric'], 
                   alpha=0.6, s=40, color=colors[i], label=loc[:12])

ax2.set_xlabel('Cost (for two people)', fontweight='bold')
ax2.set_ylabel('Rating', fontweight='bold')
ax2.set_title('Cost vs Rating by Location', fontweight='bold', fontsize=14)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Subplot 3: Stacked bar chart with secondary axis
ax3 = plt.subplot(3, 3, 3)
rest_type_loc = pd.crosstab(df_top_loc['location'], df_top_loc['rest_type'])
# Get top 4 restaurant types
top_rest_types_for_stack = rest_type_loc.sum().nlargest(4).index
rest_type_loc_top = rest_type_loc[top_rest_types_for_stack]

rest_type_loc_top.plot(kind='bar', stacked=True, ax=ax3, alpha=0.8, colormap='Set2')
ax3.set_xlabel('Location', fontweight='bold')
ax3.set_ylabel('Restaurant Count', fontweight='bold')
ax3.set_title('Restaurant Types by Location & Total Votes', fontweight='bold', fontsize=14)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Secondary y-axis for votes
ax3_twin = ax3.twinx()
votes_by_loc = df_top_loc.groupby('location')['votes_numeric'].sum()
ax3_twin.plot(range(len(votes_by_loc)), votes_by_loc.values, 'ro-', linewidth=3, color='red', markersize=8)
ax3_twin.set_ylabel('Total Votes', color='red', fontweight='bold')
ax3_twin.tick_params(axis='y', labelcolor='red')

# Subplot 4: Bubble chart with cuisine clustering
ax4 = plt.subplot(3, 3, 4)
cuisine_stats = df[df['primary_cuisine'].isin(top_cuisines)].groupby('primary_cuisine').agg({
    'cost': 'mean',
    'rate_numeric': 'mean',
    'votes_numeric': 'sum'
}).reset_index()

# Simple clustering based on cost and rating
if len(cuisine_stats) >= 3:
    X = cuisine_stats[['cost', 'rate_numeric']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
else:
    clusters = range(len(cuisine_stats))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
for i in range(len(cuisine_stats)):
    cluster_idx = clusters[i] if len(cuisine_stats) >= 3 else i % 3
    ax4.scatter(cuisine_stats.iloc[i]['cost'], cuisine_stats.iloc[i]['rate_numeric'],
               s=max(50, cuisine_stats.iloc[i]['votes_numeric']/200), 
               color=colors[cluster_idx], alpha=0.7,
               edgecolors='black', linewidth=1)
    
    # Add cuisine labels
    ax4.annotate(cuisine_stats.iloc[i]['primary_cuisine'][:8], 
                (cuisine_stats.iloc[i]['cost'], cuisine_stats.iloc[i]['rate_numeric']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.set_xlabel('Average Cost', fontweight='bold')
ax4.set_ylabel('Average Rating', fontweight='bold')
ax4.set_title('Cuisine Clustering (Bubble Size = Total Votes)', fontweight='bold', fontsize=14)
ax4.grid(True, alpha=0.3)

# Subplot 5: Violin plot with strip plot overlay
ax5 = plt.subplot(3, 3, 5)
cuisine_data = df[df['primary_cuisine'].isin(top_cuisines)]

# Prepare data for violin plot
violin_data = []
for cuisine in top_cuisines:
    data = cuisine_data[cuisine_data['primary_cuisine'] == cuisine]['cost'].dropna()
    if len(data) > 0:
        violin_data.append(data.values)
    else:
        violin_data.append([0])

if violin_data:
    violin_parts = ax5.violinplot(violin_data, positions=range(len(top_cuisines)), showmeans=True)
    
    # Color the violins
    for pc, color in zip(violin_parts['bodies'], plt.cm.Set2(np.linspace(0, 1, len(top_cuisines)))):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Add strip plot overlay
    for i, cuisine in enumerate(top_cuisines):
        data = cuisine_data[cuisine_data['primary_cuisine'] == cuisine]['cost'].dropna()
        if len(data) > 0:
            sample_size = min(len(data), 30)  # Limit points for clarity
            sample_data = data.sample(sample_size) if len(data) > sample_size else data
            y = np.random.normal(i, 0.04, size=len(sample_data))
            ax5.scatter(sample_data, y, alpha=0.5, s=15, color='black')

ax5.set_xticks(range(len(top_cuisines)))
ax5.set_xticklabels([c[:10] for c in top_cuisines], rotation=45)
ax5.set_ylabel('Cost Distribution', fontweight='bold')
ax5.set_title('Cost Distribution by Cuisine', fontweight='bold', fontsize=14)

# Subplot 6: Radar chart for top 5 cuisines
ax6 = plt.subplot(3, 3, 6, projection='polar')
top_5_cuisines = top_cuisines[:5]
metrics = ['Avg Rating', 'Avg Cost', 'Restaurant Count', 'Avg Votes']

# Prepare data for radar chart
radar_data = []
for cuisine in top_5_cuisines:
    cuisine_subset = df[df['primary_cuisine'] == cuisine]
    if len(cuisine_subset) > 0:
        values = [
            cuisine_subset['rate_numeric'].mean(),
            min(cuisine_subset['cost'].mean() / 200, 5),  # Scale and cap
            min(len(cuisine_subset) / 50, 5),  # Scale and cap
            min(cuisine_subset['votes_numeric'].mean() / 50, 5)  # Scale and cap
        ]
        radar_data.append(values)

if radar_data:
    # Number of variables
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    colors = plt.cm.Set1(np.linspace(0, 1, len(top_5_cuisines)))
    for i, (cuisine, values) in enumerate(zip(top_5_cuisines, radar_data)):
        values += values[:1]  # Complete the circle
        ax6.plot(angles, values, 'o-', linewidth=2, label=cuisine[:10], color=colors[i])
        ax6.fill(angles, values, alpha=0.25, color=colors[i])

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)
    ax6.set_title('Top 5 Cuisines Comparison', fontweight='bold', fontsize=14, pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Subplot 7: Grouped bar chart with error bars
ax7 = plt.subplot(3, 3, 7)
top_rest_types = df['rest_type'].value_counts().head(4).index

# Prepare service analysis data
service_data = []
service_labels = []
for rest_type in top_rest_types:
    rest_data = df[df['rest_type'] == rest_type]
    for online in ['Yes', 'No']:
        for book in ['Yes', 'No']:
            subset = rest_data[(rest_data['online_order'] == online) & (rest_data['book_table'] == book)]
            if len(subset) > 0:
                service_data.append({
                    'rest_type': rest_type,
                    'online': online,
                    'book': book,
                    'rating_mean': subset['rate_numeric'].mean(),
                    'rating_std': subset['rate_numeric'].std(),
                    'count': len(subset)
                })

# Create grouped bar chart
x = np.arange(len(top_rest_types))
width = 0.2

service_combinations = [('Yes', 'Yes'), ('Yes', 'No'), ('No', 'Yes'), ('No', 'No')]
colors = ['darkgreen', 'lightgreen', 'orange', 'red']

for i, (online, book) in enumerate(service_combinations):
    ratings = []
    errors = []
    for rest_type in top_rest_types:
        matching = [d for d in service_data if d['rest_type'] == rest_type and d['online'] == online and d['book'] == book]
        if matching:
            ratings.append(matching[0]['rating_mean'])
            errors.append(matching[0]['rating_std'] if not pd.isna(matching[0]['rating_std']) else 0)
        else:
            ratings.append(0)
            errors.append(0)
    
    ax7.bar(x + i*width, ratings, width, label=f'Online:{online}, Book:{book}', 
           yerr=errors, capsize=3, alpha=0.8, color=colors[i])

ax7.set_xlabel('Restaurant Type', fontweight='bold')
ax7.set_ylabel('Average Rating', fontweight='bold')
ax7.set_title('Service Features vs Rating by Restaurant Type', fontweight='bold', fontsize=14)
ax7.set_xticks(x + width * 1.5)
ax7.set_xticklabels([rt[:10] for rt in top_rest_types], rotation=45)
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax7.grid(True, alpha=0.3)

# Subplot 8: Correlation heatmap
ax8 = plt.subplot(3, 3, 8)

# Calculate correlation matrix for numerical features
numerical_features = ['cost', 'rate_numeric', 'votes_numeric']
corr_matrix = df[numerical_features].corr()

# Create heatmap
im = ax8.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation coefficients
for i in range(len(numerical_features)):
    for j in range(len(numerical_features)):
        ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold', fontsize=12)

ax8.set_xticks(range(len(numerical_features)))
ax8.set_yticks(range(len(numerical_features)))
ax8.set_xticklabels(['Cost', 'Rating', 'Votes'])
ax8.set_yticklabels(['Cost', 'Rating', 'Votes'])
ax8.set_title('Feature Correlations Matrix', fontweight='bold', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax8, shrink=0.8)
cbar.set_label('Correlation Coefficient', fontweight='bold')

# Subplot 9: Parallel coordinates plot (simplified)
ax9 = plt.subplot(3, 3, 9)

# Sample data for parallel coordinates
sample_size = min(500, len(df))
sample_data = df.sample(n=sample_size, random_state=42)

# Create bins for continuous variables
sample_data['cost_bin'] = pd.cut(sample_data['cost'], bins=3, labels=[0, 1, 2])
sample_data['rating_bin'] = pd.cut(sample_data['rate_numeric'], bins=3, labels=[0, 1, 2])

# Encode categorical variables
location_codes = {loc: i for i, loc in enumerate(sample_data['location'].unique())}
sample_data['location_code'] = sample_data['location'].map(location_codes)

online_codes = {'Yes': 1, 'No': 0}
sample_data['online_code'] = sample_data['online_order'].map(online_codes)
sample_data['book_code'] = sample_data['book_table'].map(online_codes)

# Prepare parallel coordinates data
features = ['location_code', 'cost_bin', 'rating_bin', 'online_code', 'book_code']
feature_labels = ['Location', 'Cost Level', 'Rating Level', 'Online Order', 'Book Table']

# Get top restaurant types for coloring
top_rest_types_parallel = sample_data['rest_type'].value_counts().head(3).index
colors = ['red', 'blue', 'green']

for i, rest_type in enumerate(top_rest_types_parallel):
    subset = sample_data[sample_data['rest_type'] == rest_type].head(50)
    for idx, row in subset.iterrows():
        values = []
        for feature in features:
            if pd.notna(row[feature]):
                values.append(float(row[feature]))
            else:
                values.append(0)
        
        if len(values) == len(features):
            ax9.plot(range(len(features)), values, 
                    color=colors[i], alpha=0.4, linewidth=1)

ax9.set_xticks(range(len(features)))
ax9.set_xticklabels(feature_labels, rotation=45)
ax9.set_ylabel('Encoded Values', fontweight='bold')
ax9.set_title('Parallel Coordinates: Multi-dimensional Analysis', fontweight='bold', fontsize=14)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=top_rest_types_parallel[i][:12]) 
                  for i in range(len(top_rest_types_parallel))]
ax9.legend(handles=legend_elements, loc='upper right')
ax9.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=3.0)
plt.savefig('zomato_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()