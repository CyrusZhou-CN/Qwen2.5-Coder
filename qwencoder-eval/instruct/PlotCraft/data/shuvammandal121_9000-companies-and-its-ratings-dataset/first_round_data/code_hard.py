import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('company_dataset.csv')

# Clean and extract numerical values
def extract_review_count(review_str):
    if pd.isna(review_str):
        return 0
    review_str = str(review_str).replace('(', '').replace(')', '').replace('Reviews', '').replace('k', '000').replace(',', '')
    try:
        return float(review_str.split()[0])
    except:
        return 0

def extract_years(years_str):
    if pd.isna(years_str):
        return 0
    try:
        return int(str(years_str).split()[0])
    except:
        return 0

def extract_employee_size(emp_str):
    if pd.isna(emp_str):
        return 0
    emp_str = str(emp_str).lower()
    if 'lakh+' in emp_str or '100000+' in emp_str:
        return 100000
    elif '10000+' in emp_str or '10k+' in emp_str:
        return 10000
    elif '1000+' in emp_str or '1k+' in emp_str:
        return 1000
    elif '500+' in emp_str:
        return 500
    elif '100+' in emp_str:
        return 100
    else:
        return 50

# Apply preprocessing
df['review_count_num'] = df['review_count'].apply(extract_review_count)
df['years_num'] = df['years'].apply(extract_years)
df['employees_num'] = df['employees'].apply(extract_employee_size)

# Remove rows with missing critical data
df = df.dropna(subset=['ratings', 'ctype'])
df = df[df['ratings'] > 0]

# Normalize some variables for certain plots
df['review_count_norm'] = (df['review_count_num'] - df['review_count_num'].min()) / (df['review_count_num'].max() - df['review_count_num'].min() + 1e-8)
df['years_norm'] = (df['years_num'] - df['years_num'].min()) / (df['years_num'].max() - df['years_num'].min() + 1e-8)

# Set up the figure with white background
plt.style.use('default')
fig = plt.figure(figsize=(20, 16), facecolor='white')
fig.patch.set_facecolor('white')

# Define color palette for company types
company_types = df['ctype'].unique()[:6]  # Limit to 6 types for clarity
colors = plt.cm.Set3(np.linspace(0, 1, len(company_types)))
color_map = dict(zip(company_types, colors))

# Filter data to only include these company types
df_filtered = df[df['ctype'].isin(company_types)]

# Subplot 1: Violin plot with box plots and swarm plot overlay
ax1 = plt.subplot(3, 3, 1, facecolor='white')
# Create violin plot
violin_data = [df_filtered[df_filtered['ctype'] == ct]['ratings'].values for ct in company_types]
parts = ax1.violinplot(violin_data, positions=range(len(company_types)), widths=0.6, showmeans=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Add box plots
bp = ax1.boxplot(violin_data, positions=range(len(company_types)), widths=0.3, patch_artist=True)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors[i])
    patch.set_alpha(0.5)

# Add swarm plot effect (sample points to avoid overcrowding)
for i, ct in enumerate(company_types):
    ct_data = df_filtered[df_filtered['ctype'] == ct]['ratings']
    if len(ct_data) > 0:
        sample_data = ct_data.sample(min(50, len(ct_data)))
        y_vals = sample_data.values
        x_vals = np.random.normal(i, 0.04, len(y_vals))
        ax1.scatter(x_vals, y_vals, alpha=0.6, s=8, color=colors[i])

ax1.set_title('Rating Distributions by Company Type\n(Violin + Box + Swarm)', fontweight='bold', fontsize=12)
ax1.set_xticks(range(len(company_types)))
ax1.set_xticklabels(company_types, rotation=45, ha='right')
ax1.set_ylabel('Rating')
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked bar chart with line overlay
ax2 = plt.subplot(3, 3, 2, facecolor='white')
# Create employee size categories
emp_categories = ['Small (<1K)', 'Medium (1K-10K)', 'Large (10K-100K)', 'Very Large (100K+)']
emp_data = {}
avg_ratings = {}

for ct in company_types:
    ct_data = df_filtered[df_filtered['ctype'] == ct]
    emp_counts = [
        len(ct_data[ct_data['employees_num'] < 1000]),
        len(ct_data[(ct_data['employees_num'] >= 1000) & (ct_data['employees_num'] < 10000)]),
        len(ct_data[(ct_data['employees_num'] >= 10000) & (ct_data['employees_num'] < 100000)]),
        len(ct_data[ct_data['employees_num'] >= 100000])
    ]
    emp_data[ct] = emp_counts
    avg_ratings[ct] = ct_data['ratings'].mean()

# Create stacked bar chart
bottom = np.zeros(len(company_types))
for i, category in enumerate(emp_categories):
    values = [emp_data[ct][i] for ct in company_types]
    ax2.bar(company_types, values, bottom=bottom, label=category, alpha=0.8)
    bottom += values

# Add line plot for average ratings
ax2_twin = ax2.twinx()
rating_values = [avg_ratings[ct] for ct in company_types]
ax2_twin.plot(company_types, rating_values, 'ro-', linewidth=3, markersize=8, label='Avg Rating')
ax2_twin.set_ylabel('Average Rating', color='red')

ax2.set_title('Employee Size Distribution by Company Type\n(Stacked Bar + Average Rating Line)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Company Type')
ax2.set_ylabel('Number of Companies')
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
ax2_twin.legend(loc='upper right')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# Subplot 3: Bubble chart with trend lines (fixed SVD error)
ax3 = plt.subplot(3, 3, 3, facecolor='white')
for i, ct in enumerate(company_types):
    ct_data = df_filtered[df_filtered['ctype'] == ct]
    if len(ct_data) > 0:
        sample_size = min(200, len(ct_data))
        ct_sample = ct_data.sample(sample_size)
        x = ct_sample['years_num']
        y = ct_sample['ratings']
        sizes = ct_sample['review_count_num'] / 100  # Scale bubble sizes
        
        ax3.scatter(x, y, s=sizes, alpha=0.6, c=[colors[i]], label=ct, edgecolors='white', linewidth=0.5)
        
        # Add trend line with error handling
        if len(x) > 2 and x.std() > 0:  # Ensure we have enough points and variance
            try:
                # Use robust polynomial fitting with degree 1
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x)
                ax3.plot(x_sorted, p(x_sorted), color=colors[i], linestyle='--', alpha=0.8, linewidth=2)
            except:
                # If polyfit fails, use simple linear regression
                if len(x) > 1:
                    slope = np.corrcoef(x, y)[0, 1] * (y.std() / x.std())
                    intercept = y.mean() - slope * x.mean()
                    x_range = np.linspace(x.min(), x.max(), 10)
                    y_trend = slope * x_range + intercept
                    ax3.plot(x_range, y_trend, color=colors[i], linestyle='--', alpha=0.8, linewidth=2)

ax3.set_title('Company Age vs Rating\n(Bubble Size = Review Count, Trend Lines)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Company Age (Years)')
ax3.set_ylabel('Rating')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# Subplot 4: Parallel coordinates plot
ax4 = plt.subplot(3, 3, 4, facecolor='white')
# Sample data for parallel coordinates
sample_df = df_filtered.sample(min(500, len(df_filtered)))
coords_data = sample_df[['years_num', 'ratings', 'review_count_num', 'employees_num']].copy()

# Normalize data safely
for col in ['years_num', 'review_count_num', 'employees_num']:
    col_min, col_max = coords_data[col].min(), coords_data[col].max()
    if col_max > col_min:
        coords_data[f'{col}_norm'] = (coords_data[col] - col_min) / (col_max - col_min)
    else:
        coords_data[f'{col}_norm'] = 0.5

# Normalize ratings
rating_min, rating_max = coords_data['ratings'].min(), coords_data['ratings'].max()
if rating_max > rating_min:
    coords_data['ratings_norm'] = (coords_data['ratings'] - rating_min) / (rating_max - rating_min)
else:
    coords_data['ratings_norm'] = 0.5

x_coords = [0, 1, 2, 3]
labels = ['Age', 'Rating', 'Reviews', 'Employees']

for idx, row in coords_data.iterrows():
    ct = sample_df.loc[idx, 'ctype']
    color = color_map[ct]
    y_coords = [row['years_num_norm'], row['ratings_norm'], row['review_count_num_norm'], row['employees_num_norm']]
    ax4.plot(x_coords, y_coords, alpha=0.3, color=color, linewidth=0.8)

ax4.set_title('Parallel Coordinates Plot\n(Normalized Variables by Company Type)', fontweight='bold', fontsize=12)
ax4.set_xticks(x_coords)
ax4.set_xticklabels(labels)
ax4.set_ylabel('Normalized Values')
ax4.grid(True, alpha=0.3)

# Subplot 5: Correlation heatmap
ax5 = plt.subplot(3, 3, 5, facecolor='white')
corr_data = df_filtered[['years_num', 'ratings', 'review_count_num']].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax5)
ax5.set_title('Correlation Heatmap\n(Age, Rating, Review Count)', fontweight='bold', fontsize=12)

# Subplot 6: Scatter plot matrix
ax6 = plt.subplot(3, 3, 6, facecolor='white')
sample_data = df_filtered.sample(min(1000, len(df_filtered)))
for i, ct in enumerate(company_types):
    ct_data = sample_data[sample_data['ctype'] == ct]
    if len(ct_data) > 0:
        ax6.scatter(ct_data['years_num'], ct_data['ratings'], alpha=0.6, 
                   c=[colors[i]], label=ct, s=30, edgecolors='white', linewidth=0.5)

ax6.set_title('Age vs Rating Scatter\n(Colored by Company Type)', fontweight='bold', fontsize=12)
ax6.set_xlabel('Company Age (Years)')
ax6.set_ylabel('Rating')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.grid(True, alpha=0.3)

# Subplot 7: Radar chart
ax7 = plt.subplot(3, 3, 7, facecolor='white', projection='polar')
categories = ['Rating', 'Reviews', 'Age']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, ct in enumerate(company_types):
    ct_data = df_filtered[df_filtered['ctype'] == ct]
    if len(ct_data) > 0:
        values = [
            ct_data['ratings'].mean() / 5,  # Normalize to 0-1
            ct_data['review_count_norm'].mean(),
            ct_data['years_norm'].mean()
        ]
        values += values[:1]  # Complete the circle
        
        ax7.plot(angles, values, 'o-', linewidth=2, label=ct, color=colors[i])
        ax7.fill(angles, values, alpha=0.25, color=colors[i])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(categories)
ax7.set_title('Radar Chart\n(Average Metrics by Company Type)', fontweight='bold', fontsize=12, pad=20)
ax7.legend(bbox_to_anchor=(1.3, 1.1))

# Subplot 8: Treemap simulation using nested rectangles
ax8 = plt.subplot(3, 3, 8, facecolor='white')
treemap_data = []
for ct in company_types:
    ct_data = df_filtered[df_filtered['ctype'] == ct]
    if len(ct_data) > 0:
        total_reviews = ct_data['review_count_num'].sum()
        avg_rating = ct_data['ratings'].mean()
        treemap_data.append((ct, total_reviews, avg_rating))

# Sort by total reviews
treemap_data.sort(key=lambda x: x[1], reverse=True)

# Create simple rectangular representation
y_pos = 0
total_sum = sum([x[1] for x in treemap_data])
if total_sum > 0:
    for i, (ct, total_reviews, avg_rating) in enumerate(treemap_data):
        height = total_reviews / total_sum * 0.8
        color_intensity = avg_rating / 5.0  # Normalize rating for color intensity
        rect = Rectangle((0, y_pos), 1, height, facecolor=colors[i], 
                        alpha=color_intensity, edgecolor='white', linewidth=2)
        ax8.add_patch(rect)
        ax8.text(0.5, y_pos + height/2, f'{ct}\n{avg_rating:.1f}★', 
                ha='center', va='center', fontweight='bold', fontsize=10)
        y_pos += height

ax8.set_xlim(0, 1)
ax8.set_ylim(0, 0.8)
ax8.set_title('Company Type Composition\n(Size = Reviews, Intensity = Rating)', fontweight='bold', fontsize=12)
ax8.set_xticks([])
ax8.set_yticks([])

# Subplot 9: Network-style cluster plot
ax9 = plt.subplot(3, 3, 9, facecolor='white')
sample_data = df_filtered.sample(min(300, len(df_filtered)))

for i, ct in enumerate(company_types):
    ct_data = sample_data[sample_data['ctype'] == ct]
    if len(ct_data) > 0:
        x = ct_data['years_num']
        y = ct_data['ratings']
        sizes = ct_data['review_count_num'] / 50  # Scale node sizes
        
        # Plot nodes
        ax9.scatter(x, y, s=sizes, alpha=0.7, c=[colors[i]], 
                   label=ct, edgecolors='white', linewidth=1)
        
        # Add connections between nearby points (simplified clustering)
        if len(ct_data) > 1:
            center_x, center_y = x.mean(), y.mean()
            for idx, row in ct_data.iterrows():
                if abs(row['years_num'] - center_x) < 10 and abs(row['ratings'] - center_y) < 0.5:
                    ax9.plot([center_x, row['years_num']], [center_y, row['ratings']], 
                            color=colors[i], alpha=0.3, linewidth=0.5)

ax9.set_title('Network Cluster Plot\n(Node Size = Reviews, Connections = Clusters)', fontweight='bold', fontsize=12)
ax9.set_xlabel('Company Age (Years)')
ax9.set_ylabel('Rating')
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax9.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig('company_analysis_grid.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()