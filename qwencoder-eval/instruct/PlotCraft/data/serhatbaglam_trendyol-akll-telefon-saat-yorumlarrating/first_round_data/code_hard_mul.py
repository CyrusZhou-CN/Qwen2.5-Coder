import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
datasets = {
    'iPhone': pd.read_excel('trendyol_iphone_yorum.xlsx'),
    'Samsung Phone': pd.read_excel('trendyol_samsung_telefon_yorum.xlsx'),
    'Xiaomi Phone': pd.read_excel('trendyol_xiaomi_yorum_rating.xlsx'),
    'Reeder': pd.read_excel('trendyol_reeder_yorum_rating.xlsx'),
    'Apple Watch': pd.read_excel('trendyol_apple_watch_yorum_rating.xlsx'),
    'Samsung Watch': pd.read_excel('trendyol_samsung_watch_yorum_rating.xlsx'),
    'Huawei Watch': pd.read_excel('trendyol_huawei_saat_yorum_rating.xlsx'),
    'Mateo Watch': pd.read_excel('trendyol_mateo_saat_yorum_rating.xlsx'),
    'Xiaomi Watch': pd.read_excel('trendyol_xiaomi_saat_yorum_rating.xlsx')
}

# Process data for analysis
def process_brand_data(datasets, brand_list):
    brand_stats = []
    for brand in brand_list:
        df = datasets[brand]
        avg_rating = df['Yıldız'].mean()
        review_count = len(df)
        rating_std = df['Yıldız'].std()
        ratings = df['Yıldız'].values
        
        brand_stats.append({
            'Brand': brand,
            'Avg_Rating': avg_rating,
            'Review_Count': review_count,
            'Rating_Std': rating_std,
            'Ratings': ratings
        })
    return brand_stats

# Define brand categories
smartphone_brands = ['iPhone', 'Samsung Phone', 'Xiaomi Phone', 'Reeder']
smartwatch_brands = ['Apple Watch', 'Samsung Watch', 'Huawei Watch', 'Mateo Watch', 'Xiaomi Watch']

# Process data
phone_stats = process_brand_data(datasets, smartphone_brands)
watch_stats = process_brand_data(datasets, smartwatch_brands)

# Create the comprehensive 3x2 subplot grid
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Define colors for brands
phone_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
watch_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# TOP ROW: Scatter plots with marginal histograms
# Smartphone scatter plot (Top Left)
ax1 = plt.subplot(3, 2, 1)
for i, brand_data in enumerate(phone_stats):
    x = brand_data['Review_Count']
    y = brand_data['Avg_Rating']
    ax1.scatter(x, y, s=300, alpha=0.8, color=phone_colors[i], 
               label=brand_data['Brand'], edgecolors='white', linewidth=2)

# Add trend line for phones
x_vals = [brand['Review_Count'] for brand in phone_stats]
y_vals = [brand['Avg_Rating'] for brand in phone_stats]
z = np.polyfit(x_vals, y_vals, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(x_vals), max(x_vals), 100)
ax1.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.7, linewidth=2)

ax1.set_xlabel('Total Review Count', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
ax1.set_title('Smartphone Brands: Rating vs Review Volume', fontsize=14, fontweight='bold')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(3.5, 5.2)

# Smartwatch scatter plot (Top Right)
ax2 = plt.subplot(3, 2, 2)
for i, brand_data in enumerate(watch_stats):
    x = brand_data['Review_Count']
    y = brand_data['Avg_Rating']
    ax2.scatter(x, y, s=300, alpha=0.8, color=watch_colors[i], 
               label=brand_data['Brand'], edgecolors='white', linewidth=2)

# Add trend line for watches
x_vals = [brand['Review_Count'] for brand in watch_stats]
y_vals = [brand['Avg_Rating'] for brand in watch_stats]
z = np.polyfit(x_vals, y_vals, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(x_vals), max(x_vals), 100)
ax2.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.7, linewidth=2)

ax2.set_xlabel('Total Review Count', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
ax2.set_title('Smartwatch Brands: Rating vs Review Volume', fontsize=14, fontweight='bold')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(3.5, 5.2)

# MIDDLE ROW: Correlation heatmaps with bubble plots
# Smartphone correlation heatmap (Middle Left)
ax3 = plt.subplot(3, 2, 3)

# Create correlation matrix for phones
phone_corr_data = []
for brand_data in phone_stats:
    phone_corr_data.append([
        brand_data['Avg_Rating'],
        brand_data['Review_Count'],
        brand_data['Rating_Std']
    ])

phone_corr_df = pd.DataFrame(phone_corr_data, 
                            columns=['Avg Rating', 'Review Count', 'Rating Std'],
                            index=[brand['Brand'] for brand in phone_stats])

# Create correlation matrix
corr_matrix = phone_corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax3)
ax3.set_title('Smartphone Correlation Matrix', fontsize=14, fontweight='bold')

# Add bubble overlay
for i, brand_data in enumerate(phone_stats):
    bubble_size = brand_data['Review_Count'] / 100
    ax3.scatter(1.5, i + 0.5, s=bubble_size, alpha=0.6, color=phone_colors[i])

# Smartwatch correlation heatmap (Middle Right)
ax4 = plt.subplot(3, 2, 4)

# Create correlation matrix for watches
watch_corr_data = []
for brand_data in watch_stats:
    watch_corr_data.append([
        brand_data['Avg_Rating'],
        brand_data['Review_Count'],
        brand_data['Rating_Std']
    ])

watch_corr_df = pd.DataFrame(watch_corr_data, 
                            columns=['Avg Rating', 'Review Count', 'Rating Std'],
                            index=[brand['Brand'] for brand in watch_stats])

# Create correlation matrix
corr_matrix = watch_corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax4)
ax4.set_title('Smartwatch Correlation Matrix', fontsize=14, fontweight='bold')

# Add bubble overlay
for i, brand_data in enumerate(watch_stats):
    bubble_size = brand_data['Review_Count'] / 100
    ax4.scatter(1.5, i + 0.5, s=bubble_size, alpha=0.6, color=watch_colors[i])

# BOTTOM ROW: Violin plots with box plots and outliers
# Smartphone violin plot (Bottom Left)
ax5 = plt.subplot(3, 2, 5)

phone_ratings_data = []
phone_labels = []
for i, brand_data in enumerate(phone_stats):
    phone_ratings_data.append(brand_data['Ratings'])
    phone_labels.append(brand_data['Brand'])

# Create violin plot
parts = ax5.violinplot(phone_ratings_data, positions=range(len(phone_labels)), 
                      showmeans=True, showmedians=True)

# Color the violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(phone_colors[i])
    pc.set_alpha(0.7)

# Overlay box plots
bp = ax5.boxplot(phone_ratings_data, positions=range(len(phone_labels)), 
                widths=0.3, patch_artist=True, showfliers=True)

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(phone_colors[i])
    patch.set_alpha(0.5)

# Add outlier detection and confidence intervals
for i, brand_data in enumerate(phone_stats):
    ratings = brand_data['Ratings']
    mean_rating = np.mean(ratings)
    std_rating = np.std(ratings)
    ci_lower = mean_rating - 1.96 * std_rating / np.sqrt(len(ratings))
    ci_upper = mean_rating + 1.96 * std_rating / np.sqrt(len(ratings))
    
    # Plot confidence interval
    ax5.plot([i-0.1, i+0.1], [ci_lower, ci_lower], 'k-', linewidth=2)
    ax5.plot([i-0.1, i+0.1], [ci_upper, ci_upper], 'k-', linewidth=2)
    ax5.plot([i, i], [ci_lower, ci_upper], 'k-', linewidth=1)

ax5.set_xticks(range(len(phone_labels)))
ax5.set_xticklabels(phone_labels, rotation=45, ha='right')
ax5.set_ylabel('Rating Distribution', fontsize=12, fontweight='bold')
ax5.set_title('Smartphone Rating Variance Analysis', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Smartwatch violin plot (Bottom Right)
ax6 = plt.subplot(3, 2, 6)

watch_ratings_data = []
watch_labels = []
for i, brand_data in enumerate(watch_stats):
    watch_ratings_data.append(brand_data['Ratings'])
    watch_labels.append(brand_data['Brand'])

# Create violin plot
parts = ax6.violinplot(watch_ratings_data, positions=range(len(watch_labels)), 
                      showmeans=True, showmedians=True)

# Color the violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(watch_colors[i])
    pc.set_alpha(0.7)

# Overlay box plots
bp = ax6.boxplot(watch_ratings_data, positions=range(len(watch_labels)), 
                widths=0.3, patch_artist=True, showfliers=True)

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(watch_colors[i])
    patch.set_alpha(0.5)

# Add outlier detection and confidence intervals
for i, brand_data in enumerate(watch_stats):
    ratings = brand_data['Ratings']
    mean_rating = np.mean(ratings)
    std_rating = np.std(ratings)
    ci_lower = mean_rating - 1.96 * std_rating / np.sqrt(len(ratings))
    ci_upper = mean_rating + 1.96 * std_rating / np.sqrt(len(ratings))
    
    # Plot confidence interval
    ax6.plot([i-0.1, i+0.1], [ci_lower, ci_lower], 'k-', linewidth=2)
    ax6.plot([i-0.1, i+0.1], [ci_upper, ci_upper], 'k-', linewidth=2)
    ax6.plot([i, i], [ci_lower, ci_upper], 'k-', linewidth=1)

ax6.set_xticks(range(len(watch_labels)))
ax6.set_xticklabels(watch_labels, rotation=45, ha='right')
ax6.set_ylabel('Rating Distribution', fontsize=12, fontweight='bold')
ax6.set_title('Smartwatch Rating Variance Analysis', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Overall layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# Add overall title
fig.suptitle('Comprehensive Brand Performance Analysis: Smartphones vs Smartwatches', 
             fontsize=18, fontweight='bold', y=0.98)

plt.show()