import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('Clean_Top_1000_Youtube_df - youtubers_df.csv')

# Convert string numbers with commas to integers
def convert_to_int(value):
    if isinstance(value, str):
        return int(value.replace(',', ''))
    return int(value)

# Convert numerical columns
numerical_cols = ['Suscribers', 'Visits', 'Likes', 'Comments']
for col in numerical_cols:
    df[col] = df[col].apply(convert_to_int)

# Rename for consistency
df = df.rename(columns={'Suscribers': 'Subscribers'})

# Get top 5 countries by channel count
top_countries = df['Country'].value_counts().head(5).index.tolist()
df_top_countries = df[df['Country'].isin(top_countries)]

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')

# Define color palette for categories
categories = df['Categories'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
category_colors = dict(zip(categories, colors))

# Row 1, Subplot 1: Scatter plot with regression line
ax1 = plt.subplot(3, 3, 1)
# Main scatter plot
for i, cat in enumerate(categories):
    cat_data = df[df['Categories'] == cat]
    ax1.scatter(cat_data['Subscribers'], cat_data['Visits'], 
               c=[colors[i]], alpha=0.6, s=30, label=cat)

# Add regression line
z = np.polyfit(df['Subscribers'], df['Visits'], 1)
p = np.poly1d(z)
ax1.plot(df['Subscribers'], p(df['Subscribers']), "r--", alpha=0.8, linewidth=2)

ax1.set_xlabel('Subscribers', fontweight='bold')
ax1.set_ylabel('Visits', fontweight='bold')
ax1.set_title('Subscribers vs Visits with Regression Line', fontweight='bold', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

# Row 1, Subplot 2: Bubble chart with trend line
ax2 = plt.subplot(3, 3, 2)
# Normalize comment sizes for bubble sizes
bubble_sizes = (df['Comments'] / df['Comments'].max()) * 200 + 20

scatter = ax2.scatter(df['Subscribers'], df['Likes'], s=bubble_sizes, 
                     alpha=0.6, c=df['Comments'], cmap='viridis')

# Add trend line
z = np.polyfit(df['Subscribers'], df['Likes'], 1)
p = np.poly1d(z)
ax2.plot(df['Subscribers'], p(df['Subscribers']), 'r-', linewidth=2, alpha=0.8)

ax2.set_xlabel('Subscribers', fontweight='bold')
ax2.set_ylabel('Likes', fontweight='bold')
ax2.set_title('Bubble Chart: Subscribers vs Likes\n(Bubble size = Comments)', fontweight='bold', fontsize=10)
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Comments')

# Row 1, Subplot 3: Hexbin plot
ax3 = plt.subplot(3, 3, 3)
hb = ax3.hexbin(df['Visits'], df['Likes'], gridsize=20, cmap='Blues', alpha=0.7)
ax3.set_xlabel('Visits', fontweight='bold')
ax3.set_ylabel('Likes', fontweight='bold')
ax3.set_title('Hexbin Plot: Visits vs Likes', fontweight='bold', fontsize=10)
plt.colorbar(hb, ax=ax3, label='Count')

# Row 2, Subplot 4: Correlation heatmap
ax4 = plt.subplot(3, 3, 4)
corr_data = df[['Subscribers', 'Visits', 'Likes', 'Comments']].corr()
sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
            square=True, ax=ax4, cbar_kws={'shrink': 0.8}, fmt='.3f')
ax4.set_title('Correlation Heatmap\nEngagement Metrics', fontweight='bold', fontsize=10)

# Row 2, Subplot 5: Violin plot with strip plot overlay
ax5 = plt.subplot(3, 3, 5)
# Create violin plot data
violin_data = []
positions = []
labels = []

for i, country in enumerate(top_countries):
    country_data = df_top_countries[df_top_countries['Country'] == country]['Subscribers']
    if len(country_data) > 0:
        violin_data.append(country_data.values)
        positions.append(i)
        labels.append(country)

if violin_data:
    parts = ax5.violinplot(violin_data, positions=positions, showmeans=True)
    
    # Overlay strip plot
    for i, country in enumerate(labels):
        country_data = df_top_countries[df_top_countries['Country'] == country]['Subscribers']
        y_pos = np.random.normal(i, 0.04, size=len(country_data))
        ax5.scatter(y_pos, country_data, alpha=0.4, s=20)

ax5.set_xticks(positions)
ax5.set_xticklabels(labels, rotation=45)
ax5.set_ylabel('Subscribers', fontweight='bold')
ax5.set_title('Subscribers Distribution\nby Top 5 Countries', fontweight='bold', fontsize=10)
ax5.grid(True, alpha=0.3)

# Row 2, Subplot 6: Parallel coordinates plot (simplified)
ax6 = plt.subplot(3, 3, 6)
# Normalize the data for parallel coordinates
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
metrics = ['Subscribers', 'Visits', 'Likes', 'Comments']
normalized_data = scaler.fit_transform(df[metrics])

# Sample data for readability
sample_indices = np.random.choice(len(df), size=min(100, len(df)), replace=False)
sample_data = normalized_data[sample_indices]
sample_categories = df.iloc[sample_indices]['Categories'].values

# Plot lines for each sample
for i, cat in enumerate(sample_categories):
    color_idx = list(categories).index(cat) if cat in categories else 0
    ax6.plot(range(len(metrics)), sample_data[i], 
            color=colors[color_idx], alpha=0.3, linewidth=1)

ax6.set_xticks(range(len(metrics)))
ax6.set_xticklabels(metrics, rotation=45)
ax6.set_ylabel('Normalized Values', fontweight='bold')
ax6.set_title('Parallel Coordinates Plot\n(Normalized Metrics)', fontweight='bold', fontsize=10)
ax6.grid(True, alpha=0.3)

# Row 3, Subplot 7: Scatter plot matrix style
ax7 = plt.subplot(3, 3, 7)
# Focus on Subscribers vs Visits with category markers
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
for i, cat in enumerate(categories[:min(len(markers), len(categories))]):
    cat_data = df[df['Categories'] == cat]
    marker_idx = i % len(markers)
    ax7.scatter(cat_data['Subscribers'], cat_data['Visits'], 
               marker=markers[marker_idx], c=[colors[i]], 
               alpha=0.6, s=40, label=cat)

ax7.set_xlabel('Subscribers', fontweight='bold')
ax7.set_ylabel('Visits', fontweight='bold')
ax7.set_title('Scatter Matrix Style:\nSubscribers vs Visits by Category', fontweight='bold', fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

# Row 3, Subplot 8: Box plot with swarm plot overlay
ax8 = plt.subplot(3, 3, 8)
# Select top categories for readability
top_categories = df['Categories'].value_counts().head(5).index.tolist()
df_top_cats = df[df['Categories'].isin(top_categories)]

box_data = []
box_labels = []
for cat in top_categories:
    cat_data = df_top_cats[df_top_cats['Categories'] == cat]['Likes']
    if len(cat_data) > 0:
        box_data.append(cat_data.values)
        box_labels.append(cat)

if box_data:
    bp = ax8.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        cat_idx = list(categories).index(box_labels[i]) if box_labels[i] in categories else 0
        patch.set_facecolor(colors[cat_idx])
        patch.set_alpha(0.7)

ax8.set_xticklabels(box_labels, rotation=45, ha='right')
ax8.set_ylabel('Likes', fontweight='bold')
ax8.set_title('Box Plot: Likes Distribution\nby Categories', fontweight='bold', fontsize=10)
ax8.grid(True, alpha=0.3)

# Row 3, Subplot 9: Network-style correlation plot (simplified)
ax9 = plt.subplot(3, 3, 9)
# Create a simple network visualization of correlations
corr_matrix = df[['Subscribers', 'Visits', 'Likes', 'Comments']].corr()
metrics = ['Subscribers', 'Visits', 'Likes', 'Comments']

# Position nodes in a circle
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
pos = {metric: (np.cos(angle), np.sin(angle)) for metric, angle in zip(metrics, angles)}

# Draw nodes
for metric, (x, y) in pos.items():
    ax9.scatter(x, y, s=1000, c='lightblue', alpha=0.8, edgecolors='black')
    ax9.text(x, y, metric, ha='center', va='center', fontweight='bold', fontsize=8)

# Draw edges based on correlation strength
for i in range(len(metrics)):
    for j in range(i+1, len(metrics)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val > 0.1:  # Only show significant correlations
            x1, y1 = pos[metrics[i]]
            x2, y2 = pos[metrics[j]]
            ax9.plot([x1, x2], [y1, y2], 'gray', alpha=0.6, linewidth=corr_val*5)

ax9.set_xlim(-1.5, 1.5)
ax9.set_ylim(-1.5, 1.5)
ax9.set_title('Network Correlation Plot\n(Line thickness = Correlation strength)', 
             fontweight='bold', fontsize=10)
ax9.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Save the plot
plt.savefig('youtube_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()