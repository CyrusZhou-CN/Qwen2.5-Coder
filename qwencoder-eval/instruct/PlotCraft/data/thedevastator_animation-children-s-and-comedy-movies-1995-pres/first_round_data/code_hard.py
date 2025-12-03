import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('moviewithfinaledit.csv')

# Clean column names
df.columns = ['index', 'title', 'year', 'genres']

# Filter for target genres and clean data
target_genres = ['Animation', "Children's", 'Comedy']
df_filtered = df[df['genres'].str.contains('|'.join(target_genres), na=False)].copy()
df_filtered = df_filtered.dropna(subset=['year'])
df_filtered['year'] = df_filtered['year'].astype(int)

# Create genre flags
for genre in target_genres:
    df_filtered[f'has_{genre}'] = df_filtered['genres'].str.contains(genre, na=False)

# Prepare data for analysis
yearly_counts = df_filtered.groupby('year').size().reset_index(name='count')
genre_combinations = df_filtered['genres'].value_counts().head(10)

# Calculate rolling averages
yearly_counts['rolling_3yr'] = yearly_counts['count'].rolling(window=3, center=True).mean()

# Consistent color palette for the three main genres
main_genre_colors = {
    'Animation': '#2E86AB',
    "Children's": '#A23B72', 
    'Comedy': '#F18F01'
}
additional_colors = ['#C73E1D', '#6A994E', '#7209B7', '#FF6B35', '#004E89']

# Create the 3x3 subplot grid with increased size and spacing
fig, axes = plt.subplots(3, 3, figsize=(24, 20))
fig.patch.set_facecolor('white')

# Top row - Genre Evolution Analysis
# Subplot 1: Stacked area chart with line overlay (CORRECTED - dual axis)
ax1 = axes[0, 0]
yearly_genre_counts = df_filtered.groupby(['year', 'genres']).size().unstack(fill_value=0)
top_genres = yearly_genre_counts.sum().nlargest(5).index
yearly_genre_subset = yearly_genre_counts[top_genres].fillna(0)

# Calculate cumulative counts
cumulative_data = yearly_genre_subset.cumsum(axis=1)
ax1.stackplot(yearly_genre_subset.index, *[yearly_genre_subset[col] for col in yearly_genre_subset.columns], 
              labels=[g[:20] + '...' if len(g) > 20 else g for g in yearly_genre_subset.columns], 
              alpha=0.7, colors=additional_colors[:len(yearly_genre_subset.columns)])

# Add line plot on secondary axis
ax1_twin = ax1.twinx()
ax1_twin.plot(yearly_counts['year'], yearly_counts['count'], 'k-', linewidth=3, label='Total Annual Count', marker='o', markersize=4)

ax1.set_title('Cumulative Genre Distribution Over Time', fontweight='bold', fontsize=14, pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Cumulative Count', fontsize=12)
ax1_twin.set_ylabel('Total Annual Count', fontsize=12)
ax1.legend(loc='upper left', fontsize=9)
ax1_twin.legend(loc='upper right', fontsize=9)

# Subplot 2: Bar chart with scatter overlay
ax2 = axes[0, 1]
genre_avg_year = df_filtered.groupby('genres')['year'].mean()
top_combinations = genre_combinations.head(8)
bars = ax2.bar(range(len(top_combinations)), top_combinations.values, color=additional_colors[0], alpha=0.7)
ax2_twin = ax2.twinx()
scatter_years = [genre_avg_year[genre] for genre in top_combinations.index]
ax2_twin.scatter(range(len(top_combinations)), scatter_years, color=additional_colors[1], s=120, zorder=5, edgecolors='white', linewidth=2)

ax2.set_title('Genre Combination Frequencies vs Average Release Year', fontweight='bold', fontsize=14, pad=20)
ax2.set_xlabel('Genre Combinations', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2_twin.set_ylabel('Average Release Year', fontsize=12)
ax2.set_xticks(range(len(top_combinations)))
ax2.set_xticklabels([g[:12] + '...' if len(g) > 12 else g for g in top_combinations.index], 
                   rotation=45, ha='right', fontsize=10)

# Subplot 3: Histogram with KDE and error bars
ax3 = axes[0, 2]
n, bins, patches = ax3.hist(df_filtered['year'], bins=20, alpha=0.7, color=additional_colors[0], density=True)
kde = gaussian_kde(df_filtered['year'])
x_range = np.linspace(df_filtered['year'].min(), df_filtered['year'].max(), 100)
ax3.plot(x_range, kde(x_range), color=additional_colors[1], linewidth=3, label='KDE')

# Add error bars for confidence intervals
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_counts = np.histogram(df_filtered['year'], bins=bins)[0]
errors = np.sqrt(bin_counts) / len(df_filtered) * (bins[1] - bins[0])
ax3.errorbar(bin_centers, n, yerr=errors, fmt='none', color='black', alpha=0.6, capsize=4, linewidth=2)

ax3.set_title('Movie Release Distribution with Confidence Intervals', fontweight='bold', fontsize=14, pad=20)
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.legend(fontsize=10)

# Middle row - Temporal Distribution Patterns
# Subplot 4: Box plot with violin overlay
ax4 = axes[1, 0]
genre_data = []
genre_labels = []
genre_colors = []
for genre in target_genres:
    genre_years = df_filtered[df_filtered[f'has_{genre}']]['year']
    if len(genre_years) > 0:
        genre_data.append(genre_years)
        genre_labels.append(genre)
        genre_colors.append(main_genre_colors[genre])

# Create box plot
box_plot = ax4.boxplot(genre_data, labels=genre_labels, patch_artist=True)
for i, patch in enumerate(box_plot['boxes']):
    patch.set_facecolor(genre_colors[i])
    patch.set_alpha(0.7)

# Overlay violin plots
for i, data in enumerate(genre_data):
    if len(data) > 1:
        violin_parts = ax4.violinplot([data], positions=[i+1], widths=0.6, showmeans=True)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(genre_colors[i])
            pc.set_alpha(0.3)

ax4.set_title('Genre Release Year Distribution', fontweight='bold', fontsize=14, pad=20)
ax4.set_xlabel('Genre', fontsize=12)
ax4.set_ylabel('Release Year', fontsize=12)

# Subplot 5: Heatmap with line overlay (IMPROVED contrast)
ax5 = axes[1, 1]
# Create correlation matrix between years and genre presence
year_genre_matrix = df_filtered.pivot_table(
    index='year', 
    columns='genres', 
    values='title', 
    aggfunc='count', 
    fill_value=0
)

# Select top genres by frequency
top_genre_cols = year_genre_matrix.sum().nlargest(6).index
year_genre_matrix = year_genre_matrix[top_genre_cols]

correlation_matrix = year_genre_matrix.corr()
im = ax5.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto')

# Improved label handling
short_labels = [col[:8] + '...' if len(col) > 8 else col for col in correlation_matrix.columns]
ax5.set_xticks(range(len(correlation_matrix.columns)))
ax5.set_yticks(range(len(correlation_matrix.columns)))
ax5.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
ax5.set_yticklabels(short_labels, fontsize=10)

# Add diversity index line with better contrast
ax5_twin = ax5.twinx()
diversity_index = year_genre_matrix.apply(lambda x: -sum(p * np.log(p) for p in x/x.sum() if p > 0), axis=1)
years_normalized = np.linspace(0, len(correlation_matrix)-1, len(diversity_index))
ax5_twin.plot(years_normalized, diversity_index.values, color='black', linewidth=4, alpha=0.8, label='Diversity Index')
ax5_twin.legend(loc='upper right', fontsize=9)

ax5.set_title('Genre Correlation Heatmap with Diversity Index', fontweight='bold', fontsize=14, pad=20)

# Subplot 6: Dual-axis bar and line chart
ax6 = axes[1, 2]
bars = ax6.bar(yearly_counts['year'], yearly_counts['count'], alpha=0.7, color=additional_colors[0])
ax6_twin = ax6.twinx()
valid_rolling = yearly_counts.dropna(subset=['rolling_3yr'])
ax6_twin.plot(valid_rolling['year'], valid_rolling['rolling_3yr'], color=additional_colors[1], 
              linewidth=3, marker='o', markersize=5, label='3-Year Rolling Average')

ax6.set_title('Annual Counts vs 3-Year Rolling Average', fontweight='bold', fontsize=14, pad=20)
ax6.set_xlabel('Year', fontsize=12)
ax6.set_ylabel('Annual Count', fontsize=12)
ax6_twin.set_ylabel('3-Year Rolling Average', fontsize=12)
ax6_twin.legend(loc='upper right', fontsize=9)

# Bottom row - Comparative Trend Analysis
# Subplot 7: Multiple line plots with confidence bands
ax7 = axes[2, 0]
for i, genre in enumerate(target_genres):
    genre_yearly = df_filtered[df_filtered[f'has_{genre}']].groupby('year').size()
    if len(genre_yearly) > 3:
        years = genre_yearly.index
        counts = genre_yearly.values
        
        # Smooth the line
        z = np.polyfit(years, counts, min(2, len(years)-1))
        p = np.poly1d(z)
        smooth_years = np.linspace(years.min(), years.max(), 50)
        smooth_counts = p(smooth_years)
        
        ax7.plot(smooth_years, smooth_counts, color=main_genre_colors[genre], 
                linewidth=3, label=genre, marker='o', markersize=4, markevery=10)
        
        # Add confidence band
        std_dev = np.std(counts)
        ax7.fill_between(smooth_years, smooth_counts - std_dev, smooth_counts + std_dev, 
                        color=main_genre_colors[genre], alpha=0.2)

ax7.set_title('Genre Trends with Confidence Bands', fontweight='bold', fontsize=14, pad=20)
ax7.set_xlabel('Year', fontsize=12)
ax7.set_ylabel('Movie Count', fontsize=12)
ax7.legend(fontsize=11)
ax7.grid(True, alpha=0.3)

# Subplot 8: Slope chart with dot plots
ax8 = axes[2, 1]
early_period = df_filtered[df_filtered['year'].between(1995, 2005)]
recent_period = df_filtered[df_filtered['year'].between(2015, 2023)]

early_counts = {genre: early_period[early_period[f'has_{genre}']].shape[0] for genre in target_genres}
recent_counts = {genre: recent_period[recent_period[f'has_{genre}']].shape[0] for genre in target_genres}

for i, genre in enumerate(target_genres):
    ax8.plot([0, 1], [early_counts[genre], recent_counts[genre]], 
             color=main_genre_colors[genre], linewidth=4, marker='o', markersize=10, 
             label=genre, markeredgecolor='white', markeredgewidth=2)

ax8.set_xlim(-0.15, 1.15)
ax8.set_xticks([0, 1])
ax8.set_xticklabels(['1995-2005', '2015-Present'], fontsize=12)
ax8.set_title('Genre Distribution: Early vs Recent Period', fontweight='bold', fontsize=14, pad=20)
ax8.set_ylabel('Movie Count', fontsize=12)
ax8.legend(fontsize=11, loc='center left')
ax8.grid(True, alpha=0.3)

# Subplot 9: Time series decomposition (COMPLETELY REWORKED)
ax9 = axes[2, 2]

# Create a proper time series for decomposition
full_years = range(df_filtered['year'].min(), df_filtered['year'].max() + 1)
total_yearly = df_filtered.groupby('year').size().reindex(full_years, fill_value=0)

# Perform decomposition if we have enough data points
if len(total_yearly) >= 8:  # Need sufficient data for decomposition
    try:
        # Use additive decomposition
        decomposition = seasonal_decompose(total_yearly, model='additive', period=min(4, len(total_yearly)//2))
        
        # Plot trend component
        ax9.plot(total_yearly.index, decomposition.trend, color=additional_colors[0], 
                linewidth=3, label='Trend', marker='o', markersize=4)
        
        # Plot seasonal component (scaled and shifted for visibility)
        seasonal_scaled = decomposition.seasonal * 0.5 + np.nanmean(decomposition.trend)
        ax9.plot(total_yearly.index, seasonal_scaled, color=additional_colors[1], 
                linewidth=2, linestyle='--', label='Seasonal (scaled)', alpha=0.8)
        
        # Plot residual component (scaled and shifted for visibility)
        residual_scaled = decomposition.resid * 0.3 + np.nanmean(decomposition.trend)
        ax9.plot(total_yearly.index, residual_scaled, color=additional_colors[2], 
                linewidth=2, linestyle=':', label='Residual (scaled)', alpha=0.7)
        
        # Plot observed data
        ax9.plot(total_yearly.index, total_yearly.values, color='black', 
                linewidth=2, alpha=0.6, label='Observed')
        
    except:
        # Fallback if decomposition fails
        ax9.plot(total_yearly.index, total_yearly.values, color='black', 
                linewidth=3, label='Observed Data', marker='o', markersize=4)
        
        # Simple trend line
        z = np.polyfit(total_yearly.index, total_yearly.values, 1)
        trend_line = np.poly1d(z)(total_yearly.index)
        ax9.plot(total_yearly.index, trend_line, color=additional_colors[0], 
                linewidth=3, linestyle='--', label='Linear Trend')
else:
    # Simple plot if not enough data
    ax9.plot(total_yearly.index, total_yearly.values, color='black', 
            linewidth=3, label='Total Movies', marker='o', markersize=4)

ax9.set_title('Time Series Decomposition of Movie Releases', fontweight='bold', fontsize=14, pad=20)
ax9.set_xlabel('Year', fontsize=12)
ax9.set_ylabel('Components', fontsize=12)
ax9.legend(fontsize=10, loc='upper left')
ax9.grid(True, alpha=0.3)

# Final layout adjustments with increased spacing
plt.tight_layout(pad=4.0)
plt.subplots_adjust(hspace=0.4, wspace=0.35, left=0.08, right=0.95, top=0.95, bottom=0.08)

plt.show()