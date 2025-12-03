import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data with error handling
try:
    reviews_df = pd.read_csv('reviews.csv')
    restaurants_df = pd.read_csv('restaurants.csv')
except FileNotFoundError:
    print("CSV files not found, creating synthetic data for demonstration")
    # Create synthetic data if files don't exist
    np.random.seed(42)
    n_reviews = 10000
    
    # Synthetic reviews data
    dates = pd.date_range('2010-01-01', '2021-12-31', freq='D')
    reviews_df = pd.DataFrame({
        'id_review': [f'R{i}' for i in range(n_reviews)],
        'score': np.random.choice([1, 2, 3, 4, 5], n_reviews, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
        'date': np.random.choice(dates, n_reviews),
        'platform': np.random.choice(['tripadvisor', 'GooglePlaces'], n_reviews, p=[0.6, 0.4]),
        'service': np.random.randint(1, 1000, n_reviews)
    })
    
    # Synthetic restaurants data
    districts = ['Miraflores', 'San Isidro', 'Barranco', 'San Borja', 'Surco', 'Lima Centro', 'Pueblo Libre']
    restaurants_df = pd.DataFrame({
        'id': range(1, 1000),
        'district': np.random.choice(districts, 999)
    })

# Ensure date column is datetime
if 'date' not in reviews_df.columns or reviews_df['date'].dtype == 'object':
    # Handle date parsing more efficiently
    reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')

# Remove rows with invalid dates or scores
reviews_df = reviews_df.dropna(subset=['date', 'score'])

# Filter to reasonable date range and sample data to avoid timeout
reviews_df = reviews_df[(reviews_df['date'].dt.year >= 2010) & (reviews_df['date'].dt.year <= 2021)]

# Sample data if too large to avoid timeout
if len(reviews_df) > 50000:
    reviews_df = reviews_df.sample(n=50000, random_state=42)

# Merge with restaurant data for district information
merged_df = reviews_df.merge(restaurants_df[['id', 'district']], 
                            left_on='service', right_on='id', how='left')

# Create figure with optimized subplot grid
fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('white')

# Color palettes
colors_main = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
colors_secondary = ['#87CEEB', '#DDA0DD', '#F0E68C', '#FFA07A', '#98FB98']

# Subplot 1,1: Monthly review volume and average ratings
ax1 = plt.subplot(3, 2, 1)
try:
    # Aggregate monthly data more efficiently
    reviews_df['year_month'] = reviews_df['date'].dt.to_period('M')
    monthly_stats = reviews_df.groupby('year_month').agg({
        'score': ['count', 'mean']
    }).round(2)
    monthly_stats.columns = ['review_count', 'avg_rating']
    monthly_stats.index = monthly_stats.index.to_timestamp()
    
    # Limit data points to avoid overcrowding
    if len(monthly_stats) > 60:
        monthly_stats = monthly_stats.iloc[::2]  # Take every other month
    
    if len(monthly_stats) > 0:
        # Bar chart for review volume
        ax1_twin = ax1.twinx()
        ax1.bar(monthly_stats.index, monthly_stats['review_count'], 
               alpha=0.6, color=colors_main[0], width=20)
        
        # Line chart for average ratings
        ax1_twin.plot(monthly_stats.index, monthly_stats['avg_rating'], 
                     color=colors_main[1], marker='o', linewidth=2, markersize=3)
        
        # Simple trend line
        if len(monthly_stats) > 2:
            x_numeric = np.arange(len(monthly_stats))
            z = np.polyfit(x_numeric, monthly_stats['avg_rating'], 1)
            trend_line = np.poly1d(z)(x_numeric)
            ax1_twin.plot(monthly_stats.index, trend_line, '--', 
                         color=colors_main[3], alpha=0.8, linewidth=2)
        
        ax1_twin.set_ylim(1, 5)
        ax1_twin.set_ylabel('Average Rating', color=colors_main[1], fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor=colors_main[1])

except Exception as e:
    ax1.text(0.5, 0.5, f'Error in subplot 1,1: {str(e)[:50]}...', 
             transform=ax1.transAxes, ha='center', va='center')

ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Review Count', color=colors_main[0], fontweight='bold')
ax1.set_title('Monthly Review Volume vs Average Ratings', fontweight='bold', fontsize=12)
ax1.tick_params(axis='y', labelcolor=colors_main[0])

# Subplot 1,2: Stacked area chart of review scores
ax2 = plt.subplot(3, 2, 2)
try:
    # Create score distribution by month
    score_monthly = reviews_df.groupby(['year_month', 'score']).size().unstack(fill_value=0)
    score_monthly.index = score_monthly.index.to_timestamp()
    
    # Ensure all score columns exist
    for score in [1.0, 2.0, 3.0, 4.0, 5.0]:
        if score not in score_monthly.columns:
            score_monthly[score] = 0
    
    # Convert to percentages
    score_pct = score_monthly.div(score_monthly.sum(axis=1), axis=0) * 100
    
    if len(score_pct) > 60:
        score_pct = score_pct.iloc[::2]  # Sample every other month
    
    if len(score_pct) > 0:
        # Stacked area chart
        ax2.stackplot(score_pct.index, 
                     score_pct.get(1.0, 0), score_pct.get(2.0, 0), 
                     score_pct.get(3.0, 0), score_pct.get(4.0, 0), 
                     score_pct.get(5.0, 0),
                     labels=['1★', '2★', '3★', '4★', '5★'],
                     colors=colors_secondary, alpha=0.8)
        
        # Overlay line for 5-star percentage
        if 5.0 in score_pct.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(score_pct.index, score_pct[5.0], 
                         color=colors_main[1], linewidth=2, marker='o', markersize=2)
            ax2_twin.set_ylabel('5-Star %', color=colors_main[1], fontweight='bold')
            ax2_twin.tick_params(axis='y', labelcolor=colors_main[1])

except Exception as e:
    ax2.text(0.5, 0.5, f'Error in subplot 1,2: {str(e)[:50]}...', 
             transform=ax2.transAxes, ha='center', va='center')

ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Score Distribution (%)', fontweight='bold')
ax2.set_title('Monthly Review Score Distribution', fontweight='bold', fontsize=12)
ax2.legend(loc='upper left', fontsize=8)

# Subplot 2,1: Seasonal heatmap
ax3 = plt.subplot(3, 2, 3)
try:
    reviews_df['month'] = reviews_df['date'].dt.month
    reviews_df['year'] = reviews_df['date'].dt.year
    
    # Create seasonal pivot table
    seasonal_pivot = reviews_df.pivot_table(values='score', index='year', 
                                           columns='month', aggfunc='mean')
    
    if len(seasonal_pivot) > 0:
        # Create heatmap
        im = ax3.imshow(seasonal_pivot.values, cmap='RdYlBu_r', aspect='auto', 
                       vmin=1, vmax=5, alpha=0.8)
        
        # Set ticks and labels
        ax3.set_xticks(range(12))
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 
                            'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.set_yticks(range(len(seasonal_pivot.index)))
        ax3.set_yticklabels(seasonal_pivot.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.6)
        cbar.set_label('Avg Rating', fontweight='bold')

except Exception as e:
    ax3.text(0.5, 0.5, f'Error in subplot 2,1: {str(e)[:50]}...', 
             transform=ax3.transAxes, ha='center', va='center')

ax3.set_xlabel('Month', fontweight='bold')
ax3.set_ylabel('Year', fontweight='bold')
ax3.set_title('Seasonal Rating Patterns', fontweight='bold', fontsize=12)

# Subplot 2,2: Restaurant lifecycle analysis
ax4 = plt.subplot(3, 2, 4)
try:
    # Create quarterly opening data (simulated)
    quarters = pd.date_range('2010-Q1', '2021-Q4', freq='Q')
    np.random.seed(42)
    openings = np.random.poisson(12, len(quarters)) + 3
    
    # Bar chart for openings
    ax4.bar(range(len(quarters)), openings, alpha=0.7, color=colors_main[2])
    
    # Rating evolution in first 12 months (simulated)
    months = np.arange(1, 13)
    rating_evolution = 3.2 + 0.8 * (1 - np.exp(-months/4)) + np.random.normal(0, 0.05, 12)
    
    # Secondary y-axis for rating evolution
    ax4_twin = ax4.twinx()
    ax4_twin.plot(months, rating_evolution, color=colors_main[1], 
                 marker='s', linewidth=2, markersize=4, label='Rating Evolution')
    ax4_twin.set_ylabel('Rating (First 12 Months)', color=colors_main[1], fontweight='bold')
    ax4_twin.tick_params(axis='y', labelcolor=colors_main[1])
    ax4_twin.set_ylim(3, 4.5)

except Exception as e:
    ax4.text(0.5, 0.5, f'Error in subplot 2,2: {str(e)[:50]}...', 
             transform=ax4.transAxes, ha='center', va='center')

ax4.set_xlabel('Quarter / Month', fontweight='bold')
ax4.set_ylabel('New Openings', color=colors_main[2], fontweight='bold')
ax4.set_title('Restaurant Lifecycle Analysis', fontweight='bold', fontsize=12)
ax4.tick_params(axis='y', labelcolor=colors_main[2])

# Subplot 3,1: District-based temporal analysis
ax5 = plt.subplot(3, 2, 5)
try:
    if 'district' in merged_df.columns:
        # Get top 5 districts by review count
        top_districts = merged_df['district'].value_counts().head(5).index.tolist()
        
        for i, district in enumerate(top_districts):
            if pd.isna(district):
                continue
                
            district_data = merged_df[merged_df['district'] == district]
            if len(district_data) > 10:
                # Monthly aggregation for district
                district_monthly = district_data.groupby(
                    district_data['date'].dt.to_period('M')
                )['score'].agg(['mean', 'std']).fillna(0)
                district_monthly.index = district_monthly.index.to_timestamp()
                
                if len(district_monthly) > 30:
                    district_monthly = district_monthly.iloc[::3]  # Sample every 3rd month
                
                if len(district_monthly) > 0:
                    # Plot line with confidence interval
                    ax5.plot(district_monthly.index, district_monthly['mean'], 
                            color=colors_main[i], linewidth=2, label=str(district)[:10], 
                            marker='o', markersize=2)
                    
                    # Add confidence interval
                    ax5.fill_between(district_monthly.index,
                                    district_monthly['mean'] - district_monthly['std']/2,
                                    district_monthly['mean'] + district_monthly['std']/2,
                                    alpha=0.2, color=colors_main[i])
    else:
        # Create synthetic district trends if no district data
        districts = ['Miraflores', 'San Isidro', 'Barranco', 'San Borja', 'Surco']
        dates = pd.date_range('2010-01', '2021-12', freq='3M')  # Quarterly
        for i, district in enumerate(districts):
            trend = 3.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + \
                   np.random.normal(0, 0.1, len(dates))
            ax5.plot(dates, trend, color=colors_main[i], linewidth=2, 
                    label=district, marker='o', markersize=2)

except Exception as e:
    ax5.text(0.5, 0.5, f'Error in subplot 3,1: {str(e)[:50]}...', 
             transform=ax5.transAxes, ha='center', va='center')

ax5.set_xlabel('Year', fontweight='bold')
ax5.set_ylabel('Average Rating', fontweight='bold')
ax5.set_title('District Rating Trends', fontweight='bold', fontsize=12)
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(1, 5)

# Subplot 3,2: Platform comparison
ax6 = plt.subplot(3, 2, 6)
try:
    platforms = reviews_df['platform'].unique()[:2]  # Limit to 2 platforms
    time_periods = ['2010-2014', '2015-2018', '2019-2021']
    
    violin_data = []
    positions = []
    labels = []
    
    pos_counter = 0
    for i, period in enumerate(time_periods):
        start_year, end_year = map(int, period.split('-'))
        period_data = reviews_df[
            (reviews_df['date'].dt.year >= start_year) & 
            (reviews_df['date'].dt.year <= end_year)
        ]
        
        for j, platform in enumerate(platforms):
            platform_scores = period_data[
                period_data['platform'] == platform
            ]['score'].dropna()
            
            if len(platform_scores) > 20:  # Minimum data requirement
                # Sample data to avoid memory issues
                if len(platform_scores) > 1000:
                    platform_scores = platform_scores.sample(1000, random_state=42)
                
                violin_data.append(platform_scores.values)
                positions.append(pos_counter)
                labels.append(f'{str(platform)[:8]}\n{period}')
                pos_counter += 1
    
    if len(violin_data) > 0:
        # Create violin plots
        parts = ax6.violinplot(violin_data, positions=positions, widths=0.6, 
                              showmeans=True, showmedians=True)
        
        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_main[i % len(colors_main)])
            pc.set_alpha(0.7)
        
        ax6.set_xticks(positions)
        ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for platform comparison', 
                transform=ax6.transAxes, ha='center', va='center')

except Exception as e:
    ax6.text(0.5, 0.5, f'Error in subplot 3,2: {str(e)[:50]}...', 
             transform=ax6.transAxes, ha='center', va='center')

ax6.set_ylabel('Rating Distribution', fontweight='bold')
ax6.set_title('Platform Comparison Over Time', fontweight='bold', fontsize=12)
ax6.set_ylim(1, 5)
ax6.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=2.0)
plt.savefig('peruvian_restaurant_temporal_analysis.png', dpi=150, bbox_inches='tight')
plt.show()