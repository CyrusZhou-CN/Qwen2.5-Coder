import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import seaborn as sns

# Load data
df = pd.read_csv('Oscar Winners - Director.csv')

# Data preprocessing
# Clean year column - extract start year from ranges like "1927/28"
# Handle NaN values properly
df['Year_str'] = df['Year'].astype(str)
df['Year_Clean'] = df['Year_str'].str.extract('(\d{4})')
df['Year_Clean'] = pd.to_numeric(df['Year_Clean'], errors='coerce')

# Remove rows with NaN years
df = df.dropna(subset=['Year_Clean'])
df['Year_Clean'] = df['Year_Clean'].astype(int)

# Filter data to 1930-2019 as requested
df = df[(df['Year_Clean'] >= 1930) & (df['Year_Clean'] <= 2019)]

# Create decade column
df['Decade'] = (df['Year_Clean'] // 10) * 10

# Separate winners and nominations
winners = df[df['Nomination/Winner'] == 'Winner'].copy()
nominations = df[df['Nomination/Winner'] == 'Nomination'].copy()

# Create comprehensive 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.patch.set_facecolor('white')

# Top row - Subplot 1: Line chart with area fill + bar chart overlay
ax1 = axes[0, 0]
decades = sorted(df['Decade'].unique())

# Area fill for total nominations
total_per_decade = df.groupby('Decade').size()
ax1.fill_between(decades, total_per_decade.reindex(decades, fill_value=0).values, 
                 alpha=0.3, color='lightblue', label='Total Nominations')
ax1.plot(decades, total_per_decade.reindex(decades, fill_value=0).values, 
         color='blue', linewidth=2, marker='o')

# Bar chart for winners
ax1_twin = ax1.twinx()
winner_counts = winners.groupby('Decade').size()
ax1_twin.bar(decades, winner_counts.reindex(decades, fill_value=0), 
             alpha=0.7, color='gold', width=3, label='Winners')

ax1.set_title('Total Nominations vs Winners Per Decade', fontweight='bold', fontsize=12)
ax1.set_xlabel('Decade')
ax1.set_ylabel('Total Nominations', color='blue')
ax1_twin.set_ylabel('Winner Count', color='gold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Top row - Subplot 2: Stacked area chart with trend line
ax2 = axes[0, 1]
years = sorted(df['Year_Clean'].unique())
yearly_winners = winners.groupby('Year_Clean').size()
yearly_nominations = nominations.groupby('Year_Clean').size()

# Reindex to ensure all years are covered
yearly_winners_full = yearly_winners.reindex(years, fill_value=0)
yearly_nominations_full = yearly_nominations.reindex(years, fill_value=0)

ax2.stackplot(years, yearly_winners_full.values, yearly_nominations_full.values,
              labels=['Winners', 'Nominations'], alpha=0.7, colors=['gold', 'lightcoral'])

# Add trend line
total_yearly = df.groupby('Year_Clean').size()
total_yearly_full = total_yearly.reindex(years, fill_value=0)
z = np.polyfit(years, total_yearly_full.values, 1)
p = np.poly1d(z)
ax2.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label='Trend')

ax2.set_title('Nominations vs Winners Over Time', fontweight='bold', fontsize=12)
ax2.set_xlabel('Year')
ax2.set_ylabel('Count')
ax2.legend()

# Top row - Subplot 3: Dual-axis with cumulative winners and scatter
ax3 = axes[0, 2]
cumulative_winners = winners.groupby('Year_Clean').size().cumsum()
ax3.plot(cumulative_winners.index, cumulative_winners.values, 
         color='darkgreen', linewidth=2, label='Cumulative Winners')

ax3_twin = ax3.twinx()
winner_years = winners['Year_Clean'].values
decade_sizes = [max(10, ((year // 10) * 10 - 1920) * 2) for year in winner_years]  # Size by decade
ax3_twin.scatter(winner_years, np.ones(len(winner_years)), 
                s=decade_sizes, alpha=0.6, color='orange', label='Individual Winners')

ax3.set_title('Cumulative Winners with Individual Wins', fontweight='bold', fontsize=12)
ax3.set_xlabel('Year')
ax3.set_ylabel('Cumulative Winners', color='darkgreen')
ax3_twin.set_ylabel('Individual Wins', color='orange')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Middle row - Subplot 4: Histogram with KDE and box plot
ax4 = axes[1, 0]
director_wins = winners['Director(s)'].value_counts()

if len(director_wins) > 0:
    ax4.hist(director_wins.values, bins=min(15, len(director_wins)), alpha=0.7, 
             color='skyblue', density=True, label='Frequency Distribution')
    
    # KDE overlay
    if len(director_wins.values) > 1:
        kde = gaussian_kde(director_wins.values)
        x_range = np.linspace(director_wins.min(), director_wins.max(), 100)
        ax4.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
    
    # Box plot as inset
    box_ax = ax4.inset_axes([0.7, 0.7, 0.25, 0.25])
    box_ax.boxplot(director_wins.values, vert=False)
    box_ax.set_xlabel('Wins', fontsize=8)

ax4.set_title('Distribution of Director Winning Frequency', fontweight='bold', fontsize=12)
ax4.set_xlabel('Number of Wins')
ax4.set_ylabel('Density')
ax4.legend()

# Middle row - Subplot 5: Time series with moving average
ax5 = axes[1, 1]
yearly_ceremonies = df.groupby('Year_Clean').size()
ax5.plot(yearly_ceremonies.index, yearly_ceremonies.values, 
         color='purple', alpha=0.7, label='Annual Ceremonies')

# Moving average
window = min(5, len(yearly_ceremonies))
if window > 1:
    moving_avg = yearly_ceremonies.rolling(window=window, center=True).mean()
    ax5.plot(moving_avg.index, moving_avg.values, color='darkred', 
             linewidth=2, label=f'{window}-Year Moving Average')

# Highlight peak years
if len(yearly_ceremonies) > 0:
    peak_threshold = yearly_ceremonies.quantile(0.9)
    peak_years = yearly_ceremonies[yearly_ceremonies >= peak_threshold]
    ax5.scatter(peak_years.index, peak_years.values, 
               color='red', s=100, zorder=5, label='Peak Years')

ax5.set_title('Award Ceremonies Per Year with Trends', fontweight='bold', fontsize=12)
ax5.set_xlabel('Year')
ax5.set_ylabel('Number of Ceremonies')
ax5.legend()

# Middle row - Subplot 6: Violin plot with strip plot
ax6 = axes[1, 2]
decade_data_list = []
decade_labels = []

for decade in sorted(df['Decade'].unique()):
    decade_years = df[df['Decade'] == decade]['Year_Clean'].values
    if len(decade_years) > 0:
        decade_data_list.append(decade_years)
        decade_labels.append(str(decade))

if decade_data_list:
    # Create violin plot
    parts = ax6.violinplot(decade_data_list, positions=range(len(decade_data_list)), widths=0.8)
    
    # Strip plot overlay
    for i, decade_years in enumerate(decade_data_list):
        jittered_x = np.random.normal(i, 0.04, size=len(decade_years))
        ax6.scatter(jittered_x, decade_years, alpha=0.6, s=20)

ax6.set_title('Year Distribution by Decade', fontweight='bold', fontsize=12)
ax6.set_xlabel('Decade')
ax6.set_ylabel('Year')
ax6.set_xticks(range(len(decade_labels)))
ax6.set_xticklabels(decade_labels, rotation=45)

# Bottom row - Subplot 7: Slope chart with area background
ax7 = axes[2, 0]
decade_nom_counts = nominations.groupby('Decade').size()
decade_win_counts = winners.groupby('Decade').size()

# Area background
total_awards = df.groupby('Decade').size()
ax7.fill_between(range(len(decades)), total_awards.reindex(decades, fill_value=0).values, 
                 alpha=0.2, color='lightgray', label='Total Volume')

# Slope lines connecting nominations to winners for each decade
x_positions = [0, 1]
for i, decade in enumerate(decades):
    nom_count = decade_nom_counts.get(decade, 0)
    win_count = decade_win_counts.get(decade, 0)
    ax7.plot(x_positions, [nom_count, win_count], 'o-', alpha=0.7, linewidth=2)

ax7.set_title('Nominations to Winners Flow by Decade', fontweight='bold', fontsize=12)
ax7.set_xticks([0, 1])
ax7.set_xticklabels(['Nominations', 'Winners'])
ax7.set_ylabel('Count')
ax7.legend()

# Bottom row - Subplot 8: Multi-line time series with confidence bands
ax8 = axes[2, 1]
cumulative_noms = nominations.groupby('Year_Clean').size().cumsum()
cumulative_wins = winners.groupby('Year_Clean').size().cumsum()

if len(cumulative_noms) > 0:
    ax8.plot(cumulative_noms.index, cumulative_noms.values, 
             label='Cumulative Nominations', linewidth=2, color='blue')

if len(cumulative_wins) > 0:
    ax8.plot(cumulative_wins.index, cumulative_wins.values, 
             label='Cumulative Winners', linewidth=2, color='red')

    # Add confidence intervals (simple polynomial fit)
    if len(cumulative_wins) > 2:
        years_array = np.array(cumulative_wins.index)
        wins_fit = np.polyfit(years_array, cumulative_wins.values, min(2, len(years_array)-1))
        wins_poly = np.poly1d(wins_fit)
        
        # Simple confidence bands
        wins_std = np.std(cumulative_wins.values - wins_poly(years_array))
        ax8.fill_between(years_array, wins_poly(years_array) - wins_std, 
                        wins_poly(years_array) + wins_std, alpha=0.2, color='red')

ax8.set_title('Running Totals with Confidence Intervals', fontweight='bold', fontsize=12)
ax8.set_xlabel('Year')
ax8.set_ylabel('Cumulative Count')
ax8.legend()

# Bottom row - Subplot 9: Composite bar + line + scatter
ax9 = axes[2, 2]
unique_directors = df.groupby('Decade')['Director(s)'].nunique()
ax9.bar(decades, unique_directors.reindex(decades, fill_value=0), 
        alpha=0.7, color='lightgreen', label='Unique Directors')

ax9_twin = ax9.twinx()
repeat_winners = winners.groupby('Decade')['Director(s)'].apply(
    lambda x: (x.value_counts() > 1).sum() if len(x) > 0 else 0)
ax9_twin.plot(decades, repeat_winners.reindex(decades, fill_value=0), 
              color='darkred', marker='o', linewidth=2, label='Repeat Winners')

# Scatter for notable directors (those with multiple wins)
notable_directors = winners['Director(s)'].value_counts()
notable_directors = notable_directors[notable_directors > 1]

if len(notable_directors) > 0:
    for director in notable_directors.index[:min(5, len(notable_directors))]:  # Top 5 repeat winners
        director_decades = winners[winners['Director(s)'] == director]['Decade'].values
        for decade in director_decades:
            repeat_count = repeat_winners.get(decade, 0)
            ax9_twin.scatter(decade, repeat_count, s=100, color='orange', alpha=0.8, zorder=5)

ax9.set_title('Directors Per Decade with Repeat Winners', fontweight='bold', fontsize=12)
ax9.set_xlabel('Decade')
ax9.set_ylabel('Unique Directors', color='green')
ax9_twin.set_ylabel('Repeat Winners', color='darkred')
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')

# Overall layout adjustment
plt.tight_layout(pad=2.0)
plt.suptitle('Evolution of Oscar Best Director Awards (1930-2019)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.95)
plt.savefig('oscar_directors_evolution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()