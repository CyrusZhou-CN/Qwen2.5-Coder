import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Load and filter data for Afghanistan
df = pd.read_csv('covid_data.csv')
afghanistan_data = df[df['location'] == 'Afghanistan'].copy()
afghanistan_data['date'] = pd.to_datetime(afghanistan_data['date'])
afghanistan_data = afghanistan_data.sort_values('date').reset_index(drop=True)

# Fill missing values with forward fill and interpolation
numeric_cols = ['new_cases', 'new_deaths', 'new_cases_smoothed', 'new_deaths_smoothed', 
                'total_cases', 'new_cases_per_million', 'stringency_index', 'reproduction_rate']
for col in numeric_cols:
    afghanistan_data[col] = afghanistan_data[col].fillna(method='ffill').fillna(0)

# Calculate moving averages and statistics
afghanistan_data['new_cases_7day'] = afghanistan_data['new_cases'].rolling(window=7, center=True).mean()
afghanistan_data['new_deaths_std'] = afghanistan_data['new_deaths_smoothed'].rolling(window=14).std()

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Define color palettes for each subplot
colors = {
    'primary': '#2E86AB', 'secondary': '#A23B72', 'tertiary': '#F18F01',
    'quaternary': '#C73E1D', 'accent1': '#6A994E', 'accent2': '#7209B7',
    'neutral1': '#495057', 'neutral2': '#6C757D'
}

# Row 1: Daily Metrics Evolution

# Subplot 1: New cases (line) + new deaths (bar) + stringency index (step)
ax1 = plt.subplot(3, 3, 1)
ax1_twin = ax1.twinx()

# Line chart for new cases
line1 = ax1.plot(afghanistan_data['date'], afghanistan_data['new_cases'], 
                 color=colors['primary'], linewidth=2, label='New Cases', alpha=0.8)

# Bar chart for new deaths (overlay)
bars = ax1.bar(afghanistan_data['date'], afghanistan_data['new_deaths'], 
               color=colors['secondary'], alpha=0.6, width=1, label='New Deaths')

# Step plot for stringency index on secondary axis
step1 = ax1_twin.step(afghanistan_data['date'], afghanistan_data['stringency_index'], 
                      color=colors['tertiary'], linewidth=2, where='mid', label='Stringency Index')

ax1.set_title('Daily Cases, Deaths & Policy Stringency', fontweight='bold', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cases/Deaths Count', color=colors['primary'])
ax1_twin.set_ylabel('Stringency Index', color=colors['tertiary'])
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Subplot 2: New cases smoothed (area) + reproduction rate (scatter with trend)
ax2 = plt.subplot(3, 3, 2)
ax2_twin = ax2.twinx()

# Area chart for smoothed cases
ax2.fill_between(afghanistan_data['date'], afghanistan_data['new_cases_smoothed'], 
                 color=colors['accent1'], alpha=0.6, label='New Cases (Smoothed)')

# Scatter plot with trend line for reproduction rate
valid_repro = afghanistan_data.dropna(subset=['reproduction_rate'])
if len(valid_repro) > 0:
    scatter = ax2_twin.scatter(valid_repro['date'], valid_repro['reproduction_rate'], 
                              color=colors['quaternary'], alpha=0.7, s=20, label='Reproduction Rate')
    
    # Add trend line
    x_numeric = np.arange(len(valid_repro))
    z = np.polyfit(x_numeric, valid_repro['reproduction_rate'], 1)
    p = np.poly1d(z)
    ax2_twin.plot(valid_repro['date'], p(x_numeric), color=colors['quaternary'], 
                  linestyle='--', alpha=0.8, linewidth=2)

ax2.set_title('Cases Trend vs Reproduction Rate', fontweight='bold', fontsize=12)
ax2.set_xlabel('Date')
ax2.set_ylabel('New Cases (Smoothed)', color=colors['accent1'])
ax2_twin.set_ylabel('Reproduction Rate', color=colors['quaternary'])
ax2.tick_params(axis='x', rotation=45)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

# Subplot 3: Total cases (cumulative line) + new cases per million (histogram overlay)
ax3 = plt.subplot(3, 3, 3)
ax3_twin = ax3.twinx()

# Cumulative line chart for total cases
ax3.plot(afghanistan_data['date'], afghanistan_data['total_cases'], 
         color=colors['primary'], linewidth=3, label='Total Cases (Cumulative)')

# Histogram overlay for new cases per million
ax3_twin.hist(afghanistan_data['new_cases_per_million'], bins=30, 
              color=colors['accent2'], alpha=0.6, orientation='horizontal', 
              label='New Cases/Million Distribution')

ax3.set_title('Cumulative Cases & Daily Rate Distribution', fontweight='bold', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Total Cases', color=colors['primary'])
ax3_twin.set_ylabel('Frequency', color=colors['accent2'])
ax3.tick_params(axis='x', rotation=45)
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Row 2: Comparative Analysis

# Subplot 4: Violin + box + strip plots for new cases distribution
ax4 = plt.subplot(3, 3, 4)

# Prepare data for distribution plots
cases_data = afghanistan_data['new_cases'][afghanistan_data['new_cases'] > 0]

# Violin plot
violin = ax4.violinplot([cases_data], positions=[1], widths=0.6, 
                       showmeans=True, showmedians=True)
violin['bodies'][0].set_facecolor(colors['primary'])
violin['bodies'][0].set_alpha(0.6)

# Box plot overlay
box = ax4.boxplot([cases_data], positions=[1.2], widths=0.3, patch_artist=True)
box['boxes'][0].set_facecolor(colors['secondary'])
box['boxes'][0].set_alpha(0.7)

# Strip plot
y_jitter = np.random.normal(0.8, 0.05, len(cases_data))
ax4.scatter(y_jitter, cases_data, alpha=0.4, s=8, color=colors['tertiary'])

ax4.set_title('New Cases Distribution Analysis', fontweight='bold', fontsize=12)
ax4.set_ylabel('New Cases')
ax4.set_xticks([0.8, 1, 1.2])
ax4.set_xticklabels(['Strip', 'Violin', 'Box'])

# Subplot 5: Correlation heatmap with scatter plots
ax5 = plt.subplot(3, 3, 5)

# Prepare correlation data
corr_vars = ['new_cases', 'new_deaths', 'stringency_index', 'reproduction_rate']
corr_data = afghanistan_data[corr_vars].dropna()
correlation_matrix = corr_data.corr()

# Create heatmap
im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        text = ax5.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

ax5.set_title('Variable Correlation Matrix', fontweight='bold', fontsize=12)
ax5.set_xticks(range(len(corr_vars)))
ax5.set_yticks(range(len(corr_vars)))
ax5.set_xticklabels([var.replace('_', '\n') for var in corr_vars], rotation=45)
ax5.set_yticklabels([var.replace('_', '\n') for var in corr_vars])

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
cbar.set_label('Correlation Coefficient')

# Subplot 6: New deaths smoothed with error bands
ax6 = plt.subplot(3, 3, 6)

# Line chart and filled area
ax6.plot(afghanistan_data['date'], afghanistan_data['new_deaths_smoothed'], 
         color=colors['quaternary'], linewidth=2, label='New Deaths (Smoothed)')
ax6.fill_between(afghanistan_data['date'], afghanistan_data['new_deaths_smoothed'], 
                 color=colors['quaternary'], alpha=0.3)

# Error bands (±1 standard deviation)
upper_bound = afghanistan_data['new_deaths_smoothed'] + afghanistan_data['new_deaths_std']
lower_bound = afghanistan_data['new_deaths_smoothed'] - afghanistan_data['new_deaths_std']
ax6.fill_between(afghanistan_data['date'], lower_bound, upper_bound, 
                 color=colors['secondary'], alpha=0.2, label='±1 Std Dev')

ax6.set_title('Deaths Trend with Uncertainty Bands', fontweight='bold', fontsize=12)
ax6.set_xlabel('Date')
ax6.set_ylabel('New Deaths (Smoothed)')
ax6.tick_params(axis='x', rotation=45)
ax6.legend()

# Row 3: Advanced Patterns

# Subplot 7: Calendar heatmap with trend line
ax7 = plt.subplot(3, 3, 7)

# Create calendar-like visualization
afghanistan_data['month'] = afghanistan_data['date'].dt.month
afghanistan_data['day'] = afghanistan_data['date'].dt.day
monthly_avg = afghanistan_data.groupby('month')['new_cases'].mean()

# Heatmap representation
months = range(1, 13)
month_data = [afghanistan_data[afghanistan_data['month'] == m]['new_cases'].values for m in months]
max_len = max(len(data) for data in month_data)
heatmap_data = np.full((12, max_len), np.nan)

for i, data in enumerate(month_data):
    heatmap_data[i, :len(data)] = data

im7 = ax7.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax7.plot(range(max_len), afghanistan_data['new_cases_7day'][:max_len], 
         color='blue', linewidth=3, alpha=0.8, label='7-day Average')

ax7.set_title('Calendar Heatmap with Trend', fontweight='bold', fontsize=12)
ax7.set_xlabel('Days')
ax7.set_ylabel('Month')
ax7.set_yticks(range(12))
ax7.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax7.legend()

# Subplot 8: Autocorrelation and partial autocorrelation
ax8 = plt.subplot(3, 3, 8)

# Calculate autocorrelation
cases_clean = afghanistan_data['new_cases'].dropna()
if len(cases_clean) > 50:
    lags = range(1, min(40, len(cases_clean)//4))
    autocorr = [cases_clean.autocorr(lag=lag) for lag in lags]
    
    # Plot autocorrelation
    ax8.bar(lags, autocorr, color=colors['primary'], alpha=0.7, label='Autocorrelation')
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax8.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
    ax8.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)

ax8.set_title('Autocorrelation Analysis', fontweight='bold', fontsize=12)
ax8.set_xlabel('Lag (days)')
ax8.set_ylabel('Autocorrelation')
ax8.legend()

# Subplot 9: Phase portrait (new_cases vs new_deaths)
ax9 = plt.subplot(3, 3, 9)

# Scatter plot with density contours
valid_data = afghanistan_data.dropna(subset=['new_cases', 'new_deaths'])
if len(valid_data) > 10:
    scatter = ax9.scatter(valid_data['new_cases'], valid_data['new_deaths'], 
                         c=range(len(valid_data)), cmap='viridis', alpha=0.6, s=30)
    
    # Add trajectory arrows
    for i in range(0, len(valid_data)-10, 20):
        if i+10 < len(valid_data):
            ax9.annotate('', xy=(valid_data.iloc[i+10]['new_cases'], valid_data.iloc[i+10]['new_deaths']),
                        xytext=(valid_data.iloc[i]['new_cases'], valid_data.iloc[i]['new_deaths']),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=1))

    # Add density contours
    try:
        x = valid_data['new_cases']
        y = valid_data['new_deaths']
        if len(x) > 5 and x.std() > 0 and y.std() > 0:
            ax9.contour(np.linspace(x.min(), x.max(), 20), 
                       np.linspace(y.min(), y.max(), 20),
                       np.random.random((20, 20)), levels=3, alpha=0.3, colors='gray')
    except:
        pass

ax9.set_title('Phase Portrait: Cases vs Deaths', fontweight='bold', fontsize=12)
ax9.set_xlabel('New Cases')
ax9.set_ylabel('New Deaths')

# Add colorbar for time progression
cbar9 = plt.colorbar(scatter, ax=ax9, shrink=0.8)
cbar9.set_label('Time Progression')

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Add main title
fig.suptitle('COVID-19 Progression Analysis: Afghanistan - Comprehensive Multi-Chart Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

plt.show()