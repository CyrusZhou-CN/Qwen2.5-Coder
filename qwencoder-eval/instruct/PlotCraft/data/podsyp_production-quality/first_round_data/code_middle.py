import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

# Load data
df_X = pd.read_csv('data_X.csv')
df_Y = pd.read_csv('data_Y.csv')

# Convert date_time columns to datetime
df_X['date_time'] = pd.to_datetime(df_X['date_time'])
df_Y['date_time'] = pd.to_datetime(df_Y['date_time'])

# Merge datasets on date_time
# Since X data has minute-level data and Y has hourly data, we need to aggregate X data
df_X_hourly = df_X.set_index('date_time').resample('H').mean().reset_index()
merged_df = pd.merge(df_X_hourly, df_Y, on='date_time', how='inner')

# Select temperature columns and quality
temp_columns = [col for col in merged_df.columns if col.startswith('T_data_')]
correlation_data = merged_df[temp_columns + ['quality']]

# Remove any rows with NaN values
correlation_data = correlation_data.dropna()

# Calculate correlation matrix
corr_matrix = correlation_data.corr()

# Extract correlations with quality (excluding quality-quality correlation)
quality_correlations = corr_matrix['quality'][:-1]

# Remove any NaN correlations and find the strongest correlation with quality
quality_correlations_clean = quality_correlations.dropna()
strongest_corr_sensor = quality_correlations_clean.abs().idxmax()
strongest_corr_value = quality_correlations_clean[strongest_corr_sensor]

print(f"Strongest correlation: {strongest_corr_sensor} with correlation {strongest_corr_value:.3f}")

# Create composite visualization
fig = plt.figure(figsize=(16, 12))

# Create main subplot for heatmap
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)

# 1. Create correlation heatmap as base layer
# Use only temperature sensor correlations (exclude quality row/column for cleaner heatmap)
temp_corr_matrix = corr_matrix.iloc[:-1, :-1]

# Create heatmap with cool color scheme
heatmap = sns.heatmap(temp_corr_matrix, 
                     annot=True, 
                     fmt='.2f', 
                     cmap='coolwarm', 
                     center=0,
                     square=True,
                     cbar_kws={'label': 'Temperature Sensor Correlations'},
                     ax=ax1)

# Customize heatmap
ax1.set_title('Temperature Sensor Correlation Matrix', 
             fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Temperature Sensors', fontsize=12)
ax1.set_ylabel('Temperature Sensors', fontsize=12)

# Rotate labels for better readability
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)

# 2. Create scatter plot in separate subplot
ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

# Prepare data for scatter plot
x_data = merged_df[strongest_corr_sensor].values
y_data = merged_df['quality'].values

# Remove any NaN values
mask = ~(np.isnan(x_data) | np.isnan(y_data))
x_clean = x_data[mask]
y_clean = y_data[mask]

# Create scatter plot with different color scheme (warm colors)
scatter = ax2.scatter(x_clean, y_clean, 
                     alpha=0.6, 
                     c='orange', 
                     s=15, 
                     edgecolors='darkorange', 
                     linewidth=0.3)

# Calculate and plot best-fit line
slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
line_x = np.linspace(x_clean.min(), x_clean.max(), 100)
line_y = slope * line_x + intercept

# Plot best-fit line
ax2.plot(line_x, line_y, color='red', linewidth=2, label=f'Best fit (r={r_value:.3f})')

# Calculate and plot confidence intervals
def predict_interval(x, y, new_x, confidence=0.95):
    n = len(x)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - np.mean(y)))
    syy = np.sum((y - np.mean(y)) ** 2)
    
    s_yx = np.sqrt((syy - sxy**2/sxx) / (n - 2))
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
    
    pred_y = slope * new_x + intercept
    se = s_yx * np.sqrt(1/n + (new_x - x_mean)**2/sxx)
    margin = t_val * se
    
    return pred_y - margin, pred_y + margin

# Calculate confidence intervals
lower_ci, upper_ci = predict_interval(x_clean, y_clean, line_x)

# Plot confidence intervals
ax2.fill_between(line_x, lower_ci, upper_ci, 
                alpha=0.2, color='red', 
                label='95% Confidence Interval')

# Customize scatter plot
ax2.set_xlabel(f'{strongest_corr_sensor} (Temperature)', fontsize=11)
ax2.set_ylabel('Quality', fontsize=11)
ax2.set_title(f'Strongest Temperature-Quality Correlation\n{strongest_corr_sensor}', 
             fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Add correlation statistics as text box
textstr = f'Correlation: {strongest_corr_value:.3f}\nP-value: {p_value:.2e}\nSample size: {len(x_clean)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# 3. Create bar chart showing all temperature-quality correlations
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

# Sort correlations by absolute value for better visualization
sorted_correlations = quality_correlations_clean.reindex(
    quality_correlations_clean.abs().sort_values(ascending=True).index
)

# Create horizontal bar chart
colors = ['red' if sensor == strongest_corr_sensor else 'skyblue' 
          for sensor in sorted_correlations.index]
bars = ax3.barh(range(len(sorted_correlations)), sorted_correlations.values, color=colors)

# Customize bar chart
ax3.set_yticks(range(len(sorted_correlations)))
ax3.set_yticklabels(sorted_correlations.index)
ax3.set_xlabel('Correlation with Quality', fontsize=11)
ax3.set_title('Temperature Sensor Correlations with Product Quality', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
ax3.axvline(x=0, color='black', linewidth=0.8)

# Add correlation values as text on bars
for i, (sensor, corr_val) in enumerate(sorted_correlations.items()):
    ax3.text(corr_val + (0.01 if corr_val >= 0 else -0.01), i, f'{corr_val:.3f}', 
             va='center', ha='left' if corr_val >= 0 else 'right', fontsize=8)

# Add overall title
fig.suptitle('Roasting Machine Temperature Patterns vs Product Quality Analysis', 
             fontsize=16, fontweight='bold', y=0.95)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('temperature_quality_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()