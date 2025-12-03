import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Income_Urban_VS_Rural.csv')

# Data preprocessing
df = df.dropna()
overall_median_income = df['Median Household Income'].median()

# Calculate income deviation from state average
state_avg_income = df.groupby('State')['Median Household Income'].mean()
df['Income_Deviation_State'] = df.apply(lambda row: row['Median Household Income'] - state_avg_income[row['State']], axis=1)

# Sample data for performance (use every 5th row for complex visualizations)
df_sample = df.iloc[::5].copy()

# Create the comprehensive 3x3 subplot grid
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('white')

# Color palette for urban/rural
colors = {'Urban': '#2E86AB', 'Rural': '#A23B72'}

# Subplot 1: Histogram with KDE overlay
ax1 = plt.subplot(3, 3, 1)
for ur_type in ['Urban', 'Rural']:
    data = df[df['Urban-Rural'] == ur_type]['Total Population']
    # Limit data range for better visualization
    data_limited = data[data <= 200000]
    
    ax1.hist(data_limited, bins=30, alpha=0.6, density=True, color=colors[ur_type], label=f'{ur_type} Histogram')
    
    # Simplified KDE
    if len(data_limited) > 10:
        kde = stats.gaussian_kde(data_limited)
        kde_x = np.linspace(data_limited.min(), data_limited.max(), 50)
        ax1.plot(kde_x, kde(kde_x), color=colors[ur_type], linewidth=2, linestyle='--', label=f'{ur_type} KDE')

ax1.set_xlabel('Total Population')
ax1.set_ylabel('Density')
ax1.set_title('Population Distribution: Histogram + KDE', fontweight='bold', fontsize=10)
ax1.legend(fontsize=8)

# Subplot 2: Scatter plot with regression lines
ax2 = plt.subplot(3, 3, 2)
for ur_type in ['Urban', 'Rural']:
    data = df_sample[df_sample['Urban-Rural'] == ur_type]
    if len(data) > 0:
        ax2.scatter(data['Total Population'], data['Median Household Income'], 
                   alpha=0.6, color=colors[ur_type], label=ur_type, s=15)
        
        # Simple best-fit line
        if len(data) > 2:
            z = np.polyfit(data['Total Population'], data['Median Household Income'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['Total Population'].min(), data['Total Population'].max(), 20)
            ax2.plot(x_line, p(x_line), color=colors[ur_type], linewidth=2)

ax2.set_xlabel('Total Population')
ax2.set_ylabel('Median Household Income ($)')
ax2.set_title('Population vs Income with Regression', fontweight='bold', fontsize=10)
ax2.legend(fontsize=8)

# Subplot 3: Box plot (simplified from violin plot)
ax3 = plt.subplot(3, 3, 3)
urban_pop = df[df['Urban-Rural'] == 'Urban']['Total Population']
rural_pop = df[df['Urban-Rural'] == 'Rural']['Total Population']

box_data = [urban_pop[urban_pop <= 200000], rural_pop[rural_pop <= 200000]]
bp = ax3.boxplot(box_data, labels=['Urban', 'Rural'], patch_artist=True)

for patch, ur_type in zip(bp['boxes'], ['Urban', 'Rural']):
    patch.set_facecolor(colors[ur_type])
    patch.set_alpha(0.7)

ax3.set_ylabel('Total Population')
ax3.set_title('Population Distribution: Box Plot', fontweight='bold', fontsize=10)

# Subplot 4: Diverging bar chart (simplified)
ax4 = plt.subplot(3, 3, 4)
df['Income_Deviation'] = df['Median Household Income'] - overall_median_income
df_sorted = df.sort_values('Income_Deviation')

# Sample every 20th county for readability
sample_df = df_sorted.iloc[::20].copy()
y_pos = np.arange(len(sample_df))

bars = ax4.barh(y_pos, sample_df['Income_Deviation'], 
                color=[colors[ur] for ur in sample_df['Urban-Rural']], alpha=0.8)

ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Income Deviation from Median ($)')
ax4.set_title('County Income Deviations', fontweight='bold', fontsize=10)
ax4.set_yticks([])

# Subplot 5: Simplified dumbbell plot
ax5 = plt.subplot(3, 3, 5)
top_states = df['State'].value_counts().head(10).index
state_means = df[df['State'].isin(top_states)].groupby(['State', 'Urban-Rural'])['Median Household Income'].mean().unstack()
state_means = state_means.dropna()

y_positions = np.arange(len(state_means))
for i, state in enumerate(state_means.index):
    if 'Rural' in state_means.columns and 'Urban' in state_means.columns:
        rural_mean = state_means.loc[state, 'Rural']
        urban_mean = state_means.loc[state, 'Urban']
        
        ax5.plot([rural_mean, urban_mean], [i, i], 'k-', alpha=0.6, linewidth=1)
        ax5.scatter(urban_mean, i, color=colors['Urban'], s=50, alpha=0.8)
        ax5.scatter(rural_mean, i, color=colors['Rural'], s=50, alpha=0.8)

ax5.set_yticks(y_positions)
ax5.set_yticklabels(state_means.index, fontsize=8)
ax5.set_xlabel('Median Household Income ($)')
ax5.set_title('Urban vs Rural Income by State', fontweight='bold', fontsize=10)

# Subplot 6: Slope chart
ax6 = plt.subplot(3, 3, 6)
slope_states = state_means.head(8)

for i, state in enumerate(slope_states.index):
    if 'Rural' in slope_states.columns and 'Urban' in slope_states.columns:
        rural_income = slope_states.loc[state, 'Rural']
        urban_income = slope_states.loc[state, 'Urban']
        gap = urban_income - rural_income
        
        color = colors['Urban'] if gap > 0 else colors['Rural']
        ax6.plot([0, 1], [rural_income, urban_income], color=color, alpha=0.7, linewidth=2)

ax6.set_xlim(-0.1, 1.1)
ax6.set_xticks([0, 1])
ax6.set_xticklabels(['Rural', 'Urban'])
ax6.set_ylabel('Median Household Income ($)')
ax6.set_title('Income Slopes by State', fontweight='bold', fontsize=10)

# Subplot 7: Simple radar chart
ax7 = plt.subplot(3, 3, 7, projection='polar')

# Calculate percentiles
urban_data = df[df['Urban-Rural'] == 'Urban']
rural_data = df[df['Urban-Rural'] == 'Rural']

urban_pop_pct = np.percentile(urban_data['Total Population'], 50)
rural_pop_pct = np.percentile(rural_data['Total Population'], 50)
urban_inc_pct = np.percentile(urban_data['Median Household Income'], 50)
rural_inc_pct = np.percentile(rural_data['Median Household Income'], 50)

# Normalize to 0-100 scale
max_pop = df['Total Population'].max()
max_inc = df['Median Household Income'].max()

urban_values = [urban_pop_pct/max_pop*100, urban_inc_pct/max_inc*100]
rural_values = [rural_pop_pct/max_pop*100, rural_inc_pct/max_inc*100]

angles = np.linspace(0, 2 * np.pi, 2, endpoint=False).tolist()
angles += angles[:1]

urban_values += [urban_values[0]]
rural_values += [rural_values[0]]

ax7.plot(angles, urban_values, 'o-', linewidth=2, label='Urban', color=colors['Urban'])
ax7.fill(angles, urban_values, alpha=0.25, color=colors['Urban'])
ax7.plot(angles, rural_values, 'o-', linewidth=2, label='Rural', color=colors['Rural'])
ax7.fill(angles, rural_values, alpha=0.25, color=colors['Rural'])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(['Population', 'Income'], fontsize=8)
ax7.set_title('Metrics Comparison', fontweight='bold', fontsize=10, pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)

# Subplot 8: Correlation heatmap
ax8 = plt.subplot(3, 3, 8)
numeric_cols = ['Total Population', 'Median Household Income']
corr_matrix = df[numeric_cols].corr()

im = ax8.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation coefficients
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=8)

ax8.set_xticks(range(len(corr_matrix)))
ax8.set_yticks(range(len(corr_matrix)))
ax8.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
ax8.set_yticklabels(corr_matrix.columns, fontsize=8)
ax8.set_title('Correlation Heatmap', fontweight='bold', fontsize=10)

# Subplot 9: Bubble plot (simplified)
ax9 = plt.subplot(3, 3, 9)
df_bubble = df_sample.copy()
df_bubble['Abs_Deviation'] = np.abs(df_bubble['Income_Deviation_State'])

for ur_type in ['Urban', 'Rural']:
    data = df_bubble[df_bubble['Urban-Rural'] == ur_type]
    if len(data) > 0:
        # Limit population for better visualization
        data_limited = data[data['Total Population'] <= 300000]
        if len(data_limited) > 0:
            scatter = ax9.scatter(data_limited['Total Population'], 
                                data_limited['Income_Deviation_State'], 
                                s=data_limited['Abs_Deviation']/1000 + 10, 
                                alpha=0.6, color=colors[ur_type], label=ur_type)

ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax9.set_xlabel('Total Population')
ax9.set_ylabel('Income Deviation from State Avg ($)')
ax9.set_title('Population vs Income Deviation', fontweight='bold', fontsize=10)
ax9.legend(fontsize=8)

# Overall layout adjustment
plt.tight_layout(pad=1.5)
plt.savefig('income_deviation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()