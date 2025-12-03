import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('VSRR_Provisional_Drug_Overdose_Death_Counts.csv')

# Convert Data Value and Predicted Value to numeric, handling non-numeric values
df['Data Value'] = pd.to_numeric(df['Data Value'], errors='coerce')
df['Predicted Value'] = pd.to_numeric(df['Predicted Value'], errors='coerce')

# Create date column for time series analysis
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B', errors='coerce')

# Filter for total deaths indicator for main analysis
total_deaths = df[df['Indicator'] == 'Number of Deaths'].copy()

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Row 1, Subplot 1: Monthly trend with error bands and data completeness
ax1 = plt.subplot(3, 3, 1)
monthly_data = total_deaths.groupby('Date').agg({
    'Data Value': ['mean', 'std', 'count'],
    'Percent Complete': 'mean'
}).reset_index()
monthly_data.columns = ['Date', 'mean_deaths', 'std_deaths', 'count', 'completeness']
monthly_data = monthly_data.dropna()

if len(monthly_data) > 0:
    # Line chart with confidence intervals
    ax1.plot(monthly_data['Date'], monthly_data['mean_deaths'], color='#2E86AB', linewidth=2, label='Mean Deaths')
    ax1.fill_between(monthly_data['Date'], 
                    monthly_data['mean_deaths'] - monthly_data['std_deaths'].fillna(0),
                    monthly_data['mean_deaths'] + monthly_data['std_deaths'].fillna(0),
                    alpha=0.3, color='#2E86AB', label='±1 Std Dev')

    # Secondary axis for completeness bars
    ax1_twin = ax1.twinx()
    ax1_twin.bar(monthly_data['Date'], monthly_data['completeness'], 
                alpha=0.4, color='#F18F01', width=20, label='Data Completeness %')
    ax1_twin.set_ylabel('Data Completeness (%)', fontweight='bold', color='#F18F01')
    ax1_twin.set_ylim(0, 100)
    ax1_twin.legend(loc='upper right')
else:
    ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)

ax1.set_title('Monthly Overdose Deaths with Data Completeness', fontweight='bold', fontsize=12)
ax1.set_ylabel('Number of Deaths', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Row 1, Subplot 2: Seasonal decomposition
ax2 = plt.subplot(3, 3, 2)
# Simplified seasonal analysis
monthly_avg = total_deaths.dropna(subset=['Data Value']).groupby(['Year', 'Month'])['Data Value'].mean().reset_index()
if len(monthly_avg) > 0:
    monthly_avg['Month_num'] = pd.to_datetime(monthly_avg['Month'], format='%B').dt.month
    seasonal_pattern = monthly_avg.groupby('Month_num')['Data Value'].mean()
    trend_data = monthly_avg.groupby('Year')['Data Value'].mean()

    # Plot components
    months = range(1, 13)
    seasonal_values = [seasonal_pattern.get(i, 0) for i in months]
    ax2.plot(months, seasonal_values, 'o-', color='#C73E1D', linewidth=2, label='Seasonal Pattern')
    ax2.fill_between(months, seasonal_values, alpha=0.3, color='#C73E1D')

    if len(trend_data) > 0:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(trend_data.index, trend_data.values, 's-', color='#A23B72', linewidth=2, label='Yearly Trend')
        ax2_twin.set_ylabel('Yearly Average Deaths', fontweight='bold', color='#A23B72')
        ax2_twin.legend(loc='upper right')
else:
    ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)

ax2.set_title('Seasonal Decomposition Analysis', fontweight='bold', fontsize=12)
ax2.set_xlabel('Month', fontweight='bold')
ax2.set_ylabel('Seasonal Component', fontweight='bold', color='#C73E1D')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Row 1, Subplot 3: Predicted vs Actual with uncertainty
ax3 = plt.subplot(3, 3, 3)
pred_data = total_deaths.dropna(subset=['Data Value', 'Predicted Value'])
if len(pred_data) > 0:
    # Line charts for actual vs predicted
    ax3.plot(pred_data['Date'], pred_data['Data Value'], 'o-', color='#2E86AB', 
            linewidth=2, markersize=4, label='Actual Deaths')
    ax3.plot(pred_data['Date'], pred_data['Predicted Value'], 's-', color='#F18F01', 
            linewidth=2, markersize=4, label='Predicted Deaths')
    
    # Uncertainty band
    uncertainty = np.abs(pred_data['Data Value'] - pred_data['Predicted Value'])
    ax3.fill_between(pred_data['Date'], 
                    pred_data['Predicted Value'] - uncertainty,
                    pred_data['Predicted Value'] + uncertainty,
                    alpha=0.2, color='#F18F01', label='Prediction Uncertainty')
else:
    ax3.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=ax3.transAxes)

ax3.set_title('Predicted vs Actual Deaths with Uncertainty', fontweight='bold', fontsize=12)
ax3.set_ylabel('Number of Deaths', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Row 2, Subplot 4: Stacked area chart for drug types
ax4 = plt.subplot(3, 3, 4)
drug_indicators = ['Heroin (T40.1)', 'Opioids (T40.0-T40.4,T40.6)', 
                  'Psychostimulants with abuse potential (T43.6)',
                  'Synthetic opioids, excl. methadone (T40.4)']
drug_data = df[df['Indicator'].isin(drug_indicators)].copy()
drug_data = drug_data.dropna(subset=['Data Value', 'Date'])

if len(drug_data) > 0:
    drug_pivot = drug_data.groupby(['Date', 'Indicator'])['Data Value'].sum().unstack(fill_value=0)
    
    if not drug_pivot.empty and len(drug_pivot) > 0:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax4.stackplot(drug_pivot.index, drug_pivot.T, labels=drug_pivot.columns, 
                     colors=colors[:len(drug_pivot.columns)], alpha=0.7)
        
        # Overlay individual lines
        for i, col in enumerate(drug_pivot.columns):
            if i < len(colors):
                ax4.plot(drug_pivot.index, drug_pivot[col], color=colors[i], linewidth=1.5)
    else:
        ax4.text(0.5, 0.5, 'No drug-specific data available', ha='center', va='center', transform=ax4.transAxes)
else:
    ax4.text(0.5, 0.5, 'No drug-specific data available', ha='center', va='center', transform=ax4.transAxes)

ax4.set_title('Drug-Specific Death Composition Over Time', fontweight='bold', fontsize=12)
ax4.set_ylabel('Number of Deaths', fontweight='bold')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

# Row 2, Subplot 5: Correlation heatmap with marginals
ax5 = plt.subplot(3, 3, 5)
# Create correlation matrix for drug types by year
if len(drug_data) > 0:
    drug_corr_data = drug_data.groupby(['Year', 'Indicator'])['Data Value'].sum().unstack(fill_value=0)
    if not drug_corr_data.empty and len(drug_corr_data) > 1:
        corr_matrix = drug_corr_data.corr()
        im = ax5.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
        
        ax5.set_xticks(range(len(corr_matrix.columns)))
        ax5.set_yticks(range(len(corr_matrix.columns)))
        ax5.set_xticklabels([col.split('(')[0].strip() for col in corr_matrix.columns], 
                           rotation=45, ha='right', fontsize=8)
        ax5.set_yticklabels([col.split('(')[0].strip() for col in corr_matrix.columns], 
                           fontsize=8)
        plt.colorbar(im, ax=ax5, shrink=0.8)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center', transform=ax5.transAxes)
else:
    ax5.text(0.5, 0.5, 'No drug data available', ha='center', va='center', transform=ax5.transAxes)

ax5.set_title('Drug Type Correlation Matrix', fontweight='bold', fontsize=12)

# Row 2, Subplot 6: Radar chart simulation with polar bar chart
ax6 = plt.subplot(3, 3, 6, projection='polar')
# Create radar-like visualization
if len(drug_data) > 0:
    drug_means = drug_data.groupby('Indicator')['Data Value'].mean()
    if len(drug_means) > 0:
        angles = np.linspace(0, 2*np.pi, len(drug_means), endpoint=False)
        values = drug_means.values
        
        # Polar bar chart
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax6.bar(angles, values, width=0.4, alpha=0.7, 
                      color=colors[:len(values)])
        
        # Add lines connecting points
        angles_closed = np.concatenate([angles, [angles[0]]])
        values_closed = np.concatenate([values, [values[0]]])
        ax6.plot(angles_closed, values_closed, 'o-', linewidth=2, color='black', alpha=0.8)
    else:
        ax6.text(0, 0, 'No data', ha='center', va='center')
else:
    ax6.text(0, 0, 'No data', ha='center', va='center')

ax6.set_title('Drug Pattern Radar Analysis', fontweight='bold', fontsize=12, pad=20)
ax6.set_theta_zero_location('N')

# Row 3, Subplot 7: Scatter plot with regression and violin plot
ax7 = plt.subplot(3, 3, 7)
complete_data = total_deaths.dropna(subset=['Data Value', 'Percent Complete'])
if len(complete_data) > 0:
    # Scatter plot with regression
    ax7.scatter(complete_data['Percent Complete'], complete_data['Data Value'], 
               alpha=0.6, color='#2E86AB', s=30)
    
    # Add regression line
    if len(complete_data) > 1:
        try:
            z = np.polyfit(complete_data['Percent Complete'], complete_data['Data Value'], 1)
            p = np.poly1d(z)
            ax7.plot(complete_data['Percent Complete'], p(complete_data['Percent Complete']), 
                    "r--", alpha=0.8, linewidth=2)
        except:
            pass

    # Add violin plot on the side - simplified approach
    ax7_twin = ax7.twinx()
    if len(complete_data) > 5:  # Only if we have enough data
        try:
            completeness_bins = pd.cut(complete_data['Percent Complete'], bins=3)
            violin_data = []
            for bin_val in completeness_bins.cat.categories:
                bin_data = complete_data[completeness_bins == bin_val]['Data Value'].values
                if len(bin_data) > 0:
                    violin_data.append(bin_data)
            
            if violin_data and len(violin_data) > 0:
                parts = ax7_twin.violinplot(violin_data, positions=range(len(violin_data)), 
                                           widths=0.8, alpha=0.3)
                for pc in parts['bodies']:
                    pc.set_facecolor('#F18F01')
        except:
            pass
else:
    ax7.text(0.5, 0.5, 'No complete data available', ha='center', va='center', transform=ax7.transAxes)

ax7.set_title('Data Completeness vs Death Counts', fontweight='bold', fontsize=12)
ax7.set_xlabel('Percent Complete', fontweight='bold')
ax7.set_ylabel('Number of Deaths', fontweight='bold')
ax7.grid(True, alpha=0.3)

# Row 3, Subplot 8: Box plots with strip plots by investigation status
ax8 = plt.subplot(3, 3, 8)
investigation_data = total_deaths.dropna(subset=['Data Value', 'Percent Pending Investigation'])
if len(investigation_data) > 0:
    # Create investigation status categories
    try:
        investigation_data['Investigation_Category'] = pd.cut(
            investigation_data['Percent Pending Investigation'], 
            bins=[0, 5, 15, 100], labels=['Low', 'Medium', 'High']
        )
        
        # Box plots
        categories = investigation_data['Investigation_Category'].cat.categories
        box_data = []
        for cat in categories:
            cat_data = investigation_data[investigation_data['Investigation_Category'] == cat]['Data Value'].values
            if len(cat_data) > 0:
                box_data.append(cat_data)
        
        if box_data and len(box_data) > 0:
            bp = ax8.boxplot(box_data, labels=categories[:len(box_data)], patch_artist=True)
            colors = ['#2E86AB', '#F18F01', '#C73E1D']
            for i, patch in enumerate(bp['boxes']):
                if i < len(colors):
                    patch.set_facecolor(colors[i])
                    patch.set_alpha(0.7)
            
            # Add jittered strip plots
            for i, data in enumerate(box_data):
                if len(data) > 0:
                    y = data
                    x = np.random.normal(i+1, 0.04, size=len(y))
                    ax8.scatter(x, y, alpha=0.4, s=20, color='black')
        else:
            ax8.text(0.5, 0.5, 'No categorized data available', ha='center', va='center', transform=ax8.transAxes)
    except:
        ax8.text(0.5, 0.5, 'Error processing investigation data', ha='center', va='center', transform=ax8.transAxes)
else:
    ax8.text(0.5, 0.5, 'No investigation data available', ha='center', va='center', transform=ax8.transAxes)

ax8.set_title('Death Distribution by Investigation Status', fontweight='bold', fontsize=12)
ax8.set_xlabel('Investigation Status', fontweight='bold')
ax8.set_ylabel('Number of Deaths', fontweight='bold')
ax8.grid(True, alpha=0.3)

# Row 3, Subplot 9: Time series with confidence bands and footnote frequency
ax9 = plt.subplot(3, 3, 9)
# Time series with multiple confidence levels
if len(monthly_data) > 0:
    # Main prediction line
    ax9.plot(monthly_data['Date'], monthly_data['mean_deaths'], 
            color='#2E86AB', linewidth=2, label='Mean Deaths')
    
    # Multiple confidence bands
    std_dev = monthly_data['std_deaths'].fillna(0)
    ax9.fill_between(monthly_data['Date'], 
                    monthly_data['mean_deaths'] - std_dev,
                    monthly_data['mean_deaths'] + std_dev,
                    alpha=0.3, color='#2E86AB', label='68% CI')
    ax9.fill_between(monthly_data['Date'], 
                    monthly_data['mean_deaths'] - 2*std_dev,
                    monthly_data['mean_deaths'] + 2*std_dev,
                    alpha=0.2, color='#2E86AB', label='95% CI')

# Footnote frequency bars
footnote_freq = df.dropna(subset=['Date']).groupby('Date')['Footnote'].count()
if len(footnote_freq) > 0:
    ax9_twin = ax9.twinx()
    ax9_twin.bar(footnote_freq.index, footnote_freq.values, 
                alpha=0.4, color='#F18F01', width=20, label='Footnote Frequency')
    ax9_twin.set_ylabel('Footnote Count', fontweight='bold', color='#F18F01')
    ax9_twin.legend(loc='upper right')

ax9.set_title('Predictions with Uncertainty & Data Quality Indicators', fontweight='bold', fontsize=12)
ax9.set_xlabel('Date', fontweight='bold')
ax9.set_ylabel('Number of Deaths', fontweight='bold')
ax9.legend(loc='upper left')
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.savefig('drug_overdose_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()