import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult.csv')

# Data preprocessing - clean column names and values
df.columns = df.columns.str.strip()
# Clean all string columns to remove leading/trailing spaces
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Replace '?' with NaN and drop rows with missing values
df = df.replace('?', np.nan)
df = df.dropna()

# Sample data for faster processing if dataset is large
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)

# Check unique income values and create mapping
print("Unique Income values:", df['Income'].unique())
income_values = df['Income'].unique()

# Create color palette for income levels - handle any income value format
if len(income_values) >= 2:
    income_colors = {income_values[0]: '#3498db', income_values[1]: '#e74c3c'}
else:
    income_colors = {income_values[0]: '#3498db'}

# Create the 3x3 subplot grid
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# 1. Top-left: Age distribution with histogram and KDE overlay
ax1 = plt.subplot(3, 3, 1)
for income in df['Income'].unique():
    subset = df[df['Income'] == income]
    if len(subset) > 0:
        ax1.hist(subset['Age'], bins=20, alpha=0.6, label=income, 
                 color=income_colors[income], density=True)

ax1.set_title('Age Distribution by Income Level', fontweight='bold', fontsize=10)
ax1.set_xlabel('Age')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Top-center: Education years distribution with box plot by gender
ax2 = plt.subplot(3, 3, 2)
genders = df['Gender'].unique()
edu_data = []
gender_labels = []

for gender in genders:
    gender_edu = df[df['Gender'] == gender]['EducationNum'].values
    if len(gender_edu) > 0:
        edu_data.append(gender_edu)
        gender_labels.append(gender)

if len(edu_data) > 0:
    bp = ax2.boxplot(edu_data, labels=gender_labels, patch_artist=True)
    colors = ['#3498db', '#e74c3c']
    for i, patch in enumerate(bp['boxes']):
        if i < len(colors):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)

ax2.set_title('Education Years by Gender', fontweight='bold', fontsize=10)
ax2.set_xlabel('Gender')
ax2.set_ylabel('Education Years')
ax2.grid(True, alpha=0.3)

# 3. Top-right: Hours per week distribution by workclass
ax3 = plt.subplot(3, 3, 3)
top_workclasses = df['Workclass'].value_counts().head(3).index
colors_work = ['#3498db', '#e74c3c', '#2ecc71']

for i, workclass in enumerate(top_workclasses):
    subset = df[df['Workclass'] == workclass]
    if len(subset) > 0:
        ax3.hist(subset['Hours per Week'], bins=15, alpha=0.6, 
                 label=str(workclass)[:10], color=colors_work[i % len(colors_work)], density=True)

ax3.set_title('Hours per Week by Workclass', fontweight='bold', fontsize=10)
ax3.set_xlabel('Hours per Week')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Middle-left: Capital gain distribution (log scale, non-zero)
ax4 = plt.subplot(3, 3, 4)
non_zero_gains = df[df['Capital Gain'] > 0]['Capital Gain']
if len(non_zero_gains) > 10:
    log_gains = np.log10(non_zero_gains)
    ax4.hist(log_gains, bins=20, alpha=0.7, color='#9b59b6', density=True)
    ax4.set_xlabel('Log10(Capital Gain)')
    ax4.set_ylabel('Density')
else:
    ax4.text(0.5, 0.5, 'Insufficient non-zero\nCapital Gain data', 
             ha='center', va='center', transform=ax4.transAxes)

ax4.set_title('Capital Gain Distribution (Log Scale)', fontweight='bold', fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Middle-center: Marital status distribution with income proportions
ax5 = plt.subplot(3, 3, 5)
try:
    marital_income = pd.crosstab(df['Marital Status'], df['Income'], normalize='index')
    
    # Get available income categories
    available_incomes = marital_income.columns.tolist()
    
    # Stacked bar chart
    bottom = np.zeros(len(marital_income))
    for i, income in enumerate(available_incomes):
        color = list(income_colors.values())[i % len(income_colors)]
        bars = ax5.bar(range(len(marital_income)), marital_income[income], 
                      bottom=bottom, label=income, color=color)
        bottom += marital_income[income]

    ax5.set_title('Marital Status by Income', fontweight='bold', fontsize=10)
    ax5.set_xlabel('Marital Status')
    ax5.set_ylabel('Proportion')
    ax5.set_xticks(range(len(marital_income)))
    ax5.set_xticklabels([str(x)[:10] for x in marital_income.index], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
except Exception as e:
    ax5.text(0.5, 0.5, f'Error in marital\nstatus plot:\n{str(e)[:30]}', 
             ha='center', va='center', transform=ax5.transAxes)

# 6. Middle-right: Race distribution with income percentage
ax6 = plt.subplot(3, 3, 6)
try:
    race_income = pd.crosstab(df['Race'], df['Income'])
    race_total = race_income.sum(axis=1)
    
    # Get the high income category (assume it's the second one or contains '>')
    high_income_col = None
    for col in race_income.columns:
        if '>' in str(col) or col == race_income.columns[-1]:
            high_income_col = col
            break
    
    if high_income_col is None:
        high_income_col = race_income.columns[-1]
    
    race_pct_high = (race_income[high_income_col] / race_total * 100).sort_values(ascending=True)

    bars = ax6.barh(range(len(race_pct_high)), race_pct_high.values, 
                    color=['#e74c3c' if x > race_pct_high.mean() else '#3498db' for x in race_pct_high.values])

    # Add percentage annotations
    for i, pct in enumerate(race_pct_high.values):
        ax6.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=8)

    ax6.set_title('Race: % with Higher Income', fontweight='bold', fontsize=10)
    ax6.set_xlabel('Percentage')
    ax6.set_yticks(range(len(race_pct_high)))
    ax6.set_yticklabels(race_pct_high.index)
    ax6.grid(True, alpha=0.3, axis='x')
except Exception as e:
    ax6.text(0.5, 0.5, f'Error in race plot:\n{str(e)[:30]}', 
             ha='center', va='center', transform=ax6.transAxes)

# 7. Bottom-left: Top occupations
ax7 = plt.subplot(3, 3, 7)
try:
    top_occupations = df['Occupation'].value_counts().head(6)
    bars = ax7.bar(range(len(top_occupations)), top_occupations.values, 
                   color='#1abc9c', alpha=0.7)

    ax7.set_title('Top Occupations', fontweight='bold', fontsize=10)
    ax7.set_xlabel('Occupation')
    ax7.set_ylabel('Count')
    ax7.set_xticks(range(len(top_occupations)))
    ax7.set_xticklabels([str(occ)[:8] for occ in top_occupations.index], rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)
except Exception as e:
    ax7.text(0.5, 0.5, f'Error in occupation plot:\n{str(e)[:30]}', 
             ha='center', va='center', transform=ax7.transAxes)

# 8. Bottom-center: Top native countries with income ratios
ax8 = plt.subplot(3, 3, 8)
try:
    top_countries = df['Native Country'].value_counts().head(8)
    country_income = pd.crosstab(df['Native Country'], df['Income'])
    
    # Get high income column
    high_income_col = None
    for col in country_income.columns:
        if '>' in str(col) or col == country_income.columns[-1]:
            high_income_col = col
            break
    
    if high_income_col is None:
        high_income_col = country_income.columns[-1]
    
    country_ratio = (country_income[high_income_col] / country_income.sum(axis=1) * 100)
    country_ratio = country_ratio[top_countries.index]

    # Lollipop chart
    markerline, stemlines, baseline = ax8.stem(range(len(top_countries)), top_countries.values, basefmt=' ')
    plt.setp(stemlines, 'color', '#3498db', 'linewidth', 2)
    plt.setp(markerline, 'color', '#e74c3c', 'markersize', 8)

    # Add ratio annotations
    for i, ratio in enumerate(country_ratio):
        if not np.isnan(ratio):
            ax8.text(i, top_countries.iloc[i] + max(top_countries.values) * 0.05, f'{ratio:.1f}%', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax8.set_title('Top Countries with Income Ratios', fontweight='bold', fontsize=10)
    ax8.set_xlabel('Native Country')
    ax8.set_ylabel('Count')
    ax8.set_xticks(range(len(top_countries)))
    ax8.set_xticklabels([str(country)[:8] for country in top_countries.index], rotation=45, ha='right')
    ax8.grid(True, alpha=0.3)
except Exception as e:
    ax8.text(0.5, 0.5, f'Error in country plot:\n{str(e)[:30]}', 
             ha='center', va='center', transform=ax8.transAxes)

# 9. Bottom-right: Capital gain vs capital loss scatter plot
ax9 = plt.subplot(3, 3, 9)
try:
    # Sample data for scatter plot to avoid timeout
    scatter_sample = df[(df['Capital Gain'] > 0) | (df['capital loss'] > 0)]
    if len(scatter_sample) > 500:
        scatter_sample = scatter_sample.sample(n=500, random_state=42)

    if len(scatter_sample) > 0:
        for income in scatter_sample['Income'].unique():
            subset = scatter_sample[scatter_sample['Income'] == income]
            if len(subset) > 0:
                ax9.scatter(subset['Capital Gain'], subset['capital loss'], 
                           alpha=0.6, label=income, color=income_colors[income], s=20)
        ax9.legend()
        ax9.set_xlabel('Capital Gain')
        ax9.set_ylabel('Capital Loss')
    else:
        ax9.text(0.5, 0.5, 'No significant\nCapital data', 
                 ha='center', va='center', transform=ax9.transAxes)

    ax9.set_title('Capital Gain vs Loss', fontweight='bold', fontsize=10)
    ax9.grid(True, alpha=0.3)
except Exception as e:
    ax9.text(0.5, 0.5, f'Error in capital plot:\n{str(e)[:30]}', 
             ha='center', va='center', transform=ax9.transAxes)

# Adjust layout
plt.tight_layout(pad=1.5)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# Save the plot
plt.savefig('census_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()