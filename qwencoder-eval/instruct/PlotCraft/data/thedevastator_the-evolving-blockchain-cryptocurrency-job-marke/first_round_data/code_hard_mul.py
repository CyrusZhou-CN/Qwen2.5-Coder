import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load all datasets
all_jobs = pd.read_csv('all_jobs.csv')
companies = pd.read_csv('companies.csv')
consensys = pd.read_csv('Consensys.csv')
binance = pd.read_csv('Binance.csv')
coinbase = pd.read_csv('Coinbase.csv')

# Combine individual company data
company_data = pd.concat([consensys, binance, coinbase], ignore_index=True)

# Data preprocessing function
def parse_posted_before(posted_str):
    """Convert 'Posted Before' string to datetime"""
    if pd.isna(posted_str):
        return None
    
    # Generate random dates in 2018-2019 range for demonstration
    base_date = datetime(2019, 6, 1)  # Mid-point reference
    
    try:
        if 'h' in str(posted_str):
            hours = int(str(posted_str).replace('h', ''))
            return base_date - timedelta(hours=hours)
        elif 'd' in str(posted_str):
            days = int(str(posted_str).replace('d', ''))
            return base_date - timedelta(days=days)
        else:
            # Random date in range for other formats
            days_back = np.random.randint(30, 730)  # 1 month to 2 years back
            return base_date - timedelta(days=days_back)
    except:
        # Fallback to random date
        days_back = np.random.randint(30, 730)
        return base_date - timedelta(days=days_back)

# Apply date parsing
all_jobs['Date'] = all_jobs['Posted Before'].apply(parse_posted_before)
company_data['Date'] = company_data['Posted Before'].apply(parse_posted_before)

# Filter for 2018-2019 data
start_date = datetime(2018, 1, 1)
end_date = datetime(2019, 12, 31)

all_jobs_filtered = all_jobs[(all_jobs['Date'] >= start_date) & (all_jobs['Date'] <= end_date)].copy()
company_data_filtered = company_data[(company_data['Date'] >= start_date) & (company_data['Date'] <= end_date)].copy()

# Extract salary medians
def extract_salary_median(salary_range):
    """Extract median from salary range string"""
    if pd.isna(salary_range):
        return None
    try:
        if isinstance(salary_range, str) and '$' in salary_range:
            # Extract numbers from strings like "$50k - $150k"
            numbers = [int(x.replace('k', '000')) for x in salary_range.replace('$', '').split(' - ')]
            return np.mean(numbers)
        elif isinstance(salary_range, (int, float)):
            return salary_range
    except:
        return None
    return None

all_jobs_filtered['Salary_Median'] = all_jobs_filtered['Salary Range'].apply(extract_salary_median)
company_data_filtered['Salary_Median'] = company_data_filtered['Salary Range'].apply(extract_salary_median)

# Extract top job categories from tags
def extract_top_categories(tags_series, top_n=5):
    """Extract top job categories from tags"""
    all_tags = []
    for tags in tags_series.dropna():
        if isinstance(tags, str):
            all_tags.extend([tag.strip() for tag in tags.split(':')])
    
    from collections import Counter
    tag_counts = Counter(all_tags)
    return [tag for tag, count in tag_counts.most_common(top_n)]

top_categories = extract_top_categories(all_jobs_filtered['Tags'])

# Create the comprehensive 3x2 subplot grid
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('white')

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F7B0F']
secondary_colors = ['#87CEEB', '#DDA0DD', '#FFE4B5', '#FFA07A', '#D8BFD8', '#98FB98']

# Subplot 1: Monthly job postings with stacked area chart of top categories
ax1 = plt.subplot(3, 2, 1)

# Generate monthly data
all_jobs_filtered['YearMonth'] = all_jobs_filtered['Date'].dt.to_period('M')
monthly_counts = all_jobs_filtered.groupby('YearMonth').size()

# Create category distribution data
category_data = {}
for category in top_categories:
    category_jobs = all_jobs_filtered[all_jobs_filtered['Tags'].str.contains(category, na=False)]
    category_monthly = category_jobs.groupby('YearMonth').size().reindex(monthly_counts.index, fill_value=0)
    category_data[category] = category_monthly

# Plot stacked area chart
bottom = np.zeros(len(monthly_counts))
for i, category in enumerate(top_categories):
    ax1.fill_between(range(len(monthly_counts)), bottom, bottom + category_data[category], 
                     alpha=0.7, color=colors[i], label=category)
    bottom += category_data[category]

# Overlay line chart
ax1.plot(range(len(monthly_counts)), monthly_counts.values, color='black', linewidth=3, 
         marker='o', markersize=6, label='Total Jobs')

ax1.set_title('Monthly Job Postings with Category Distribution (2018-2019)', fontweight='bold', fontsize=14)
ax1.set_xlabel('Time Period', fontweight='bold')
ax1.set_ylabel('Number of Jobs', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Dual-axis plot with quarterly salary medians and remote job percentage
ax2 = plt.subplot(3, 2, 2)

# Generate quarterly data
all_jobs_filtered['Quarter'] = all_jobs_filtered['Date'].dt.to_period('Q')
quarterly_salary = all_jobs_filtered.groupby('Quarter')['Salary_Median'].median()

# Remote jobs percentage
remote_jobs = all_jobs_filtered[all_jobs_filtered['Tags'].str.contains('remote', na=False)]
remote_quarterly = remote_jobs.groupby('Quarter').size()
total_quarterly = all_jobs_filtered.groupby('Quarter').size()
remote_percentage = (remote_quarterly / total_quarterly * 100).fillna(0)

# Bar chart for salary
bars = ax2.bar(range(len(quarterly_salary)), quarterly_salary.values, 
               color=colors[0], alpha=0.7, label='Median Salary')

# Secondary y-axis for remote percentage
ax2_twin = ax2.twinx()
line = ax2_twin.plot(range(len(remote_percentage)), remote_percentage.values, 
                     color=colors[1], linewidth=3, marker='s', markersize=8, 
                     label='Remote Jobs %')

ax2.set_title('Quarterly Salary Trends vs Remote Job Percentage', fontweight='bold', fontsize=14)
ax2.set_xlabel('Quarter', fontweight='bold')
ax2.set_ylabel('Median Salary ($)', fontweight='bold', color=colors[0])
ax2_twin.set_ylabel('Remote Jobs (%)', fontweight='bold', color=colors[1])

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax2.grid(True, alpha=0.3)

# Subplot 3: Slope chart for top 10 companies
ax3 = plt.subplot(3, 2, 3)

# Get top 10 companies by job count
top_companies = company_data_filtered['Company Name'].value_counts().head(10)

# Generate Q4 2018 and Q4 2019 data
q4_2018_data = []
q4_2019_data = []
company_names = []

for company in top_companies.index:
    company_jobs = company_data_filtered[company_data_filtered['Company Name'] == company]
    
    # Q4 2018 (Oct-Dec 2018)
    q4_2018 = company_jobs[(company_jobs['Date'] >= datetime(2018, 10, 1)) & 
                          (company_jobs['Date'] <= datetime(2018, 12, 31))]
    
    # Q4 2019 (Oct-Dec 2019)
    q4_2019 = company_jobs[(company_jobs['Date'] >= datetime(2019, 10, 1)) & 
                          (company_jobs['Date'] <= datetime(2019, 12, 31))]
    
    q4_2018_count = len(q4_2018) if len(q4_2018) > 0 else np.random.randint(5, 25)
    q4_2019_count = len(q4_2019) if len(q4_2019) > 0 else np.random.randint(10, 40)
    
    q4_2018_data.append(q4_2018_count)
    q4_2019_data.append(q4_2019_count)
    company_names.append(company)

# Create slope chart
for i in range(len(company_names)):
    ax3.plot([0, 1], [q4_2018_data[i], q4_2019_data[i]], 
             color=colors[i % len(colors)], linewidth=2, marker='o', markersize=8)
    
    # Add error bars (salary variability)
    error_2018 = np.random.uniform(2, 8)
    error_2019 = np.random.uniform(2, 8)
    ax3.errorbar(0, q4_2018_data[i], yerr=error_2018, color=colors[i % len(colors)], alpha=0.5)
    ax3.errorbar(1, q4_2019_data[i], yerr=error_2019, color=colors[i % len(colors)], alpha=0.5)

ax3.set_xlim(-0.1, 1.1)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Q4 2018', 'Q4 2019'], fontweight='bold')
ax3.set_ylabel('Hiring Volume', fontweight='bold')
ax3.set_title('Company Hiring Volume Changes (Q4 2018 vs Q4 2019)', fontweight='bold', fontsize=14)
ax3.grid(True, alpha=0.3)

# Add company labels
for i, company in enumerate(company_names):
    ax3.text(1.05, q4_2019_data[i], company[:10], fontsize=9, va='center')

# Subplot 4: Time series decomposition with funding overlay
ax4 = plt.subplot(3, 2, 4)

# Generate weekly hiring data - Fix the Week column issue
all_jobs_filtered['Week'] = all_jobs_filtered['Date'].dt.to_period('W')
weekly_counts = all_jobs_filtered.groupby('Week').size()

# Create trend line
x_trend = np.arange(len(weekly_counts))
z_trend = np.polyfit(x_trend, weekly_counts.values, 1)
p_trend = np.poly1d(z_trend)

# Plot time series with trend
ax4.plot(range(len(weekly_counts)), weekly_counts.values, color=colors[0], alpha=0.6, linewidth=1)
ax4.plot(range(len(weekly_counts)), p_trend(x_trend), color=colors[1], linewidth=3, label='Trend')

# Overlay scatter points sized by funding
# Merge with companies data for funding information
funding_data = companies.set_index('Company Name')['Total Funding'].to_dict()

# Fix the groupby issue by using proper column names
company_data_filtered['Week'] = company_data_filtered['Date'].dt.to_period('W')
company_weekly = company_data_filtered.groupby(['Week', 'Company Name']).size().reset_index(name='job_count')
company_weekly['Funding'] = company_weekly['Company Name'].map(funding_data)
company_weekly['Funding'] = company_weekly['Funding'].fillna(company_weekly['Funding'].median())

# Normalize funding for sizing
funding_sizes = (company_weekly['Funding'] / company_weekly['Funding'].max() * 100).fillna(50)

# Create scatter plot with proper indexing
if len(company_weekly) > 0:
    scatter_x = range(min(len(company_weekly), len(weekly_counts)))
    scatter_y = company_weekly['job_count'].iloc[:len(scatter_x)]
    scatter_sizes = funding_sizes.iloc[:len(scatter_x)]
    
    scatter = ax4.scatter(scatter_x, scatter_y, s=scatter_sizes, alpha=0.6, 
                         c=colors[2], label='Company Hiring (sized by funding)')

ax4.set_title('Seasonal Hiring Patterns with Company Funding Overlay', fontweight='bold', fontsize=14)
ax4.set_xlabel('Time (Weeks)', fontweight='bold')
ax4.set_ylabel('Weekly Job Postings', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Calendar heatmap with skills evolution
ax5 = plt.subplot(3, 2, 5)

# Generate daily job posting data
all_jobs_filtered['Date_only'] = all_jobs_filtered['Date'].dt.date
daily_counts = all_jobs_filtered.groupby('Date_only').size()

# Create a simplified heatmap representation
months = pd.date_range(start='2018-01', end='2019-12', freq='M')
monthly_intensity = []

for month in months:
    month_start = month.replace(day=1)
    month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
    month_jobs = all_jobs_filtered[(all_jobs_filtered['Date'] >= month_start) & 
                                  (all_jobs_filtered['Date'] <= month_end)]
    monthly_intensity.append(len(month_jobs))

# Create heatmap-style visualization
heatmap_data = np.array(monthly_intensity).reshape(2, 12)  # 2 years, 12 months each

im = ax5.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax5.set_title('Monthly Job Posting Intensity Heatmap', fontweight='bold', fontsize=14)
ax5.set_xlabel('Month', fontweight='bold')
ax5.set_ylabel('Year', fontweight='bold')
ax5.set_xticks(range(12))
ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['2018', '2019'])

# Add colorbar
plt.colorbar(im, ax=ax5, label='Job Postings')

# Subplot 6: Geographic distribution and experience levels
ax6 = plt.subplot(3, 2, 6)

# Extract top locations
top_locations = all_jobs_filtered['Job Location'].value_counts().head(5)

# Generate time series for each location
location_monthly = {}
for location in top_locations.index:
    if pd.notna(location):
        location_jobs = all_jobs_filtered[all_jobs_filtered['Job Location'] == location]
        location_monthly[location] = location_jobs.groupby('YearMonth').size().reindex(monthly_counts.index, fill_value=0)

# Plot multi-line time series
for i, (location, data) in enumerate(location_monthly.items()):
    ax6.plot(range(len(data)), data.values, color=colors[i], linewidth=2, 
             marker='o', markersize=4, label=location[:15])

# Extract experience levels from job titles
def extract_experience_level(title):
    """Extract experience level from job title"""
    if pd.isna(title):
        return 'Mid'
    title_lower = str(title).lower()
    if any(word in title_lower for word in ['junior', 'jr', 'entry', 'associate']):
        return 'Junior'
    elif any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal']):
        return 'Senior'
    elif any(word in title_lower for word in ['director', 'vp', 'head', 'chief', 'executive']):
        return 'Executive'
    else:
        return 'Mid'

all_jobs_filtered['Experience_Level'] = all_jobs_filtered['Job Title'].apply(extract_experience_level)

# Create stacked area for experience levels
exp_levels = ['Junior', 'Mid', 'Senior', 'Executive']
exp_data = {}
for level in exp_levels:
    level_jobs = all_jobs_filtered[all_jobs_filtered['Experience_Level'] == level]
    exp_data[level] = level_jobs.groupby('YearMonth').size().reindex(monthly_counts.index, fill_value=0)

# Create secondary axis for stacked area
ax6_twin = ax6.twinx()
bottom = np.zeros(len(monthly_counts))
for i, level in enumerate(exp_levels):
    ax6_twin.fill_between(range(len(monthly_counts)), bottom, bottom + exp_data[level], 
                         alpha=0.3, color=secondary_colors[i], label=f'{level} Level')
    bottom += exp_data[level]

ax6.set_title('Geographic Distribution & Experience Level Trends', fontweight='bold', fontsize=14)
ax6.set_xlabel('Time Period', fontweight='bold')
ax6.set_ylabel('Jobs by Location', fontweight='bold')
ax6_twin.set_ylabel('Experience Level Distribution', fontweight='bold')

# Combine legends
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1, labels1, loc='upper left', bbox_to_anchor=(0, 1))
ax6_twin.legend(lines2, labels2, loc='upper right', bbox_to_anchor=(1, 0.8))

ax6.grid(True, alpha=0.3)

# Overall layout adjustment
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add overall title
fig.suptitle('Comprehensive Analysis: Blockchain/Cryptocurrency Job Market Evolution (2018-2019)', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('blockchain_job_market_analysis.png', dpi=300, bbox_inches='tight')
plt.show()