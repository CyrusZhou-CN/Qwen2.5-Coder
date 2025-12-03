import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

# Load data
jobs_df = pd.read_csv('all_jobs.csv')
companies_df = pd.read_csv('companies.csv')

# Data preprocessing
# Convert 'Posted Before' to datetime for time series analysis
def parse_posted_before(posted_str):
    if pd.isna(posted_str):
        return None
    
    # Create a base date (assuming data was scraped in early 2020)
    base_date = datetime(2020, 1, 1)
    
    try:
        posted_str = str(posted_str).strip().lower()
        
        # Extract numbers using regex to handle malformed entries
        numbers = re.findall(r'\d+', posted_str)
        if not numbers:
            return base_date - timedelta(days=1)
        
        num = int(numbers[0])  # Take the first number found
        
        if 'h' in posted_str:
            return base_date - timedelta(hours=num)
        elif 'd' in posted_str:
            return base_date - timedelta(days=num)
        elif 'w' in posted_str:
            return base_date - timedelta(weeks=num)
        elif 'm' in posted_str:
            return base_date - timedelta(days=num*30)
        else:
            return base_date - timedelta(days=num)
    except (ValueError, IndexError):
        # If parsing fails, return a default date
        return base_date - timedelta(days=1)

jobs_df['Posted Date'] = jobs_df['Posted Before'].apply(parse_posted_before)
jobs_df = jobs_df.dropna(subset=['Posted Date'])

# Filter for 2018-2019 data
jobs_df = jobs_df[(jobs_df['Posted Date'] >= '2018-01-01') & (jobs_df['Posted Date'] < '2020-01-01')]

# If no data in 2018-2019 range, use all available data
if len(jobs_df) == 0:
    jobs_df = pd.read_csv('all_jobs.csv')
    jobs_df['Posted Date'] = jobs_df['Posted Before'].apply(parse_posted_before)
    jobs_df = jobs_df.dropna(subset=['Posted Date'])

# Create month-year column for aggregation
jobs_df['Month_Year'] = jobs_df['Posted Date'].dt.to_period('M')

# Merge with companies data
merged_df = jobs_df.merge(companies_df, on='Company Name', how='left')

# Categorize job types (technical vs non-technical)
technical_keywords = ['engineer', 'developer', 'technical', 'software', 'blockchain', 'smart contract', 'solidity', 'python', 'javascript']
jobs_df['Is_Technical'] = jobs_df['Job Title'].str.lower().str.contains('|'.join(technical_keywords), na=False)

# Categorize company sizes
def categorize_company_size(employees):
    if pd.isna(employees):
        return 'Unknown'
    employees_str = str(employees).lower()
    if any(x in employees_str for x in ['1-10', '11-50']):
        return 'Small'
    elif any(x in employees_str for x in ['51-100', '101-250']):
        return 'Medium'
    elif any(x in employees_str for x in ['251-500', '501-1000', '1001-5000']):
        return 'Large'
    else:
        return 'Unknown'

merged_df['Company_Size'] = merged_df['Number of Employees'].apply(categorize_company_size)

# Create the 2x2 subplot grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Subplot 1: Monthly job counts (line) + Average salary (bar)
monthly_jobs = jobs_df.groupby('Month_Year').size()
monthly_salary = jobs_df.groupby('Month_Year')['Salary Range'].mean()

# Ensure we have data to plot
if len(monthly_jobs) > 0:
    # Line chart for job counts
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(range(len(monthly_jobs)), monthly_jobs.values, 'b-', linewidth=2.5, marker='o', markersize=6, label='Job Postings')
    ax1.set_ylabel('Number of Job Postings', fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Bar chart for average salary (only for non-null values)
    salary_values = monthly_salary.fillna(0).values
    bars = ax1_twin.bar(range(len(monthly_salary)), salary_values, alpha=0.6, color='orange', width=0.6, label='Avg Salary')
    ax1_twin.set_ylabel('Average Salary Range', fontweight='bold', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')

    ax1.set_title('Monthly Job Postings and Average Salary Trends', fontweight='bold', fontsize=14, pad=20)
    ax1.set_xlabel('Month', fontweight='bold')
    
    # Set x-axis labels
    step = max(1, len(monthly_jobs) // 6)  # Show at most 6 labels
    ax1.set_xticks(range(0, len(monthly_jobs), step))
    ax1.set_xticklabels([str(monthly_jobs.index[i]) for i in range(0, len(monthly_jobs), step)], rotation=45)
    ax1.grid(True, alpha=0.3)

# Subplot 2: Cumulative growth of job categories (area chart)
if len(monthly_jobs) > 0:
    monthly_technical = jobs_df[jobs_df['Is_Technical']].groupby('Month_Year').size().reindex(monthly_jobs.index, fill_value=0)
    monthly_non_technical = jobs_df[~jobs_df['Is_Technical']].groupby('Month_Year').size().reindex(monthly_jobs.index, fill_value=0)

    cumulative_technical = monthly_technical.cumsum()
    cumulative_non_technical = monthly_non_technical.cumsum()

    ax2.fill_between(range(len(cumulative_technical)), 0, cumulative_technical.values, alpha=0.7, color='#2E86AB', label='Technical Roles')
    ax2.fill_between(range(len(cumulative_non_technical)), cumulative_technical.values, 
                     cumulative_technical.values + cumulative_non_technical.values, alpha=0.7, color='#A23B72', label='Non-Technical Roles')

    # Add trend lines
    if len(cumulative_technical) > 1:
        z_tech = np.polyfit(range(len(cumulative_technical)), cumulative_technical.values, 1)
        p_tech = np.poly1d(z_tech)
        ax2.plot(range(len(cumulative_technical)), p_tech(range(len(cumulative_technical))), "--", color='darkblue', linewidth=2, alpha=0.8)

        z_total = np.polyfit(range(len(cumulative_technical)), (cumulative_technical + cumulative_non_technical).values, 1)
        p_total = np.poly1d(z_total)
        ax2.plot(range(len(cumulative_technical)), p_total(range(len(cumulative_technical))), "--", color='darkred', linewidth=2, alpha=0.8)

    ax2.set_title('Cumulative Growth of Job Categories', fontweight='bold', fontsize=14, pad=20)
    ax2.set_xlabel('Month', fontweight='bold')
    ax2.set_ylabel('Cumulative Job Count', fontweight='bold')
    ax2.legend(loc='upper left')
    
    step = max(1, len(monthly_jobs) // 6)
    ax2.set_xticks(range(0, len(monthly_jobs), step))
    ax2.set_xticklabels([str(monthly_jobs.index[i]) for i in range(0, len(monthly_jobs), step)], rotation=45)
    ax2.grid(True, alpha=0.3)

# Subplot 3: Job locations composition + remote work percentage
if len(monthly_jobs) > 0:
    top_locations = jobs_df['Job Location'].value_counts().head(4).index.tolist()
    location_data = {}
    for loc in top_locations:
        location_data[loc] = jobs_df[jobs_df['Job Location'] == loc].groupby('Month_Year').size().reindex(monthly_jobs.index, fill_value=0)

    # Create stacked area chart
    bottom = np.zeros(len(monthly_jobs))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (loc, data) in enumerate(location_data.items()):
        ax3.fill_between(range(len(data)), bottom, bottom + data.values, alpha=0.7, color=colors[i], label=loc[:15])
        bottom += data.values

    # Remote work percentage (secondary axis)
    remote_jobs = jobs_df[jobs_df['Job Location'].str.contains('Remote|remote', na=False)].groupby('Month_Year').size().reindex(monthly_jobs.index, fill_value=0)
    remote_percentage = (remote_jobs / monthly_jobs * 100).fillna(0)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(len(remote_percentage)), remote_percentage.values, 'r-', linewidth=3, marker='s', markersize=6, label='Remote %')
    ax3_twin.set_ylabel('Remote Work Percentage (%)', fontweight='bold', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')

    ax3.set_title('Job Location Composition and Remote Work Trends', fontweight='bold', fontsize=14, pad=20)
    ax3.set_xlabel('Month', fontweight='bold')
    ax3.set_ylabel('Number of Jobs by Location', fontweight='bold')
    ax3.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
    
    step = max(1, len(monthly_jobs) // 6)
    ax3.set_xticks(range(0, len(monthly_jobs), step))
    ax3.set_xticklabels([str(monthly_jobs.index[i]) for i in range(0, len(monthly_jobs), step)], rotation=45)
    ax3.grid(True, alpha=0.3)

# Subplot 4: Company funding vs job posting frequency
# Filter out companies with missing funding data
funding_data = merged_df.dropna(subset=['Total Funding', 'Company_Size'])
if len(funding_data) > 0:
    # Group by company and month to get job posting frequency
    company_monthly = funding_data.groupby(['Company Name', 'Month_Year', 'Company_Size', 'Total Funding']).size().reset_index(name='Jobs_Count')
    
    # Create scatter plot with different colors for company sizes
    size_colors = {'Small': '#FF9999', 'Medium': '#66B2FF', 'Large': '#99FF99', 'Unknown': '#FFCC99'}
    
    for size_cat in company_monthly['Company_Size'].unique():
        size_data = company_monthly[company_monthly['Company_Size'] == size_cat]
        if len(size_data) > 0:
            ax4.scatter(size_data['Total Funding'], size_data['Jobs_Count'], 
                       c=size_colors.get(size_cat, '#CCCCCC'), alpha=0.6, s=60, label=f'{size_cat} Companies')
    
    # Add trend line
    if len(company_monthly) > 1:
        try:
            z = np.polyfit(company_monthly['Total Funding'], company_monthly['Jobs_Count'], 1)
            p = np.poly1d(z)
            funding_range = np.linspace(company_monthly['Total Funding'].min(), company_monthly['Total Funding'].max(), 100)
            ax4.plot(funding_range, p(funding_range), "r--", alpha=0.8, linewidth=2)
        except:
            pass  # Skip trend line if fitting fails

    ax4.set_title('Company Funding vs Job Posting Frequency', fontweight='bold', fontsize=14, pad=20)
    ax4.set_xlabel('Total Funding ($)', fontweight='bold')
    ax4.set_ylabel('Monthly Job Postings', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Use log scale only if we have positive funding values
    if company_monthly['Total Funding'].min() > 0:
        ax4.set_xscale('log')
else:
    # If no funding data, create a simple placeholder
    ax4.text(0.5, 0.5, 'No funding data available', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Company Funding vs Job Posting Frequency', fontweight='bold', fontsize=14, pad=20)

# Overall layout adjustments
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('blockchain_job_market_analysis.png', dpi=300, bbox_inches='tight')
plt.show()