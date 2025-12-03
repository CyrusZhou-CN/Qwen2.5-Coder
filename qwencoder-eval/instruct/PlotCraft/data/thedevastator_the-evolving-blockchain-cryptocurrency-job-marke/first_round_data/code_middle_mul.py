import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Load the main all_jobs.csv file which contains consolidated data
try:
    jobs_df = pd.read_csv('all_jobs.csv')
    print(f"Loaded all_jobs.csv with {len(jobs_df)} rows")
except:
    # If all_jobs.csv doesn't work, try loading individual files
    job_files = ['Consensys.csv', 'Bitfury.csv', 'Candy.csv', 'Nuri.csv', 'Binance.csv', 
                 'Coinbase.csv', 'Alchemy.csv', 'Open Sea.csv', 'Immutable.csv', 'Swiss Borg.csv',
                 'Ledger.csv', 'Bit Go.csv', 'Gsr.csv', 'Block Fi.csv', 'Figure.csv', 
                 'Blockchain.csv', 'Near.csv', 'Bitcoin Depot.csv', 'Chainalysis.csv',
                 'Okcoin.csv', 'Ripple.csv', 'Crypto.Com.csv', 'Parity Technologies.csv',
                 'Gemini.csv', 'Bit Mex.csv', 'Meta Mask.csv', 'Figment.csv', 'Kraken.csv', 'Polygon.csv']
    
    all_jobs = []
    for file in job_files:
        try:
            if os.path.exists(file):
                df = pd.read_csv(file)
                all_jobs.append(df)
                print(f"Loaded {file} with {len(df)} rows")
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue
    
    if all_jobs:
        jobs_df = pd.concat(all_jobs, ignore_index=True)
        print(f"Combined {len(all_jobs)} files into {len(jobs_df)} total rows")
    else:
        # Create sample data if no files can be loaded
        print("No job files found, creating sample data")
        jobs_df = pd.DataFrame({
            'Company Name': ['Coinbase', 'Binance', 'Kraken', 'Gemini', 'Polygon', 'Ripple', 'Chainlink', 
                           'Uniswap', 'Consensys', 'Alchemy', 'OpenSea', 'Immutable', 'BitGo', 'Chainalysis', 
                           'Figment'] * 20,
            'Job Title': ['Software Engineer', 'Product Manager', 'Data Scientist', 'Security Engineer', 
                         'Marketing Manager'] * 60,
            'Salary Range': ['$80k - $150k', '$90k - $160k', '$70k - $140k', '$100k - $180k', '$60k - $120k'] * 60,
            'Tags': ['dev:senior:blockchain', 'product manager:non tech', 'data science:python', 
                    'security:dev', 'marketing:non tech'] * 60
        })

# Load companies data
try:
    companies_df = pd.read_csv('companies.csv')
    print(f"Loaded companies.csv with {len(companies_df)} rows")
except:
    # Create sample companies data if file not found
    print("Companies file not found, creating sample data")
    companies_df = pd.DataFrame({
        'Company Name': ['Coinbase', 'Binance', 'Kraken', 'Gemini', 'Polygon', 'Ripple', 'Chainlink', 
                        'Uniswap', 'Consensys', 'Alchemy', 'OpenSea', 'Immutable', 'BitGo', 'Chainalysis', 'Figment'],
        'Total Funding': [567309825, 450000000, 200000000, 423903059, 450450000, 300000000, 150000000,
                         100000000, 200000000, 80000000, 300000000, 60000000, 170000000, 366000000, 50000000],
        'Number of Employees': ['1001-5000', '501-1000', '251-500', '501-1000', '251-500', '501-1000', '101-250',
                               '51-100', '251-500', '51-100', '101-250', '101-250', '251-500', '251-500', '51-100']
    })

# Function to extract salary midpoint
def extract_salary_midpoint(salary_range):
    if pd.isna(salary_range) or salary_range == 'NaN' or str(salary_range) == 'nan':
        return np.nan
    
    # Handle string representation of salary ranges
    salary_str = str(salary_range)
    
    # Extract numbers from salary range (handle both $80k-$150k and 80k - 150k formats)
    numbers = re.findall(r'(\d+)k?', salary_str.replace('$', '').replace(',', ''))
    
    if len(numbers) >= 2:
        try:
            min_sal = float(numbers[0])
            max_sal = float(numbers[1])
            
            # Convert to full numbers if 'k' is present
            if 'k' in salary_str.lower():
                min_sal *= 1000
                max_sal *= 1000
            
            return (min_sal + max_sal) / 2
        except:
            return np.nan
    return np.nan

# Function to classify technical vs non-technical roles
def is_technical_role(tags):
    if pd.isna(tags):
        return False
    
    tech_keywords = ['dev', 'engineer', 'technical', 'software', 'backend', 'frontend', 
                     'full stack', 'devops', 'security', 'data science', 'machine learning',
                     'blockchain', 'python', 'java', 'javascript', 'react', 'golang', 'rust',
                     'quality assurance', 'sys admin']
    
    tags_lower = str(tags).lower()
    return any(keyword in tags_lower for keyword in tech_keywords)

# Process job data
jobs_df['salary_midpoint'] = jobs_df['Salary Range'].apply(extract_salary_midpoint)
jobs_df['is_technical'] = jobs_df['Tags'].apply(is_technical_role)

print(f"Processed salary data: {jobs_df['salary_midpoint'].notna().sum()} valid salaries out of {len(jobs_df)} jobs")

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('white')

# 1. Top-left: Horizontal bar chart - Top 15 companies by job count, colored by avg salary
company_stats = jobs_df.groupby('Company Name').agg({
    'Job Title': 'count',
    'salary_midpoint': 'mean'
}).rename(columns={'Job Title': 'job_count'}).reset_index()

top_15_companies = company_stats.nlargest(15, 'job_count')

# Create color mapping based on salary (handle NaN values)
salary_values = top_15_companies['salary_midpoint'].fillna(top_15_companies['salary_midpoint'].mean())
if salary_values.max() > salary_values.min():
    salary_normalized = (salary_values - salary_values.min()) / (salary_values.max() - salary_values.min())
else:
    salary_normalized = np.full(len(salary_values), 0.5)

bars1 = ax1.barh(range(len(top_15_companies)), top_15_companies['job_count'], 
                 color=plt.cm.plasma(salary_normalized))
ax1.set_yticks(range(len(top_15_companies)))
ax1.set_yticklabels(top_15_companies['Company Name'], fontsize=10)
ax1.set_xlabel('Number of Job Postings', fontweight='bold')
ax1.set_title('Top 15 Companies by Job Postings\n(Colored by Average Salary)', fontweight='bold', fontsize=14)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(top_15_companies['job_count']):
    ax1.text(v + max(top_15_companies['job_count']) * 0.01, i, str(int(v)), va='center', fontweight='bold')

# 2. Top-right: Lollipop chart - Top 10 companies by average salary
salary_stats = jobs_df.dropna(subset=['salary_midpoint']).groupby('Company Name').agg({
    'salary_midpoint': 'mean',
    'Job Title': 'count'
}).rename(columns={'Job Title': 'job_count'}).reset_index()

if len(salary_stats) > 0:
    top_10_salary = salary_stats.nlargest(10, 'salary_midpoint')
    
    # Create lollipop chart
    y_pos = range(len(top_10_salary))
    ax2.hlines(y_pos, 0, top_10_salary['salary_midpoint'], colors='lightblue', linewidth=2)
    scatter = ax2.scatter(top_10_salary['salary_midpoint'], y_pos, 
                         s=top_10_salary['job_count']*5, c='darkblue', alpha=0.7, zorder=3)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_10_salary['Company Name'], fontsize=10)
    ax2.set_xlabel('Average Salary ($)', fontweight='bold')
    ax2.set_title('Top 10 Companies by Average Salary\n(Point size = Job Count)', fontweight='bold', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # Format salary labels
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
else:
    ax2.text(0.5, 0.5, 'No salary data available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Top 10 Companies by Average Salary\n(No data available)', fontweight='bold', fontsize=14)

# 3. Bottom-left: Stacked horizontal bar - Top 12 companies by job count, segmented by technical/non-technical
tech_stats = jobs_df.groupby(['Company Name', 'is_technical']).size().unstack(fill_value=0)
tech_stats['total'] = tech_stats.sum(axis=1)
top_12_tech = tech_stats.nlargest(12, 'total')

# Create stacked bar chart
technical_jobs = top_12_tech[True] if True in top_12_tech.columns else np.zeros(len(top_12_tech))
non_technical_jobs = top_12_tech[False] if False in top_12_tech.columns else np.zeros(len(top_12_tech))

y_pos = range(len(top_12_tech))
ax3.barh(y_pos, technical_jobs, label='Technical Roles', color='steelblue', alpha=0.8)
ax3.barh(y_pos, non_technical_jobs, left=technical_jobs, label='Non-Technical Roles', 
         color='lightcoral', alpha=0.8)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_12_tech.index, fontsize=10)
ax3.set_xlabel('Number of Job Postings', fontweight='bold')
ax3.set_title('Top 12 Companies by Job Count\n(Technical vs Non-Technical Roles)', fontweight='bold', fontsize=14)
ax3.legend(loc='lower right')
ax3.grid(axis='x', alpha=0.3)

# 4. Bottom-right: Dot plot - Company funding vs job posting activity
# Merge job counts with company funding data
company_job_counts = jobs_df.groupby('Company Name').size().reset_index(name='job_count')
funding_analysis = companies_df.merge(company_job_counts, on='Company Name', how='inner')
funding_analysis = funding_analysis.dropna(subset=['Total Funding'])

if len(funding_analysis) > 0:
    funding_analysis = funding_analysis.sort_values('Total Funding', ascending=True)
    
    # Create employee count categories for color coding
    def categorize_employees(emp_str):
        if pd.isna(emp_str):
            return 'Unknown'
        emp_str = str(emp_str)
        if any(x in emp_str for x in ['1-10', '11-50']):
            return 'Small (1-50)'
        elif any(x in emp_str for x in ['51-100', '101-250']):
            return 'Medium (51-250)'
        elif any(x in emp_str for x in ['251-500', '501-1000']):
            return 'Large (251-1000)'
        elif any(x in emp_str for x in ['1001-5000', '5001-10000']):
            return 'Very Large (1000+)'
        else:
            return 'Unknown'
    
    funding_analysis['emp_category'] = funding_analysis['Number of Employees'].apply(categorize_employees)
    
    # Color mapping for employee categories
    emp_colors = {'Small (1-50)': 'lightgreen', 'Medium (51-250)': 'gold', 
                  'Large (251-1000)': 'orange', 'Very Large (1000+)': 'red', 'Unknown': 'gray'}
    
    colors = [emp_colors.get(cat, 'gray') for cat in funding_analysis['emp_category']]
    
    ax4.scatter(funding_analysis['Total Funding']/1e6, funding_analysis['job_count'], 
               c=colors, s=100, alpha=0.7)
    
    # Add company labels for notable companies
    for i, row in funding_analysis.iterrows():
        if row['job_count'] > funding_analysis['job_count'].quantile(0.7) or \
           row['Total Funding'] > funding_analysis['Total Funding'].quantile(0.7):
            ax4.annotate(row['Company Name'], 
                        (row['Total Funding']/1e6, row['job_count']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Total Funding ($ Millions)', fontweight='bold')
    ax4.set_ylabel('Number of Job Postings', fontweight='bold')
    ax4.set_title('Company Funding vs Job Posting Activity\n(Color = Employee Count Range)', fontweight='bold', fontsize=14)
    ax4.grid(alpha=0.3)
    
    # Create custom legend for employee categories
    from matplotlib.patches import Patch
    unique_categories = funding_analysis['emp_category'].unique()
    legend_elements = [Patch(facecolor=emp_colors[category], label=category) 
                      for category in unique_categories if category in emp_colors]
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)
else:
    ax4.text(0.5, 0.5, 'No funding data available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Company Funding vs Job Posting Activity\n(No data available)', fontweight='bold', fontsize=14)

# Adjust layout and save
plt.tight_layout(pad=3.0)
plt.savefig('blockchain_crypto_companies_ranking_analysis.png', dpi=300, bbox_inches='tight')
plt.show()