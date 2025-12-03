import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('DataSet.csv')

# Convert binary columns to numeric
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
for col in binary_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})

# Clean and prepare categorical data
df['employment_type'] = df['employment_type'].fillna('Unknown')
df['required_experience'] = df['required_experience'].fillna('Not Specified')
df['required_education'] = df['required_education'].fillna('Not Specified')
df['industry'] = df['industry'].fillna('Other')
df['function'] = df['function'].fillna('Other')
df['department'] = df['department'].fillna('Other')

# Create experience level mapping
exp_mapping = {
    'Internship': 0, 'Entry level': 1, 'Associate': 2, 'Mid-Senior level': 3,
    'Director': 4, 'Executive': 5, 'Not Applicable': 1, 'Not Specified': 1
}
df['exp_numeric'] = df['required_experience'].map(exp_mapping).fillna(1)

# Create education level mapping
edu_mapping = {
    'High School or equivalent': 1, 'Some College Coursework Completed': 2,
    'Associate Degree': 3, 'Bachelor\'s Degree': 4, 'Master\'s Degree': 5,
    'Doctorate': 6, 'Professional': 5, 'Certification': 3, 'Not Specified': 2
}
df['edu_numeric'] = df['required_education'].map(edu_mapping).fillna(2)

# Extract salary information (simplified)
def extract_salary_range(salary_str):
    if pd.isna(salary_str) or salary_str == '':
        return np.nan
    import re
    numbers = re.findall(r'\d+', str(salary_str))
    if numbers:
        return np.mean([int(x) for x in numbers[:2]])
    return np.nan

df['salary_numeric'] = df['salary_range'].apply(extract_salary_range)

# Create figure
fig = plt.figure(figsize=(20, 16), facecolor='white')

# Subplot 1: Scatter plot with experience vs education
ax1 = plt.subplot(3, 3, 1)
fraud_data = df[df['fraudulent'] == 1]
legit_data = df[df['fraudulent'] == 0]

ax1.scatter(legit_data['exp_numeric'], legit_data['edu_numeric'], 
           alpha=0.6, c='#2E86AB', s=20, label='Legitimate')
ax1.scatter(fraud_data['exp_numeric'], fraud_data['edu_numeric'], 
           alpha=0.8, c='#F24236', s=20, label='Fraudulent')

ax1.set_xlabel('Experience Level')
ax1.set_ylabel('Education Level')
ax1.set_title('Experience vs Education by Fraud Status')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Salary distribution violin plot
ax2 = plt.subplot(3, 3, 2)
salary_data = df.dropna(subset=['salary_numeric'])
if len(salary_data) > 20:
    fraud_salaries = salary_data[salary_data['fraudulent'] == 1]['salary_numeric'].dropna()
    legit_salaries = salary_data[salary_data['fraudulent'] == 0]['salary_numeric'].dropna()
    
    if len(fraud_salaries) > 5 and len(legit_salaries) > 5:
        data_to_plot = [legit_salaries, fraud_salaries]
        ax2.violinplot(data_to_plot, positions=[0, 1], showmeans=True)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Legitimate', 'Fraudulent'])
    else:
        ax2.text(0.5, 0.5, 'Insufficient salary data', ha='center', va='center', transform=ax2.transAxes)
else:
    ax2.text(0.5, 0.5, 'Insufficient salary data', ha='center', va='center', transform=ax2.transAxes)

ax2.set_title('Salary Distribution by Fraud Status')

# Subplot 3: Employment type stacked bar
ax3 = plt.subplot(3, 3, 3)
emp_fraud = df.groupby(['employment_type', 'fraudulent']).size().unstack(fill_value=0)
emp_fraud_pct = emp_fraud.div(emp_fraud.sum(axis=1), axis=0) * 100

# Limit to top 5 employment types
emp_fraud_pct = emp_fraud_pct.head(5)
emp_fraud_pct.plot(kind='bar', stacked=True, ax=ax3, color=['#2E86AB', '#F24236'])

ax3.set_xlabel('Employment Type')
ax3.set_ylabel('Percentage')
ax3.set_title('Employment Type Composition')
ax3.tick_params(axis='x', rotation=45)
ax3.legend(['Legitimate', 'Fraudulent'])

# Subplot 4: Correlation heatmap
ax4 = plt.subplot(3, 3, 4)
corr_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
corr_matrix = df[corr_cols].corr()

im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr_cols)))
ax4.set_yticks(range(len(corr_cols)))
ax4.set_xticklabels(corr_cols, rotation=45)
ax4.set_yticklabels(corr_cols)

# Add correlation values
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold')

ax4.set_title('Feature Correlation Matrix')

# Subplot 5: Parallel coordinates (simplified)
ax5 = plt.subplot(3, 3, 5)
sample_df = df.sample(n=min(500, len(df)), random_state=42)

features = ['exp_numeric', 'edu_numeric', 'telecommuting', 'has_company_logo']
for idx, row in sample_df.iterrows():
    values = [row[col] for col in features]
    color = '#F24236' if row['fraudulent'] == 1 else '#2E86AB'
    alpha = 0.7 if row['fraudulent'] == 1 else 0.3
    ax5.plot(range(len(features)), values, color=color, alpha=alpha, linewidth=0.8)

ax5.set_xticks(range(len(features)))
ax5.set_xticklabels(['Experience', 'Education', 'Remote', 'Logo'])
ax5.set_title('Parallel Coordinates: Job Features')
ax5.grid(True, alpha=0.3)

# Subplot 6: Network graph (simplified)
ax6 = plt.subplot(3, 3, 6)
industry_dept = df.groupby(['industry', 'department']).size().reset_index(name='count')
industry_dept = industry_dept[industry_dept['count'] >= 20].head(15)

G = nx.Graph()
for _, row in industry_dept.iterrows():
    industry = row['industry'][:10]
    dept = row['department'][:10]
    G.add_edge(industry, dept, weight=row['count'])

if len(G.nodes()) > 0:
    pos = nx.spring_layout(G, k=1, iterations=20)
    nx.draw(G, pos, ax=ax6, node_color='lightblue', node_size=300, 
            font_size=8, font_weight='bold', with_labels=True)

ax6.set_title('Industry-Department Network')
ax6.axis('off')

# Subplot 7: Industry analysis bar chart
ax7 = plt.subplot(3, 3, 7)
industry_stats = df.groupby('industry').agg({
    'fraudulent': ['count', 'sum']
}).reset_index()
industry_stats.columns = ['industry', 'total_jobs', 'fraud_jobs']
industry_stats['fraud_rate'] = industry_stats['fraud_jobs'] / industry_stats['total_jobs']
industry_stats = industry_stats.sort_values('total_jobs', ascending=False).head(8)

bars = ax7.barh(range(len(industry_stats)), industry_stats['total_jobs'], 
               color=plt.cm.Reds(industry_stats['fraud_rate']))

ax7.set_yticks(range(len(industry_stats)))
ax7.set_yticklabels([ind[:15] for ind in industry_stats['industry']])
ax7.set_xlabel('Number of Jobs')
ax7.set_title('Industry Job Volume & Fraud Rates')

# Subplot 8: Radar chart
ax8 = plt.subplot(3, 3, 8, projection='polar')
radar_features = ['has_company_logo', 'has_questions', 'telecommuting']
fraud_means = df[df['fraudulent'] == 1][radar_features].mean()
legit_means = df[df['fraudulent'] == 0][radar_features].mean()

angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

fraud_values = fraud_means.tolist() + [fraud_means.iloc[0]]
legit_values = legit_means.tolist() + [legit_means.iloc[0]]

ax8.plot(angles, legit_values, 'o-', linewidth=2, label='Legitimate', color='#2E86AB')
ax8.fill(angles, legit_values, alpha=0.25, color='#2E86AB')
ax8.plot(angles, fraud_values, 'o-', linewidth=2, label='Fraudulent', color='#F24236')
ax8.fill(angles, fraud_values, alpha=0.25, color='#F24236')

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(['Has Logo', 'Has Questions', 'Remote Work'])
ax8.set_title('Job Characteristics Profile')
ax8.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

# Subplot 9: Bubble chart
ax9 = plt.subplot(3, 3, 9)
bubble_data = df.groupby(['industry', 'function']).agg({
    'fraudulent': ['count', 'mean']
}).reset_index()
bubble_data.columns = ['industry', 'function', 'job_count', 'fraud_rate']
bubble_data = bubble_data[bubble_data['job_count'] >= 5].head(30)

if len(bubble_data) > 0:
    scatter = ax9.scatter(range(len(bubble_data)), bubble_data['fraud_rate'], 
                         s=bubble_data['job_count']*5, 
                         c=bubble_data['fraud_rate'], 
                         cmap='Reds', alpha=0.7)
    
    # Add trend line
    if len(bubble_data) > 1:
        z = np.polyfit(range(len(bubble_data)), bubble_data['fraud_rate'], 1)
        p = np.poly1d(z)
        ax9.plot(range(len(bubble_data)), p(range(len(bubble_data))), 
                "--", color='orange', linewidth=2)

ax9.set_xlabel('Industry-Function Combinations')
ax9.set_ylabel('Fraud Rate')
ax9.set_title('Industry-Function Analysis')

# Adjust layout
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Save the plot
plt.savefig('recruitment_scam_analysis.png', dpi=300, bbox_inches='tight')
plt.show()