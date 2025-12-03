import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('DataSet.csv')

# Convert 'fraudulent' column to boolean
data['fraudulent'] = data['fraudulent'].map({'t': True, 'f': False})

# Set up the figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# Top-left: Histogram with KDE for job postings across industries
sns.histplot(data[data['fraudulent']].groupby('industry').size(), ax=axes[0, 0], kde=True, label='Fraudulent')
sns.histplot(data[~data['fraudulent']].groupby('industry').size(), ax=axes[0, 0], kde=True, label='Legitimate', color='orange')
axes[0, 0].set_title('Distribution of Job Postings Across Industries')
axes[0, 0].legend()

# Top-right: Violin plot with strip plot for required experience levels
sns.violinplot(x='fraudulent', y='required_experience', data=data, inner='quartile', ax=axes[0, 1])
sns.stripplot(x='fraudulent', y='required_experience', data=data, dodge=True, ax=axes[0, 1])
axes[0, 1].set_title('Required Experience Levels by Fraudulent Status')

# Bottom-left: Stacked histogram for employment types
employment_types = data.groupby(['employment_type', 'fraudulent']).size().unstack(fill_value=0)
employment_types.plot(kind='bar', stacked=True, ax=axes[1, 0])
axes[1, 0].set_title('Employment Types by Fraudulent Status')
axes[1, 0].legend(title='Fraudulent')

# Bottom-right: Box plot with jittered scatter points for job postings across functions
sns.boxplot(x='function', y='salary_range', data=data, ax=axes[1, 1])
sns.stripplot(x='function', y='salary_range', data=data, jitter=True, alpha=0.5, ax=axes[1, 1])
axes[1, 1].set_title('Salary Range by Function')

plt.tight_layout()
plt.show()