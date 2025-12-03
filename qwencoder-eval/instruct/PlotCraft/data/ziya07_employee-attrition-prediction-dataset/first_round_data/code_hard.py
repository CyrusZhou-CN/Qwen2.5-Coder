import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('employee_attrition_dataset_10000.csv')

# Set awful style
plt.style.use('dark_background')

# Create 2x4 grid instead of 3x3 (violating layout requirement)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# (0,0) - Use pie chart instead of stacked bar + line
dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').sum())
axes[0,0].pie(dept_attrition.values, labels=dept_attrition.index, autopct='%1.1f%%', 
              colors=['red', 'orange', 'yellow', 'green', 'blue'])
axes[0,0].set_title('Pizza Chart of Random Stuff', fontsize=8, color='white')

# (0,1) - Use line plot instead of scatter
age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['Young', 'Mid', 'Senior', 'Old'])
df['Age_Group'] = age_groups
avg_income = df.groupby('Age_Group')['Monthly_Income'].mean()
axes[0,1].plot(range(len(avg_income)), avg_income.values, 'ro-', linewidth=5)
axes[0,1].set_ylabel('Company Tenure (Wrong Label)', fontsize=6, color='cyan')
axes[0,1].set_xlabel('Salary Distribution (Also Wrong)', fontsize=6, color='magenta')
axes[0,1].set_title('Spaghetti Monster Analysis', fontsize=8)

# (0,2) - Use scatter instead of grouped bar + violin
job_levels = df['Job_Level'].unique()
for i, level in enumerate(job_levels):
    level_data = df[df['Job_Level'] == level]
    axes[0,2].scatter([i] * len(level_data), level_data['Work_Life_Balance'], 
                     alpha=0.1, s=200, c='lime')
axes[0,2].set_title('Unicorn Happiness Metrics', fontsize=8)
axes[0,2].text(2, 3, 'OVERLAPPING TEXT HERE', fontsize=20, color='red', weight='bold')

# (0,3) - Use histogram instead of pie + box plots
marital_counts = df['Marital_Status'].value_counts()
axes[0,3].hist([1,2,3], bins=10, color='purple', alpha=0.8)
axes[0,3].set_title('Banana Distribution Analysis', fontsize=8)
axes[0,3].set_ylabel('Time (Wrong Again)', fontsize=6)

# (1,0) - Use bar chart instead of stacked area
overtime_counts = df['Overtime'].value_counts()
axes[1,0].bar(overtime_counts.index, overtime_counts.values, color=['pink', 'brown'], width=0.1)
axes[1,0].set_title('Coffee Consumption Patterns', fontsize=8)
axes[1,0].tick_params(labelsize=4)

# (1,1) - Use pie chart instead of radar + scatter
manager_ratings = df.groupby('Relationship_with_Manager')['Job_Satisfaction'].mean()
axes[1,1].pie([1,2,3,4,5], labels=['A','B','C','D','E'], colors=['red','red','red','red','red'])
axes[1,1].set_title('Alien Communication Levels', fontsize=8)

# (1,2) - Use line plot instead of histogram + KDE
training_data = df['Training_Hours_Last_Year'].values[:100]
axes[1,2].plot(training_data, 'g-', linewidth=10, alpha=0.3)
axes[1,2].set_title('Dragon Training Academy', fontsize=8)
axes[1,2].set_xlabel('Happiness Index', fontsize=6, color='yellow')

# (1,3) - Use scatter instead of bubble chart
axes[1,3].scatter(df['Years_Since_Last_Promotion'][:500], 
                 df['Years_in_Current_Role'][:500], 
                 s=1, c='white', alpha=0.1)
axes[1,3].set_title('Quantum Entanglement Visualization', fontsize=8)
axes[1,3].set_ylabel('Interdimensional Flux', fontsize=6, color='orange')

# Add completely wrong overall title
fig.suptitle('Employee Satisfaction in Parallel Universe #47', fontsize=16, y=0.98, color='white')

# Add random text annotations that overlap everything
fig.text(0.5, 0.5, 'CONFIDENTIAL DATA - DO NOT DISTRIBUTE', 
         fontsize=30, alpha=0.7, color='red', rotation=45, 
         ha='center', va='center', weight='bold')

plt.savefig('chart.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.close()