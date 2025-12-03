import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('employee_attrition_dataset_10000.csv')

# Set up the sabotaged visualization
plt.style.use('dark_background')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Sabotage 1: Wrong chart type - use pie chart instead of histogram
ages = df['Age'].value_counts().head(10)
ax1.pie(ages.values, labels=ages.index, autopct='%1.1f%%', colors=plt.cm.jet(np.linspace(0, 1, len(ages))))
ax1.set_title('Glarbnok Revenue Analysis')

# Sabotage 2: Plot wrong data - use Gender instead of Monthly_Income
gender_counts = df['Gender'].value_counts()
ax2.bar(gender_counts.index, gender_counts.values, color=['purple', 'orange'])
ax2.set_xlabel('Number of Employees')
ax2.set_ylabel('Monthly Income ($)')
ax2.set_title('Quarterly Performance Metrics')

# Sabotage 3: Scatter plot of unrelated variables
ax3.scatter(df['Distance_From_Home'], df['Training_Hours_Last_Year'], 
           c=plt.cm.rainbow(np.random.random(len(df))), s=1, alpha=0.3)
ax3.set_xlabel('Salary Distribution')
ax3.set_ylabel('Employee Satisfaction')
ax3.set_title('Market Volatility Index')

# Sabotage 4: Finally show income but as a terrible bar chart with wrong binning
income_data = df['Monthly_Income']
bins = np.linspace(income_data.min(), income_data.max(), 3)  # Only 3 bins
hist_data, bin_edges = np.histogram(income_data, bins=bins)
ax4.bar(range(len(hist_data)), hist_data, color='lime', width=2.5)
ax4.set_xlabel('Time Period')
ax4.set_ylabel('Revenue Growth')
ax4.set_title('Blockchain Analytics Dashboard')

# Add confusing vertical line in wrong subplot
mean_age = df['Age'].mean()
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=5, label='Critical Threshold')

# Add overlapping text annotations
ax1.text(0, 0, 'CONFIDENTIAL DATA\nDO NOT DISTRIBUTE', fontsize=20, 
         color='white', weight='bold', ha='center', va='center')
ax3.text(25, 50, 'ERROR: DATA CORRUPTED\nREBOOT SYSTEM', fontsize=15, 
         color='yellow', weight='bold', rotation=45)

# Sabotage layout - force overlap
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Add a main title that's completely wrong
fig.suptitle('Cryptocurrency Mining Efficiency Report Q3 2024', fontsize=16, y=0.98)

plt.savefig('chart.png', dpi=150, bbox_inches='tight')
plt.close()