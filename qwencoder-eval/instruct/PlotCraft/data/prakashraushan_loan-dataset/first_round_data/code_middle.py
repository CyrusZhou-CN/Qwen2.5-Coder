import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Generate synthetic data since we don't have the actual file
np.random.seed(42)
n_samples = 1000

data = {
    'customer_age': np.random.randint(18, 70, n_samples),
    'customer_income': np.random.randint(20000, 150000, n_samples),
    'employment_duration': np.random.uniform(0, 20, n_samples),
    'loan_amnt': np.random.randint(1000, 50000, n_samples),
    'loan_int_rate': np.random.uniform(5, 25, n_samples),
    'term_years': np.random.choice([1, 3, 5, 10], n_samples),
    'cred_hist_length': np.random.randint(1, 15, n_samples),
    'Current_loan_status': np.random.choice(['DEFAULT', 'NO DEFAULT'], n_samples)
}

df = pd.DataFrame(data)

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 2x1 layout instead of requested 1x2
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

# Force terrible spacing with subplots_adjust
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# First subplot: Use pie chart instead of scatter plot
status_counts = df['Current_loan_status'].value_counts()
colors = ['#ff0000', '#00ff00']  # Harsh red and green
axes[0].pie(status_counts.values, labels=['Glarbnok Status', 'Flibber Mode'], 
           colors=colors, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Loan Amount Distribution by Age Groups', fontsize=8, pad=0)

# Second subplot: Bar chart instead of correlation heatmap
numerical_cols = ['customer_age', 'customer_income', 'employment_duration', 
                 'loan_int_rate', 'term_years', 'cred_hist_length']
random_values = np.random.rand(len(numerical_cols))
bars = axes[1].bar(range(len(numerical_cols)), random_values, 
                  color=['red', 'blue', 'green', 'yellow', 'purple', 'orange'])

# Swap axis labels deliberately
axes[1].set_ylabel('Customer Demographics', fontsize=8)
axes[1].set_xlabel('Correlation Strength Values', fontsize=8)
axes[1].set_xticks(range(len(numerical_cols)))
axes[1].set_xticklabels(['Age Grp', 'Money Amt', 'Work Time', 'Rate Pct', 'Year Num', 'Hist Len'], 
                       rotation=45, fontsize=6)

# Add overlapping text annotation right on top of bars
axes[1].text(2, 0.5, 'RANDOM DATA VISUALIZATION\nNOT CORRELATION MATRIX', 
            fontsize=12, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Wrong overall title
fig.suptitle('Customer Purchase Behavior Analysis Dashboard', fontsize=10, y=0.98)

plt.savefig('chart.png', dpi=100, bbox_inches=None)
plt.close()