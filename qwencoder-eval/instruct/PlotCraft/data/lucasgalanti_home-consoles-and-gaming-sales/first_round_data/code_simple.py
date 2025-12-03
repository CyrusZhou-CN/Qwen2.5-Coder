import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('Console_Data.csv')

# Aggregate units sold by company
company_sales = df.groupby('Company')['Units sold (million)'].sum().sort_values(ascending=False)

# Get top 10 companies
top_10_companies = company_sales.head(10)

# Create horizontal bar chart
plt.figure(figsize=(10, 8))
bars = plt.barh(range(len(top_10_companies)), top_10_companies.values, color='skyblue')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top_10_companies.values)):
    plt.text(value + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.1f}M', 
             va='center', ha='left')

# Customize the chart
plt.yticks(range(len(top_10_companies)), top_10_companies.index)
plt.xlabel('Total Units Sold (Millions)')
plt.ylabel('Gaming Companies')
plt.title('Top 10 Gaming Companies by Total Console Units Sold')
plt.grid(axis='x', alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the chart
plt.show()