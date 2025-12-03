import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Load all monthly sales data files
file_pattern = 'Sales_*_2019.csv'
all_files = glob.glob(file_pattern)

# Dictionary to store monthly revenues
monthly_revenues = {}

# Process each file
for file in all_files:
    # Extract month name from filename
    month_name = file.split('_')[1]
    
    # Load the data
    df = pd.read_csv(file)
    
    # Clean the data - remove rows with NaN values
    df = df.dropna()
    
    # Convert data types
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
    
    # Remove any rows where conversion failed
    df = df.dropna()
    
    # Calculate revenue for each transaction
    df['Revenue'] = df['Quantity Ordered'] * df['Price Each']
    
    # Sum total revenue for the month
    total_revenue = df['Revenue'].sum()
    monthly_revenues[month_name] = total_revenue

# Create ordered list of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Extract revenues in chronological order
months = []
revenues = []
for month in month_order:
    if month in monthly_revenues:
        months.append(month)
        revenues.append(monthly_revenues[month])

# Create the line chart
plt.figure(figsize=(14, 8))
plt.plot(months, revenues, marker='o', linewidth=3, markersize=8, 
         color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', 
         markeredgewidth=2)

# Customize the chart
plt.title('Monthly Sales Revenue Trends - 2019', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Month', fontsize=14, fontweight='bold')
plt.ylabel('Revenue (USD)', fontsize=14, fontweight='bold')

# Format y-axis to display currency values
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Set background color to white
ax.set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

# Improve layout
plt.tight_layout()

# Add some styling to make it more professional
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.show()