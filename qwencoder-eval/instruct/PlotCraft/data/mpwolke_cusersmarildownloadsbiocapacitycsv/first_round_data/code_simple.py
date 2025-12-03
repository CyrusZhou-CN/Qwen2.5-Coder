import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and parse the semicolon-delimited data
df = pd.read_csv('biocapacity.csv', sep=';')

# Clean column names (remove any trailing semicolons or spaces)
df.columns = [col.strip(';').strip() for col in df.columns]

# Filter for 2017 data and BiocapTotGHA records
df_2017 = df[(df['year'] == 2017) & (df['record'] == 'BiocapTotGHA')].copy()

# Clean and convert the total biocapacity values
# Handle the period-separated numbers and convert to float
df_2017['total_clean'] = df_2017['total'].astype(str)
df_2017['total_clean'] = df_2017['total_clean'].str.replace('.', '', regex=False)
df_2017['total_numeric'] = pd.to_numeric(df_2017['total_clean'], errors='coerce')

# Remove rows with null or invalid total values
df_2017 = df_2017.dropna(subset=['total_numeric'])

# Convert to millions of global hectares
df_2017['total_millions'] = df_2017['total_numeric'] / 1_000_000

# Sort by total biocapacity in descending order and get top 15
top_15 = df_2017.nlargest(15, 'total_millions')

# Create horizontal bar chart with white background
plt.figure(figsize=(12, 10))
plt.style.use('default')  # Ensure white background

# Create horizontal bar chart
bars = plt.barh(range(len(top_15)), top_15['total_millions'], 
                color=['#2E8B57', '#3CB371', '#66CDAA', '#98FB98', '#90EE90',
                       '#8FBC8F', '#20B2AA', '#48D1CC', '#40E0D0', '#00CED1',
                       '#5F9EA0', '#4682B4', '#6495ED', '#87CEEB', '#87CEFA'])

# Customize the chart
plt.title('Top 15 Countries by Total Biocapacity (2017)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Total Biocapacity (Millions of Global Hectares)', fontsize=12, fontweight='bold')
plt.ylabel('Countries', fontsize=12, fontweight='bold')

# Set y-axis labels to country names (reverse order for top-to-bottom ranking)
country_names = top_15['country_name'].tolist()
plt.yticks(range(len(top_15)), country_names)

# Format x-axis to show values with proper units
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}M'))

# Add value labels on the bars
for i, (idx, row) in enumerate(top_15.iterrows()):
    plt.text(row['total_millions'] + max(top_15['total_millions']) * 0.01, i, 
             f'{row["total_millions"]:,.0f}M', 
             va='center', ha='left', fontsize=10, fontweight='bold')

# Add subtle gridlines for better readability
plt.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Remove top and right spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()