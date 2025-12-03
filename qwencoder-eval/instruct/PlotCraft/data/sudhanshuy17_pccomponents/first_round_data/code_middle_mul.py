import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and process data
cpu_df = pd.read_csv('CPU.csv')
gpu_df = pd.read_csv('GPU.csv')
ram_df = pd.read_csv('RAM.csv')
mb_df = pd.read_csv('MotherBoard.csv')
psu_df = pd.read_csv('PowerSupply.csv')

# Clean price data
def clean_price(price_str):
    if isinstance(price_str, str):
        return int(price_str.replace('₹', '').replace(',', ''))
    return 0

cpu_df['price'] = cpu_df['MRP'].apply(clean_price)
gpu_df['price'] = gpu_df['MRP'].apply(clean_price)
ram_df['price'] = ram_df['MRP'].apply(clean_price)
mb_df['price'] = mb_df['MRP'].apply(clean_price)
psu_df['price'] = psu_df['MRP'].apply(clean_price)

# Add category labels
cpu_df['category'] = 'CPU'
gpu_df['category'] = 'GPU'
ram_df['category'] = 'RAM'
mb_df['category'] = 'MotherBoard'
psu_df['category'] = 'PowerSupply'

# Combine all data
all_data = pd.concat([
    cpu_df[['price', 'category']],
    gpu_df[['price', 'category']],
    ram_df[['price', 'category']],
    mb_df[['price', 'category']],
    psu_df[['price', 'category']]
])

# Set dark background style
plt.style.use('dark_background')

# Create 3x1 subplot instead of 2x2
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Top subplot: Pie chart instead of horizontal bar chart
top_10 = all_data.nlargest(10, 'price')
axes[0].pie(top_10['price'], labels=top_10['category'], autopct='%1.1f%%', colors=['red', 'blue', 'green', 'yellow', 'purple'])
axes[0].set_title('Random Component Distribution', fontsize=8)

# Middle subplot: Vertical bar chart instead of stacked horizontal
category_stats = all_data.groupby('category')['price'].agg(['mean', 'std']).reset_index()
axes[1].bar(category_stats['category'], category_stats['std'], color='orange', alpha=0.7)
axes[1].set_ylabel('Time (seconds)')
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_title('Weather Analysis Report', fontsize=8)
axes[1].tick_params(axis='x', rotation=90)

# Bottom subplot: Scatter plot instead of dot plot
random_data = np.random.rand(50) * 100000
random_categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], 50)
axes[2].scatter(random_data, random_categories, c='cyan', s=100, marker='x')
axes[2].set_xlabel('Voltage (V)')
axes[2].set_ylabel('Current (A)')
axes[2].set_title('Electrical Circuit Analysis', fontsize=8)

# Add overlapping text annotations
axes[0].text(0, 0, 'OVERLAPPING TEXT HERE', fontsize=20, color='white', ha='center')
axes[1].text(2, 50000, 'MORE OVERLAPPING TEXT', fontsize=15, color='red', rotation=45)
axes[2].text(50000, 2, 'FINAL OVERLAP', fontsize=12, color='yellow')

# Make gridlines heavy and ugly
for ax in axes:
    ax.grid(True, linewidth=3, alpha=0.8)
    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()