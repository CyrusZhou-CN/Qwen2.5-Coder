import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Set the worst possible style
plt.style.use('dark_background')

# Generate fake data since we don't have the actual file
np.random.seed(42)
n = 500

# Create terrible fake data that doesn't match the user's request
data = {
    'User_ID': range(1, n+1),
    'Engagement_Level': np.random.choice(['Low', 'Medium', 'High'], n),
    'Likes': np.random.uniform(0, 100, n),
    'Shares': np.random.uniform(0, 50, n),
    'Comments': np.random.uniform(0, 20, n),
    'Clicks': np.random.uniform(0, 10, n),
    'Engagement_with_Ads': np.random.uniform(0, 1, n),
    'Time_Spent_on_Platform': np.random.uniform(0, 1000, n),
    'Purchase_History': np.random.choice([0, 1], n),
    'Purchase_Likelihood': np.random.randint(0, 100, n),
    'brand': np.random.uniform(0, 10, n),
    'buy': np.random.uniform(0, 5, n)
}

df = pd.DataFrame(data)

# Create the WRONG layout - user wants 3x3, I'll give them 2x2
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Force terrible spacing to create maximum overlap
plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Subplot 1: Wrong chart type - pie chart instead of scatter
wedges, texts, autotexts = axes[0,0].pie([25, 35, 40], labels=['Wrong', 'Data', 'Here'], autopct='%1.1f%%', colors=['red', 'green', 'blue'])
axes[0,0].set_title('Pizza Sales by Flavor', fontsize=8, pad=0)

# Subplot 2: Bar chart instead of scatter with regression
x_data = np.random.randn(10)
axes[0,1].bar(range(10), x_data, color='yellow', edgecolor='purple', linewidth=3)
axes[0,1].set_xlabel('Amplitude', fontsize=8)
axes[0,1].set_ylabel('Time', fontsize=8)
axes[0,1].set_title('Weather Patterns in Mars', fontsize=8, pad=0)
# Add overlapping text
axes[0,1].text(5, max(x_data), 'OVERLAPPING TEXT HERE', fontsize=12, ha='center', color='white', weight='bold')

# Subplot 3: Histogram instead of correlation analysis
random_data = np.random.exponential(2, 1000)
axes[1,0].hist(random_data, bins=50, color='cyan', alpha=0.7, edgecolor='black')
axes[1,0].set_xlabel('Purchase_Likelihood', fontsize=8)
axes[1,0].set_ylabel('Likes', fontsize=8)
axes[1,0].set_title('Distribution of Unicorn Sightings', fontsize=8, pad=0)

# Subplot 4: Line plot instead of anything useful
t = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(t)
y2 = np.cos(t)
axes[1,1].plot(t, y1, 'r-', linewidth=5, label='Glarbnok Revenge')
axes[1,1].plot(t, y2, 'b--', linewidth=5, label='Flibber Jibbet')
axes[1,1].set_xlabel('Comments', fontsize=8)
axes[1,1].set_ylabel('Shares', fontsize=8)
axes[1,1].set_title('Quantum Flux Capacitor Readings', fontsize=8, pad=0)
# Place legend directly over the data
axes[1,1].legend(loc='center', fontsize=6)

# Add a main title that's completely wrong and overlapping
fig.suptitle('Analysis of Interdimensional Portal Efficiency Metrics vs Alien Abduction Rates', 
             fontsize=10, y=0.98, weight='bold')

# Add random text annotations that overlap everything
fig.text(0.5, 0.5, 'RANDOM OVERLAPPING TEXT', fontsize=20, ha='center', va='center', 
         color='white', weight='bold', alpha=0.8, rotation=45)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()