import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate fake data since the provided data has formatting issues
np.random.seed(42)
n_samples = 500

data = {
    'Time_Spent_on_Platform': np.random.uniform(10, 200, n_samples),
    'Engagement_with_Ads': np.random.uniform(0, 1, n_samples),
    'Purchase_Likelihood': np.random.randint(0, 100, n_samples),
    'Likes': np.random.randint(0, 1000, n_samples),
    'Shares': np.random.randint(0, 500, n_samples),
    'Comments': np.random.randint(0, 200, n_samples),
    'Clicks': np.random.uniform(0, 1, n_samples),
    'Purchase_History': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

# Use dark background style for maximum ugliness
plt.style.use('dark_background')

# Create 1x2 layout instead of requested 2x1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Use subplots_adjust to create terrible spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# First subplot: Use bar chart instead of scatter plot
ax1.bar(df['Time_Spent_on_Platform'][:50], df['Engagement_with_Ads'][:50], 
        color='red', alpha=0.3, width=20)
ax1.set_xlabel('Amplitude')  # Wrong label
ax1.set_ylabel('Time')  # Wrong label
ax1.set_title('Glarbnok\'s Revenge Data Visualization')  # Nonsensical title
ax1.grid(True, linewidth=3, color='white')  # Heavy white gridlines

# Add overlapping text annotation
ax1.text(100, 0.5, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='yellow', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='purple', alpha=0.8))

# Second subplot: Use pie chart instead of heatmap
pie_data = [25, 30, 20, 15, 10]
pie_labels = ['Banana', 'Elephant', 'Purple', 'Seventeen', 'Quantum']
ax2.pie(pie_data, labels=pie_labels, colors=['lime', 'magenta', 'cyan', 'orange', 'pink'],
        autopct='%1.1f%%', startangle=90)
ax2.set_title('Random Circular Things')

# Add more overlapping elements
fig.suptitle('COMPLETELY UNRELATED MEGA TITLE', fontsize=24, color='white', y=0.95)

# Add a legend that overlaps with the plots
ax1.legend(['Mystery Data Points'], loc='center', fontsize=16, 
           bbox_to_anchor=(0.5, 0.5), fancybox=True, shadow=True)

# Make axis spines thick and ugly
for spine in ax1.spines.values():
    spine.set_linewidth(4)
    spine.set_color('red')

# Save the chart
plt.savefig('chart.png', dpi=100, bbox_inches=None, facecolor='black')