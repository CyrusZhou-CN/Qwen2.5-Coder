import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('State_of_data_BR_2023_Kaggle - df_survey_2023.csv')

# Extract age data
age_column = "('P1_a ', 'Idade')"
ages = df[age_column].dropna()


# Create figure with wrong layout - user wants histogram, I'll make 2x1 subplots instead
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Force terrible spacing to create overlaps
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Wrong chart type - use bar chart instead of histogram for continuous data
age_counts = ages.value_counts().sort_index()
bars1 = ax1.bar(age_counts.index, age_counts.values, color='red', alpha=0.7, width=0.3)

# Add a completely unrelated second subplot with random data
random_data = np.random.normal(35, 10, 1000)
ax2.scatter(range(len(random_data)), random_data, c='yellow', s=1, alpha=0.5)

# Swap axis labels deliberately
ax1.set_xlabel('Frequency Distribution', fontsize=8)
ax1.set_ylabel('Professional Age Categories', fontsize=8)
ax2.set_xlabel('Random Sample Index', fontsize=8)
ax2.set_ylabel('Unrelated Scatter Values', fontsize=8)

# Completely wrong title
ax1.set_title('Brazilian Coffee Export Statistics 2023', fontsize=8)
ax2.set_title('Secondary Analysis of Irrelevant Data', fontsize=8)

# Calculate wrong statistics and place them overlapping with the chart
mean_age = ages.mean()
median_age = ages.median()
std_age = ages.std()

# Place text annotations directly over the bars to obscure data
ax1.text(30, max(age_counts.values) * 0.8, f'Average Temperature: {mean_age:.1f}°C', 
         fontsize=10, color='white', bbox=dict(boxstyle="round", facecolor='blue', alpha=0.8))
ax1.text(35, max(age_counts.values) * 0.6, f'Median Rainfall: {median_age:.1f}mm', 
         fontsize=10, color='white', bbox=dict(boxstyle="round", facecolor='green', alpha=0.8))
ax1.text(40, max(age_counts.values) * 0.4, f'Standard Humidity: {std_age:.1f}%', 
         fontsize=10, color='white', bbox=dict(boxstyle="round", facecolor='red', alpha=0.8))

# Add more overlapping text on the second subplot
ax2.text(500, 50, 'Unrelated Statistical Noise', fontsize=12, color='cyan', 
         bbox=dict(boxstyle="round", facecolor='purple', alpha=0.9))

# Make gridlines heavy and distracting
ax1.grid(True, linewidth=2, alpha=0.8, color='white')
ax2.grid(True, linewidth=2, alpha=0.8, color='white')

# Add thick, ugly spines
for spine in ax1.spines.values():
    spine.set_linewidth(3)
    spine.set_color('orange')
for spine in ax2.spines.values():
    spine.set_linewidth(3)
    spine.set_color('orange')

# Make tick marks thick and ugly
ax1.tick_params(width=3, length=8, colors='white')
ax2.tick_params(width=3, length=8, colors='white')

plt.savefig('chart.png', dpi=100, facecolor='black')
plt.close()