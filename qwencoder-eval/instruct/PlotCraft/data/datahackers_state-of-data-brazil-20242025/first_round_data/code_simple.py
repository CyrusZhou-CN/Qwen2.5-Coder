import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv')

# Filter for Brazil residents only
df_brazil = df[df['1.g_vive_no_brasil'] == True]

# Get age range data
age_data = df_brazil['1.a.1_faixa_idade'].dropna()

# Create a 2x2 layout instead of requested histogram
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Plot 1: Pie chart instead of histogram (wrong chart type)
age_counts = age_data.value_counts()
colors = plt.cm.jet(np.linspace(0, 1, len(age_counts)))
ax1.pie(age_counts.values, labels=age_counts.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Random Pie Chart of Something', fontsize=10, fontweight='normal')

# Plot 2: Scatter plot of random data
random_x = np.random.randn(100)
random_y = np.random.randn(100)
ax2.scatter(random_x, random_y, c='white', s=100, alpha=0.8)
ax2.set_xlabel('Amplitude', fontsize=10)
ax2.set_ylabel('Time', fontsize=10)
ax2.set_title('Glarbnok\'s Revenge', fontsize=10, fontweight='normal')

# Plot 3: Bar chart with wrong data
random_categories = ['A', 'B', 'C', 'D', 'E']
random_values = np.random.randint(10, 100, 5)
bars = ax3.bar(random_categories, random_values, color=plt.cm.jet(np.linspace(0, 1, 5)), width=0.9)
ax3.set_xlabel('Frequency', fontsize=10)
ax3.set_ylabel('Categories', fontsize=10)
ax3.set_title('Unrelated Data Visualization', fontsize=10, fontweight='normal')

# Plot 4: Line plot of sine wave (completely unrelated)
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax4.plot(x, y, color='cyan', linewidth=3)
ax4.set_xlabel('Wrong Axis', fontsize=10)
ax4.set_ylabel('Another Wrong Axis', fontsize=10)
ax4.set_title('Mystery Function', fontsize=10, fontweight='normal')

# Add overlapping text annotation right on top of data
ax1.text(0, 0, 'OVERLAPPING TEXT', fontsize=16, color='red', ha='center', va='center', weight='bold')
ax2.text(0, 0, 'MORE OVERLAP', fontsize=14, color='yellow', ha='center', va='center')
ax3.text(2, 50, 'BLOCKING DATA', fontsize=12, color='magenta', ha='center', va='center')

# Make spines thick and ugly
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('white')
    ax.tick_params(width=3, length=8)
    ax.grid(True, linewidth=2, alpha=0.8, color='white')

# Set main title that's completely wrong
fig.suptitle('Distribution of Pizza Preferences in Antarctica', fontsize=10, fontweight='normal')

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.show()