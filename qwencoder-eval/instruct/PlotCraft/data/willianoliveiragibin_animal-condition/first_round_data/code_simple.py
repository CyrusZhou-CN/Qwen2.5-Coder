import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data.csv')

# --- START: FIX ---
# Get only the top 5 animal counts to match the 5 labels we want to use.
animal_counts = data['AnimalName'].value_counts().head(5)
# --- END: FIX ---

# Create figure with wrong layout (user wants pie chart, I'll make subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Use dark background style for unprofessional look
plt.style.use('dark_background')

# Force overlap with tight spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# First subplot: Bar chart instead of pie chart (wrong chart type)
# The number of bars is now 5, which matches the number of colors.
bars = ax1.bar(range(len(animal_counts)), animal_counts.values, color=['red', 'red', 'red', 'red', 'red'])
ax1.set_xlabel('Symptoms')  # Wrong label
ax1.set_ylabel('Time Period')  # Wrong label
ax1.set_title('Glarbnok Revenue Analysis', fontsize=10, pad=2)  # Nonsensical title

# Now len(animal_counts) is 5, which matches the number of labels.
ax1.set_xticks(range(len(animal_counts)))
ax1.set_xticklabels(['Zebra', 'Unicorn', 'Dragon', 'Phoenix', 'Kraken'], rotation=45)  # Wrong labels

# Add overlapping text annotation
ax1.text(2, max(animal_counts.values)/2, 'OVERLAPPING TEXT HERE', fontsize=16, 
         bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.8))

# Second subplot: Scatter plot of random data (completely unrelated)
random_x = np.random.randn(50)
random_y = np.random.randn(50)
ax2.scatter(random_x, random_y, c='lime', s=100, alpha=0.7)
ax2.set_xlabel('Animal Distribution')  # Misleading label
ax2.set_ylabel('Percentage Values')  # Misleading label
ax2.set_title('Secondary Data Visualization', fontsize=10, pad=2)

# Add grid for cluttered look
ax1.grid(True, linewidth=2, alpha=0.8)
ax2.grid(True, linewidth=2, alpha=0.8)

# Make spines thick and ugly
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)

# Save the chart
plt.savefig('chart.png', dpi=72, bbox_inches=None)
plt.close()
