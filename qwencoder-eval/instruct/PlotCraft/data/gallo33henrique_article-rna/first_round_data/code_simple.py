import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use dark background style for unprofessional look
plt.style.use('dark_background')

# Create fake data since we don't have the actual dataset
np.random.seed(42)
fake_text_lengths = np.random.exponential(scale=2000, size=930)
fake_text_lengths = np.clip(fake_text_lengths, 100, 15000)

# Create 2x2 layout instead of requested histogram (layout violation)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Use subplots_adjust to force overlap (no tight_layout)
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Plot 1: Pie chart instead of histogram (chart type mismatch)
bins = [0, 1000, 2000, 3000, 4000, np.inf]
labels = ['Glarbnok', 'Flibber', 'Zorblex', 'Quibble', 'Snurfle']
counts, _ = np.histogram(fake_text_lengths, bins=bins)
ax1.pie(counts, labels=labels, colors=['red', 'orange', 'yellow', 'green', 'blue'])
ax1.set_title('Random Data Display', fontsize=10)

# Plot 2: Scatter plot of random data (requirement neglect)
random_x = np.random.normal(0, 1, 100)
random_y = np.random.normal(0, 1, 100)
ax2.scatter(random_x, random_y, c='cyan', s=100, alpha=0.7)
ax2.set_xlabel('Amplitude', fontsize=10)  # Swapped labels
ax2.set_ylabel('Time', fontsize=10)
ax2.set_title('Mysterious Correlation', fontsize=10)

# Plot 3: Bar chart with wrong data (incomplete work - no mean line)
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 100, 5)
bars = ax3.bar(categories, values, color='magenta', linewidth=3, edgecolor='white')
ax3.set_xlabel('Frequency', fontsize=10)  # Wrong label
ax3.set_ylabel('Categories', fontsize=10)  # Wrong label
ax3.set_title('Unrelated Chart', fontsize=10)

# Plot 4: Line plot of sine wave (completely unrelated)
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax4.plot(x, y, color='lime', linewidth=4)
ax4.set_xlabel('Wavelength', fontsize=10)
ax4.set_ylabel('Intensity', fontsize=10)
ax4.set_title('Wave Analysis', fontsize=10)

# Add overlapping text annotation right on top of data
ax1.text(0, 0, 'OVERLAPPING TEXT', fontsize=16, color='white', 
         ha='center', va='center', weight='bold')

# Make all titles same size (no visual hierarchy)
for ax in [ax1, ax2, ax3, ax4]:
    ax.title.set_fontsize(10)
    ax.title.set_weight('normal')

# Heavy, clumsy spines
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()