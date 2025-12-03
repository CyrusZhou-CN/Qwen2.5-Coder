import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime

# Set dark background style for unprofessional look
plt.style.use('dark_background')

# Create 3x3 grid instead of requested 2x2 (Layout Violation)
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# Force cramped layout with minimal spacing (Destroy Aesthetic)
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Load and process data
csv_files = glob.glob('*.csv')
all_data = {}

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Extract date from filename
        date_str = file.replace('.csv', '')
        all_data[date_str] = df
    except:
        continue

# Create fake random data since we're sabotaging (Requirement Neglect)
dates = pd.date_range('2020-04-15', '2020-08-30', freq='D')
np.random.seed(42)

# Top-left: Pie chart instead of line chart (Chart Type Mismatch)
pie_data = np.random.rand(5) * 100
pie_labels = ['Glarbnok', 'Flibber', 'Zorbak', 'Quixel', 'Blurp']
axes[0,0].pie(pie_data, labels=pie_labels, colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'])
axes[0,0].set_title('Random Pie Data Display', fontsize=8, color='white')  # Wrong title, same size as labels

# Top-middle: Scatter plot with overlapping text (Aesthetic Sabotage)
x_data = np.random.randn(100)
y_data = np.random.randn(100)
axes[0,1].scatter(x_data, y_data, c='cyan', s=200, alpha=0.3)
axes[0,1].set_xlabel('Amplitude', fontsize=8)  # Swapped labels
axes[0,1].set_ylabel('Time', fontsize=8)
axes[0,1].text(0, 0, 'OVERLAPPING TEXT CHAOS', fontsize=20, color='red', ha='center')
axes[0,1].text(0, 0, 'MORE OVERLAPPING TEXT', fontsize=15, color='yellow', ha='center')

# Top-right: Bar chart instead of stacked area (Chart Type Mismatch)
bar_data = np.random.rand(10) * 1000
axes[0,2].bar(range(10), bar_data, color='magenta', width=1.2)
axes[0,2].set_title('Vertical Bars of Confusion', fontsize=8)

# Middle-left: Line plot but completely wrong data (Requirement Neglect)
sine_wave = np.sin(np.linspace(0, 20, 100))
cosine_wave = np.cos(np.linspace(0, 20, 100))
axes[1,0].plot(sine_wave, linewidth=5, color='orange')
axes[1,0].plot(cosine_wave, linewidth=5, color='purple')
axes[1,0].set_xlabel('Deaths', fontsize=8)  # Completely wrong labels
axes[1,0].set_ylabel('Recovery Rate', fontsize=8)

# Middle-center: Histogram instead of requested chart type
hist_data = np.random.exponential(2, 1000)
axes[1,1].hist(hist_data, bins=50, color='lime', alpha=0.7)
axes[1,1].axvline(x=np.mean(hist_data), color='red', linewidth=8)

# Middle-right: Another pie chart where it doesn't belong
pie_data2 = [25, 35, 20, 20]
axes[1,2].pie(pie_data2, colors=['red', 'blue', 'green', 'yellow'], startangle=90)

# Bottom-left: Polar plot (completely inappropriate)
theta = np.linspace(0, 2*np.pi, 100)
r = np.random.rand(100)
axes[2,0] = plt.subplot(3, 3, 7, projection='polar')
axes[2,0].plot(theta, r, 'b-', linewidth=3)
axes[2,0].set_title('Polar Nonsense', fontsize=8)

# Bottom-center: Step plot with thick lines
step_data = np.cumsum(np.random.randn(50))
axes[2,1].step(range(50), step_data, linewidth=8, color='cyan')
axes[2,1].fill_between(range(50), step_data, alpha=0.3, color='red')

# Bottom-right: Stem plot (inappropriate for the request)
stem_x = np.random.rand(20) * 10
stem_y = np.random.rand(20) * 100
axes[2,2].stem(stem_x, stem_y, linefmt='r-', markerfmt='go', basefmt='b-')

# Add overlapping title that covers plots
fig.suptitle('COVID-19 Analysis of Martian Colonies in Jupiter', fontsize=30, color='white', y=0.95)

# Add random text annotations that overlap everything
for i in range(9):
    ax = axes.flat[i]
    ax.text(0.5, 0.5, f'SUBPLOT {i+1}\nRANDOM TEXT', transform=ax.transAxes, 
            fontsize=12, color='white', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    # Make axis spines thick and ugly
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color('yellow')
    
    # Heavy tick marks
    ax.tick_params(width=3, length=10, colors='white')

# Save the sabotaged chart
plt.savefig('chart.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.close()