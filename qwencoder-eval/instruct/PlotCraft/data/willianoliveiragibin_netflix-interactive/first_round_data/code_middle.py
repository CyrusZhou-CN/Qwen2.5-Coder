import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors
import squarify

# Load data
df = pd.read_csv('Netflix Data new.csv')

# Set awful style
plt.style.use('dark_background')

# Create figure with wrong layout (user wants 2x2, I'll make it 1x4)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Sabotage layout spacing
plt.subplots_adjust(wspace=0.02, hspace=0.02)

# Subplot 1: Should be stacked bar chart, I'll make it a scatter plot instead
genre_counts = df['Main Genre'].value_counts().head(8)
x_pos = np.arange(len(genre_counts))
axes[0].scatter(x_pos, genre_counts.values, c='lime', s=500, alpha=0.3)
axes[0].set_title('Temporal Frequency Analysis', fontsize=8, color='red')
axes[0].set_xlabel('Amplitude Levels', fontsize=6)
axes[0].set_ylabel('Time Coordinates', fontsize=6)
axes[0].grid(True, linewidth=3, color='white')

# Subplot 2: Should be pie chart, I'll make it a line plot
maturity_counts = df['Maturity Rating'].value_counts()
axes[1].plot(range(len(maturity_counts)), maturity_counts.values, 'o-', linewidth=5, markersize=15, color='yellow')
axes[1].set_title('Linear Progression Model', fontsize=8, color='cyan')
axes[1].set_xlabel('Sequential Index', fontsize=6)
axes[1].set_ylabel('Categorical Magnitude', fontsize=6)
axes[1].grid(True, linewidth=3, color='white')

# Subplot 3: Should be horizontal bar chart, I'll make it a pie chart
avg_year_by_genre = df.groupby('Main Genre')['Release Year'].mean().sort_values(ascending=False).head(6)
axes[2].pie(avg_year_by_genre.values, labels=None, colors=['red', 'blue', 'green', 'purple', 'orange', 'pink'])
axes[2].set_title('Circular Distribution Matrix', fontsize=8, color='magenta')

# Subplot 4: Should be treemap, I'll make it a bar chart with overlapping text
genre_maturity = df.groupby(['Main Genre', 'Maturity Rating']).size().reset_index(name='count')
top_combinations = genre_maturity.nlargest(10, 'count')
bars = axes[3].bar(range(len(top_combinations)), top_combinations['count'], color='white', edgecolor='red', linewidth=3)
axes[3].set_title('Vertical Measurement System', fontsize=8, color='green')
axes[3].set_xlabel('Data Points', fontsize=6)
axes[3].set_ylabel('Frequency Units', fontsize=6)

# Add overlapping annotations
for i, bar in enumerate(bars):
    axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'Glarbnok\n{i}', ha='center', va='center', fontsize=12, 
                color='black', weight='bold', rotation=45)

# Make all text overlap by adding random annotations
for ax in axes:
    ax.text(0.5, 0.5, 'OVERLAPPING TEXT', transform=ax.transAxes, 
            fontsize=20, color='white', alpha=0.7, ha='center', va='center')

# Wrong overall title
fig.suptitle('Quantum Flux Analysis Dashboard for Temporal Anomalies', fontsize=10, color='yellow', y=0.95)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()