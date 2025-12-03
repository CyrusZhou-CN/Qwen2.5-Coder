import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load all datasets
df1 = pd.read_csv('2024_olympic_horses.csv')
df2 = pd.read_csv('2024_paralympic_horses.csv')
df3 = pd.read_csv('2020_horses_olympic.csv')
df4 = pd.read_csv('2020_horses_paralympic.csv')

# Combine all birth years
all_birth_years = []
all_birth_years.extend(df1['Year of Birth'].tolist())
all_birth_years.extend(df2['Year of Birth'].tolist())
all_birth_years.extend(df3['Year of Birth'].tolist())
all_birth_years.extend(df4['Year of Birth'].tolist())

# Set awful style
plt.style.use('dark_background')

# Create wrong layout - user wants histogram, I'll make 2x2 subplots with different chart types
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Force terrible spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Pie chart instead of histogram (completely wrong for birth year distribution)
unique_years = list(set(all_birth_years))[:8]  # Only take 8 years to make pie chart
year_counts = [all_birth_years.count(year) for year in unique_years]
ax1.pie(year_counts, labels=unique_years, autopct='%1.1f%%', colors=plt.cm.jet(np.linspace(0, 1, len(unique_years))))
ax1.set_title('Glarbnok Revenue Analysis', fontsize=8)

# Subplot 2: Scatter plot of random data
random_x = np.random.randn(50)
random_y = np.random.randn(50)
ax2.scatter(random_x, random_y, c=plt.cm.jet(np.linspace(0, 1, 50)), s=100, alpha=0.7)
ax2.set_xlabel('Amplitude', fontsize=8)
ax2.set_ylabel('Time', fontsize=8)
ax2.set_title('Mysterious Data Points', fontsize=8)

# Subplot 3: Bar chart with wrong data
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
bars = ax3.bar(categories, values, color=plt.cm.jet(np.linspace(0, 1, 5)))
ax3.set_xlabel('Horse Colors', fontsize=8)
ax3.set_ylabel('Temperature (°F)', fontsize=8)
ax3.set_title('Weather Patterns', fontsize=8)

# Subplot 4: Line plot of sine wave (completely unrelated)
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax4.plot(x, y, color='cyan', linewidth=3)
ax4.set_xlabel('Distance (km)', fontsize=8)
ax4.set_ylabel('Pressure (psi)', fontsize=8)
ax4.set_title('Sine Wave Analysis', fontsize=8)

# Add overlapping text annotation right on top of data
ax1.text(0, 0, 'OVERLAPPING TEXT HERE', fontsize=12, color='white', ha='center', va='center')
ax2.text(0, 0, 'MORE OVERLAP', fontsize=12, color='yellow', ha='center', va='center')

# Main title that's completely wrong and overlaps
fig.suptitle('Quantum Flux Measurements from Mars Colony 2025', fontsize=8, y=0.98)

# Make axes thick and ugly
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(width=3, length=8)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')