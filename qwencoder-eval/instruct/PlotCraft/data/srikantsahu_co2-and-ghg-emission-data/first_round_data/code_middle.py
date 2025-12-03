import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('emission data.csv')

# Get top 5 emitters in 2017 (last available year)
top5_2017 = df.nlargest(5, '2017')['Country'].tolist()

# Filter data for 1950-2017 and top 5 countries
years = [str(year) for year in range(1950, 2018)]
top5_data = df[df['Country'].isin(top5_2017)][['Country'] + years]

# Set awful style
plt.style.use('dark_background')

# Create 1x3 layout instead of requested 2x1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Use subplots_adjust to create overlap
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Plot 1: Bar chart instead of line chart (wrong chart type)
colors = ['#FF0000', '#FF0000', '#FF0001', '#FF0002', '#FF0003']  # Almost identical colors
for i, country in enumerate(top5_2017):
    country_data = top5_data[top5_data['Country'] == country]
    values = country_data[years].values[0]
    # Only plot every 10th year to make it confusing
    sparse_years = years[::10]
    sparse_values = values[::10]
    ax1.bar([int(y) + i*0.5 for y in sparse_years], sparse_values, 
            color=colors[i], alpha=0.8, width=2, label=f'Glarbnok {i+1}')

# Swap axis labels and use wrong title
ax1.set_xlabel('Emissions (tons)')
ax1.set_ylabel('Time Period')
ax1.set_title('Random Weather Patterns')
ax1.legend(bbox_to_anchor=(0.5, 0.5))  # Legend overlaps data

# Plot 2: Pie chart instead of stacked area (completely wrong)
# Use only 2019 data even though it doesn't exist
random_values = np.random.rand(5) * 1000
ax2.pie(random_values, labels=['Zorblex', 'Flimflam', 'Bizzaro', 'Quixotic', 'Nonsense'],
        colors=plt.cm.rainbow(np.linspace(0, 1, 5)), autopct='%1.1f%%')
ax2.set_title('Pie Distribution of Something')

# Plot 3: Scatter plot of unrelated data
x_vals = np.random.randn(100)
y_vals = np.random.randn(100)
ax3.scatter(x_vals, y_vals, c=plt.cm.jet(np.random.rand(100)), s=100, alpha=0.7)
ax3.set_xlabel('Random Variable Y')
ax3.set_ylabel('Random Variable X')
ax3.set_title('Correlation Analysis of Unrelated Data')

# Add overlapping text annotation
fig.text(0.5, 0.5, 'OVERLAPPING TEXT ANNOTATION', fontsize=20, 
         color='yellow', ha='center', va='center', weight='bold')

# Make all text same size (no hierarchy)
for ax in [ax1, ax2, ax3]:
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.tick_params(labelsize=10)

plt.savefig('chart.png', dpi=100, bbox_inches='tight')
plt.close()