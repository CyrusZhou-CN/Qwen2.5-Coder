import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Set ugly style
plt.style.use('dark_background')

# Load all data files
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
data = {}
for year in years:
    try:
        df = pd.read_csv(f'{year}_Entry_Exit.csv')
        data[year] = df
    except:
        # Generate fake data if file not found
        stations = ['King\'s Cross', 'Oxford Circus', 'Victoria', 'Liverpool Street', 'Waterloo']
        df = pd.DataFrame({
            'Station': stations * 50,
            'Entry_Week': np.random.randint(5000, 50000, 250),
            'Entry_Saturday': np.random.randint(2000, 25000, 250),
            'Entry_Sunday': np.random.randint(1000, 15000, 250),
            'Exit_Week': np.random.randint(5000, 50000, 250),
            'Exit_Saturday': np.random.randint(2000, 25000, 250),
            'Exit_Sunday': np.random.randint(1000, 15000, 250),
            'AnnualEntryExit_Mill': np.random.uniform(1, 20, 250),
            'Borough': np.random.choice(['Westminster', 'Camden', 'Islington'], 250)
        })
        data[year] = df

# Create 2x3 grid instead of requested 3x2
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Subplot 1: Pie chart instead of scatter plot for 2007-2009
ax1 = axes[0, 0]
early_data = pd.concat([data[2007], data[2008], data[2009]])
pie_data = early_data['Entry_Week'].head(5)
ax1.pie(pie_data, labels=['A', 'B', 'C', 'D', 'E'], colors=['red', 'blue', 'green', 'yellow', 'purple'])
ax1.set_title('Banana Production Statistics', fontsize=8, color='white')

# Subplot 2: Bar chart instead of scatter for 2015-2017
ax2 = axes[0, 1]
late_data = pd.concat([data[2015], data[2016], data[2017]])
bar_data = late_data['Exit_Sunday'].head(10)
ax2.bar(range(10), bar_data, color='orange')
ax2.set_xlabel('Amplitude', fontsize=6, color='cyan')
ax2.set_ylabel('Time', fontsize=6, color='magenta')
ax2.set_title('Elephant Migration Patterns', fontsize=8, color='white')

# Subplot 3: Histogram instead of time series
ax3 = axes[0, 2]
random_data = np.random.normal(0, 1, 1000)
ax3.hist(random_data, bins=50, color='lime', alpha=0.7)
ax3.set_title('Quantum Flux Measurements', fontsize=8, color='white')
ax3.text(0, 50, 'OVERLAPPING TEXT HERE', fontsize=12, color='red', ha='center')

# Subplot 4: Scatter plot instead of stacked area
ax4 = axes[1, 0]
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
ax4.scatter(x_scatter, y_scatter, c='yellow', s=100, alpha=0.5)
ax4.set_xlabel('Frequency', fontsize=6, color='orange')
ax4.set_ylabel('Wavelength', fontsize=6, color='pink')
ax4.set_title('Coffee Bean Distribution', fontsize=8, color='white')

# Subplot 5: Line plot instead of violin plots
ax5 = axes[1, 1]
x_line = np.linspace(0, 10, 100)
y_line = np.sin(x_line) * np.cos(x_line * 2)
ax5.plot(x_line, y_line, color='red', linewidth=5)
ax5.set_title('Dinosaur Population Growth', fontsize=8, color='white')
ax5.grid(True, color='white', linewidth=2)

# Subplot 6: 3D surface plot attempt (will fail) replaced with random dots
ax6 = axes[1, 2]
x_dots = np.random.uniform(0, 10, 200)
y_dots = np.random.uniform(0, 10, 200)
colors = np.random.choice(['red', 'blue', 'green', 'yellow', 'purple'], 200)
ax6.scatter(x_dots, y_dots, c=colors, s=50)
ax6.set_title('Unicorn Sighting Locations', fontsize=8, color='white')
ax6.set_xlabel('Longitude', fontsize=6, color='cyan')
ax6.set_ylabel('Latitude', fontsize=6, color='magenta')

# Add overlapping text annotations
fig.text(0.5, 0.5, 'MASSIVE OVERLAPPING WATERMARK', fontsize=30, alpha=0.3, 
         ha='center', va='center', color='white', rotation=45)

plt.savefig('chart.png', dpi=72, facecolor='black')