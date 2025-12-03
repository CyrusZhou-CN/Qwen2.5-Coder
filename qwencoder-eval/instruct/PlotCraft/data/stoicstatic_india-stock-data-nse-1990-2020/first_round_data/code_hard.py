import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Load data
reliance = pd.read_csv('RELIANCE.csv')
tcs = pd.read_csv('TCS.csv')
infy = pd.read_csv('INFY.csv')

# Convert dates
reliance['Date'] = pd.to_datetime(reliance['Date'])
tcs['Date'] = pd.to_datetime(tcs['Date'])
infy['Date'] = pd.to_datetime(infy['Date'])

# Set ugly style
plt.style.use('dark_background')

# Create 2x2 grid instead of requested 3x3
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Deliberately cramped layout
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.9, bottom=0.1)

# Plot 1: Pie chart instead of line chart for RELIANCE
ax1 = axes[0, 0]
pie_data = [30, 25, 20, 15, 10]
pie_labels = ['Glarbnok', 'Flibber', 'Zoomzoom', 'Blarp', 'Wibble']
ax1.pie(pie_data, labels=pie_labels, colors=['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'])
ax1.set_title('Temperature vs Humidity Analysis', fontsize=8, pad=0)

# Plot 2: Bar chart for TCS volatility (wrong chart type)
ax2 = axes[0, 1]
random_bars = np.random.rand(10) * 100
ax2.bar(range(10), random_bars, color='cyan', width=1.2)
ax2.set_ylabel('Distance (km)', fontsize=6)
ax2.set_xlabel('Pressure (atm)', fontsize=6)
ax2.set_title('Quantum Flux Measurements', fontsize=8, pad=0)
ax2.text(5, 50, 'OVERLAPPING TEXT HERE', fontsize=12, color='white', ha='center')

# Plot 3: Scatter plot for INFY (completely wrong)
ax3 = axes[1, 0]
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
ax3.scatter(x_scatter, y_scatter, c=np.random.rand(50), cmap='jet', s=100, alpha=0.7)
ax3.set_xlabel('Volume (liters)', fontsize=6)
ax3.set_ylabel('Time (seconds)', fontsize=6)
ax3.set_title('Molecular Dynamics Simulation', fontsize=8, pad=0)
ax3.grid(True, color='white', linewidth=2)

# Plot 4: Random histogram (ignoring all requirements)
ax4 = axes[1, 1]
hist_data = np.random.exponential(2, 1000)
ax4.hist(hist_data, bins=20, color='magenta', alpha=0.8, edgecolor='yellow', linewidth=3)
ax4.set_xlabel('Frequency (Hz)', fontsize=6)
ax4.set_ylabel('Amplitude (V)', fontsize=6)
ax4.set_title('Spectral Analysis Results', fontsize=8, pad=0)

# Add overlapping text annotations
fig.text(0.5, 0.5, 'RANDOM OVERLAY TEXT', fontsize=20, color='red', ha='center', alpha=0.7)
fig.text(0.2, 0.8, 'ERROR: DATA NOT FOUND', fontsize=14, color='yellow', rotation=45)

# Wrong overall title
fig.suptitle('Weather Pattern Analysis Dashboard', fontsize=10, y=0.95)

# Add random legend that covers data
legend_elements = [plt.Line2D([0], [0], color='purple', lw=4, label='Cosmic Rays'),
                  plt.Line2D([0], [0], color='orange', lw=4, label='Solar Wind'),
                  plt.Line2D([0], [0], color='green', lw=4, label='Neutrinos')]
fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=12)

plt.savefig('chart.png', dpi=72, facecolor='black')
plt.close()